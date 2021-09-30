import os
import logging
import argparse
import traceback
import signal

from pathlib import Path
from typing import Union, List

from tqdm import tqdm
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import corner

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# data distributed parallel
import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import bilby
from pycbc.catalog import Merger

# local imports
from models import flows

from gwpe.pytorch.datasets import BasisCoefficientsDataset

def setup_nccl(rank, world_size):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(0, 180))

def cleanup_nccl():
    distributed.destroy_process_group()

def generate_gw150914_context(
    n: int,
    noise_dir: Union[str, os.PathLike],
    psd_dir: Union[str, os.PathLike],
    basis_dir: Union[str, os.PathLike],
    static_args_ini: str,
    event_args_ini: str='gwpe/config_files/event_args.ini',
    ifos: List[str]=['H1','L1'],
    verbose: bool=False,
):
    """Function loads GW150914 segment from O1 dataset, applies signal processing steps
    such as whitening and low-pass filtering, then projects to reduced basis coefficients."""
    from gwpe.utils import read_ini_config
    from gwpe.noise import NoiseTimeline, get_tukey_window, load_psd_from_file
    from gwpe.basis import SVDBasis

    _, static_args = read_ini_config(static_args_ini)
    _, event_args = read_ini_config(event_args_ini)

    basis = SVDBasis(basis_dir, static_args_ini, ifos, preload=False)
    basis.load(time_translations=False, verbose=verbose)
    if n is not None: basis.truncate(n)

    # get GW150914 Test data
    timeline = NoiseTimeline(data_dir=noise_dir, ifos=ifos)
    strains = timeline.get_strains(
        int(Merger('GW150914').time - event_args['seconds_before_event']),
        int(event_args['waveform_length'])
    )

    psds = {}
    for ifo in ifos:
        # coloured noise from psd
        psd_file = Path(psd_dir) / f'{ifo}_PSD.npy'
        assert psd_file.is_file(), f"{psd_file} does not exist."
        psds[ifo] = load_psd_from_file(psd_file, delta_f=event_args['delta_f'])

    event_strain = np.empty((1, len(ifos), static_args['fd_length']), dtype=np.complex64)
    for i, ifo in enumerate(strains):
        # whiten with settings associated to longer strain
        strain = strains[ifo] * get_tukey_window(event_args['sample_length'])  # hann window
        strain = strain.to_frequencyseries(delta_f=event_args['delta_f'])  # fft
        strain /= psds[ifo]**0.5    # whiten
        strain[:int(event_args['f_lower'] / event_args['delta_f'])] = 0.  # lowpass filter
        
        # time slice to specified length and reset x axis to new time 
        event_time = (event_args['seconds_before_event'] + Merger('GW150914').time % 1)
        start_time = event_time - static_args['seconds_before_event']
        end_time = start_time + (static_args['td_length'] * static_args['delta_t'])
        strain.start_time = 0.
        
        strain = strain.to_timeseries()  # ifft
        strain = strain.time_slice(start_time, end_time)
        
        # save frequency domain waveform for values up to 1024Hz
        event_strain[:, i] = strain.to_frequencyseries(delta_f=static_args['delta_f'])[:static_args['fd_length']]

    coefficients = np.stack([(event_strain[:, i, :] @ basis.V[i]) for i in range(event_strain.shape[1])], axis=1)
    
    # flatten for 1-d residual network input
    coefficients = np.concatenate([coefficients.real, coefficients.imag], axis=1)
    coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])
    
    return coefficients

def tensorboard_writer(
    queue: mp.Queue,
    log_dir: str,
    parameters: List[str],
    labels: List[str],
):
    if log_dir is None:
        tb = SummaryWriter()
    else:
        tb = SummaryWriter(log_dir)

    # bilby setup - specify the output directory and the name of the bilby run
    result = bilby.result.read_in_result(
        outdir='../gravflows/bilby_runs/GW150914',
        label='GW150914'
    )

    bilby_parameters = [
        'mass_1', 'mass_2', 'phase', 'geocent_time',
        'luminosity_distance', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
        'phi_12', 'phi_jl', 'theta_jn', 'psi', 'ra', 'dec'
    ]
    bilby_samples = result.posterior[bilby_parameters].values

    # # Shift the time of coalescence by the trigger time
    bilby_samples[:,3] = bilby_samples[:,3] - Merger('GW150914').time

    bilby_df = pd.DataFrame(bilby_samples.astype(np.float32), columns=bilby_parameters)
    bilby_df = bilby_df.rename(columns={'luminosity_distance': 'distance', 'geocent_time': 'time'})
    bilby_df = bilby_df.loc[:, parameters]
    
    domain = [
        [25, 60],  # mass 1
        [15, 50],  # mass 2
        [0, 2*np.pi],  # phase 
        [0,1],  # a_1
        [0,1],  # a 2
        [0,np.pi],  # tilt 1
        [0,np.pi],  # tilt 2
        [0, 2*np.pi],  # phi_12
        [0, 2*np.pi],  # phi_jl
        [0,np.pi],  # theta_jn
        [0,np.pi],  # psi
        [0.4,3.4],  # ra
        [-np.pi/2,.1],  # dec
        # [0.005,0.055],  # tc
        [100,800],  # distance
    ]

    cosmoprior = bilby.gw.prior.UniformSourceFrame(
        name='luminosity_distance',
        minimum=1e2,
        maximum=1e3,
    )

    while True:
        try:
            epoch, scalars, samples = queue.get()
            
            for key, value in scalars.items():
                tb.add_scalar(key, value, epoch)
            
            if samples is not None:
                assert isinstance(samples, torch.Tensor)
                samples_df = pd.DataFrame(samples.numpy(), columns=parameters)
                weights = cosmoprior.prob(samples_df['distance'])
                weights = weights / np.mean(weights)

                fig = corner.corner(
                    bilby_df,
                    levels=[0.5, 0.9],
                    scale_hist=True,
                    plot_datapoints=False,
                    labels=labels,
                    color='red',
                )

                corner.corner(
                    samples_df,
                    levels=[0.5, 0.9],
                    scale_hist=True,
                    plot_datapoints=False,
                    labels=labels,
                    show_titles=True,
                    fontsize=18,
                    fig=fig,
                    range=domain,
                    weights=weights * len(bilby_samples) / len(samples_df),
                )

                fig.suptitle('GW150914 Parameter Estimation: NSF (black) vs. Bilby (red)', fontsize=22)
                tb.add_figure('posteriors/GW150914', fig, epoch)

            tb.flush()

        except Exception as e:
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGSTOP)

# Training with PyTorch native DataDistributedParallel (DDP)
def train(
    rank: int,
    world_size: int,
    lr: float=5e-4,
    batch_size: int=1000,
    epochs: int=500,
    interval: int=10,
    save: int=100,
    num_workers: int=4,
    num_basis: int=100,
    verbose: bool=False,
    profile: bool=False,
):
    assert 0 < batch_size, "batch_size must be a positive integer."
    assert 0 < epochs, "epochs must be a positive integer."
    assert (0 <= interval) and (interval <= epochs), "Interval must be a non-negative integer between 0 and epochs."
    assert (0 <= save) and (save <= epochs), "Save must be a non-negative integer between 0 and epochs."

    # setup data distributed parallel training
    setup_nccl(rank, world_size)  # world size is total gpus
    torch.cuda.set_device(rank)  # rank is gpu index

    # directories
    data_dir = Path('/mnt/datahole/daniel/gravflows/datasets/train/')
    log_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{os.uname().nodename}"

    save_dir = Path('gwpe/model_weights/')
    experiment_dir = save_dir / log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # config files
    waveform_params_ini = str(data_dir / 'config_files/parameters.ini')
    static_args_ini = str(data_dir / 'config_files/static_args.ini')

    # validation
    noise_dir = Path('/mnt/datahole/daniel/gwosc/O1')
    psd_dir = Path("/mnt/datahole/daniel/gravflows/datasets/train/PSD/")
    basis_dir = Path('/mnt/datahole/daniel/gravflows/datasets/basis/')

    # generate gw150914 sample as reduced basis - to do: implement on rank 0 then distribute
    gw150914_context = generate_gw150914_context(num_basis, noise_dir, psd_dir, basis_dir, static_args_ini)
    gw150914_context = torch.tensor(gw150914_context, device=rank)
    gw150914_samples = None  # every "interval" we generate samples for vis, else None

    # training data
    dataset = BasisCoefficientsDataset(
        data_dir=data_dir,
        static_args_ini=static_args_ini,
        parameters_ini=waveform_params_ini,
    )

    sampler = DistributedSampler(
        dataset,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
        seed=rank,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=dataset._worker_init_fn,
        collate_fn=dataset._collate_fn,
    )

    # validation data
    val_dataset = BasisCoefficientsDataset(
        data_dir=data_dir,
        static_args_ini=static_args_ini,
        parameters_ini=waveform_params_ini,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
        seed=rank,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=val_dataset._worker_init_fn,
        collate_fn=val_dataset._collate_fn,
    )

    # set torch profiling runs
    # wait = 1  # ignore first batch
    # warmup = 1
    # active = 4
    # repeat = 2

    # tensorboard
    if rank == 0:
        # tb = SummaryWriter(f'gwpe/runs/{log_dir}')
        queue = mp.SimpleQueue()
        tb_process = mp.Process(
            target=tensorboard_writer,
            args=(queue, f'gwpe/runs/{log_dir}', dataset.generator.parameters, dataset.generator.latex)
        )
        tb_process.start()

    # instantiate neural spline coupling flow
    flow = flows.create_NDE_model(
        input_dim=14,  # we do not predict coalescence time 
        context_dim=4*num_basis,
        num_flow_steps=20,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 5,
            'activation': 'elu',
        }
    )

    # sync_bn_flow = nn.SyncBatchNorm.convert_sync_batchnorm(flow)
    flow = DDP(flow.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # run training loop
    flow.train()
    train_loss = torch.zeros((1,), device=rank, requires_grad=False)
    val_loss = torch.zeros((1,), device=rank, requires_grad=False)

    disable_pbar = False if verbose and (rank == 0) else True  # tqdm progress bar
    with tqdm(total=len(dataloader)*epochs, disable=disable_pbar) as progress:
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'gwpe/runs/{log_dir}'),
        #     record_shapes=True,
        #     with_stack=True
        # ) as profiler:

        for epoch in range(1, 1+epochs):
            if rank == 0:
                progress.set_postfix({'epoch': epoch})
                progress.set_description(f'[{log_dir}] Training', refresh=True)

            # let all processes sync up before starting with a new epoch of training
            distributed.barrier()
            iterator = iter(enumerate(dataloader))
            step, (coefficients, parameters) = next(iterator)
            
            coefficients = coefficients.to(rank, non_blocking=True)
            parameters = parameters.to(rank, non_blocking=True)

            complete = False
            while not complete:
                optimizer.zero_grad()

                # if profile:
                    # https://github.com/guyang3532/kineto/blob/readme/tb_plugin/docs/gpu_utilization.md
                    ## WARNING: profiler may not handle async pinned memory transfer properly?
                    # i.e. may not record CPU vs GPU wall times correctly
                    # may be related to reported blocks per SM/achieved occupancy negative bug
                    # this was an open issue for pytorch 1.9 as of july 9 - nightly may fix it
                    # https://github.com/pytorch/kineto/issues/325#issuecomment-869362218
                    # if (step >= (wait + warmup + active) * repeat):
                    #     break

                # negative log-likelihood conditional on strain over mini-batch
                loss = -flow.module.log_prob(parameters, context=coefficients).mean()

                try:
                    # async get data from CPU and move to GPU during model forward
                    step, (coefficients, parameters) = next(iterator)
                    
                    coefficients = coefficients.to(rank, non_blocking=True)
                    parameters = parameters.to(rank, non_blocking=True)

                except StopIteration:
                    # exit while loop if iterator is complete
                    complete = True
                    
                loss.backward()
                optimizer.step()
                scheduler.step()
                # if profile: profiler.step()

                # total loss summed over each sample in batch
                train_loss += loss.detach() * coefficients.shape[0]
                if rank == 0: progress.update(1)


            # gather total loss during epoch between each GPU worker as list of tensors
            world_loss = [torch.ones_like(train_loss) for _ in range(world_size)]
            distributed.all_gather(world_loss, train_loss)
            train_loss *= 0.0  # reset loss for next epoch
            
            if (interval != 0) and (epoch % interval == 0):
                if rank==0:
                    progress.set_description(f'[{log_dir}] Validating', refresh=True)

                with torch.no_grad():
                    # evaluate model on validation dataset

                    iterator = iter(enumerate(val_loader))
                    step, (coefficients, parameters) = next(iterator)
                    
                    coefficients = coefficients.to(rank, non_blocking=True)
                    parameters = parameters.to(rank, non_blocking=True)

                    complete = False
                    while not complete:

                        # negative log-likelihood conditional on strain over mini-batch
                        loss = -flow.module.log_prob(parameters, context=coefficients).mean()

                        try:
                            # async get data from CPU and move to GPU during model forward
                            step, (coefficients, parameters) = next(iterator)
                            
                            coefficients = coefficients.to(rank, non_blocking=True)
                            parameters = parameters.to(rank, non_blocking=True)

                        except StopIteration:
                            # exit while loop if iterator is complete
                            complete = True

                        # total loss summed over each sample in batch
                        val_loss += loss.detach() * coefficients.shape[0]

                    # gather total loss during epoch between each GPU worker as list of tensors
                    world_val_loss = [torch.ones_like(val_loss) for _ in range(world_size)]
                    distributed.all_gather(world_val_loss, val_loss)
                    val_loss *= 0.0  # reset loss for next epoch

                # validation posteriors
                if rank==0:
                    progress.set_description(f'[{log_dir}] Sampling posteriors', refresh=True)

                samples = flows.sample_flow(
                    flow.module,
                    n=5000,
                    context=gw150914_context,
                    output_device='cuda',
                    dtype=torch.float32,
                )[0]

                # gather samples from all gpus
                gw150914_samples = [torch.ones_like(samples) for _ in range(world_size)]
                distributed.all_gather(gw150914_samples, samples)

        if (rank == 0) and not profile:
            progress.set_description(f'[{log_dir}] Sending to TensorBoard', refresh=True)
            
            scalars = {
                'loss/train': torch.cat(world_loss).sum().item() / len(dataloader.dataset)
            } 
            
            if (interval != 0) and (epoch % interval == 0):

                scalars['loss/validation'] = torch.cat(world_val_loss).sum().item() / len(val_loader.dataset)

                # convert gw150914 samples to cpu and undo standardization
                gw150914_samples = torch.cat(gw150914_samples).cpu()
                gw150914_samples = (gw150914_samples * dataset.std) + dataset.mean
                
            # send data to async process to generate matplotlib figures
            queue.put((epoch, scalars, gw150914_samples))
            gw150914_samples = None   # reset to None for epochs where there is no corner plot

            if (save != 0) and (epoch % save == 0):
                # save checkpoint and write computationally expensive data to tb
                torch.save(flow.module.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')

    # destroy processes from distributed training
    if rank == 0:
        tb_process.terminate()
        
    cleanup_nccl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--profile', default=False, action="store_true")

    # logging
    logging.getLogger('bilby').setLevel(logging.INFO)

    args = parser.parse_args()

    assert isinstance(args.num_gpus, int), "num_gpus argument must be an integer."
    assert args.num_gpus > 0 and args.num_gpus <= torch.cuda.device_count(), f"{args.num_gpus} not a valid number of GPU devices."

    # data distributed parallel
    mp.spawn(
        train,
        args=tuple(args.__dict__.values()),  # assumes parser loaded in correct order
        nprocs=args.num_gpus,
        join=True
    )