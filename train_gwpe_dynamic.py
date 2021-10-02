import os
import logging
import argparse
import traceback
import signal
import time

from pathlib import Path
from typing import Union, Optional, List

from tqdm import tqdm
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import corner

import matplotlib.patches as mpatches
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
from pycbc.types import FrequencySeries

# local imports
from models import flows

from gwpe.basis import SVDBasis
from gwpe.pytorch.datasets import BasisCoefficientsDataset, BasisEncoderDataset
from gwpe.utils import read_ini_config
from gwpe.utils.events import generate_gw150914_context

def setup_nccl(rank, world_size):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(0, 180))

def cleanup_nccl():
    distributed.destroy_process_group()

def tensorboard_writer(
    queue: mp.Queue,
    log_dir: str,
    parameters: List[str],
    labels: List[str],
    static_args_ini: str,
    num_basis: int,
    val_coefficients: Optional[torch.Tensor]=None,
    val_gts: Optional[torch.Tensor]=None,
    figure_titles: Optional[List[str]]=None,
):

    # suppress luminosity distance debug messages
    logger = logging.getLogger('bilby')
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    if log_dir is None:
        tb = SummaryWriter()
    else:
        tb = SummaryWriter(log_dir)

    _, static_args = read_ini_config(static_args_ini)
    ifos = ('H1', 'L1')
    interferometers = {'H1': 'Hanford', 'L1': 'Livingston', 'V1': 'Virgo', 'K1': 'KAGRA'}
    basis_dir = Path('/mnt/datahole/daniel/gravflows/datasets/basis/')
    basis = SVDBasis(basis_dir, static_args_ini, ifos, preload=False)
    basis.load(time_translations=False, verbose=False)
    basis.truncate(num_basis)

    val_coefficients = val_coefficients.numpy()
    for j in range(val_coefficients.shape[0]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
        
        for i, ifo in enumerate(ifos):
            reconstruction = val_coefficients[j, i] @ basis.Vh[i]
            reconstruction = FrequencySeries(reconstruction, delta_f=static_args['delta_f'])
            strain = reconstruction.to_timeseries(delta_t=static_args['delta_t'])
            ax.plot(strain.sample_times, strain, label=interferometers[ifo], alpha=0.6)
            
        ax.set_title(f'Reconstructed {figure_titles[j]} Strain')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')  # units?
        ax.set_xlim((static_args['seconds_before_event']-1, static_args['seconds_before_event']+1))
        ax.legend(loc='upper left')
        ax.grid('both')

        # ax.axvline(static_args['seconds_before_event'], color='r', linestyle='--')  # merger time marker
        # ax.set_xticks([static_args['seconds_before_event']], minor=True)  # add low frequency cutoff to ticks
        # ax.set_xticklabels(['$t_{c}$'], minor=True, color='r')

        tb.add_figure(f'reconstructions/{figure_titles[j]}', fig)

    del reconstruction
    del val_coefficients
    del basis

    # bilby setup - specify the output directory and the name of the bilby run
    result = bilby.result.read_in_result(
        outdir='bilby_runs/GW150914',
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
        [10, 80],  # mass 1
        [10, 80],  # mass 2
        [0, 2*np.pi],  # phase 
        [0, 1],  # a_1
        [0, 1],  # a 2
        [0, np.pi],  # tilt 1
        [0, np.pi],  # tilt 2
        [0, 2*np.pi],  # phi_12
        [0, 2*np.pi],  # phi_jl
        [0, np.pi],  # theta_jn
        [0, np.pi],  # psi
        [0, np.pi],  # ra
        [-np.pi/2, np.pi/2],  # dec
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

            if samples is not None:
                # requires (batch, samples, parameters)
                assert len(samples.shape) == 3, "samples must be of shape (batch, samples, parameters)"

                if figure_titles is not None:
                    # to do - better handling of passing figure info through queue
                    assert samples.shape[0] == len(figure_titles), (
                        "sample.shape[0] and figure_titles must have matching lengths"
                    )
                else:
                    figure_titles = ['']*samples.shape[0]

            for key, value in scalars.items():
                tb.add_scalar(key, value, epoch)
            
            if samples is not None:
                assert isinstance(samples, torch.Tensor)
                for i in range(samples.shape[0]):

                    fig = plt.figure(figsize=(20,21))

                    if i == 0:
                        # GW150914 ONLY - hardcoded to first position
                        samples_df = pd.DataFrame(samples[i].numpy(), columns=parameters)
                        weights = cosmoprior.prob(samples_df['distance'])
                        weights = weights / np.mean(weights)
                        

                        corner.corner(
                            bilby_df,
                            fig=fig,
                            labels=labels,
                            levels=[0.5, 0.9],
                            quantiles=[0.25, 0.75],
                            color='tab:orange',
                            scale_hist=True,
                            plot_datapoints=False,
                        )

                        corner.corner(
                            samples_df,
                            fig=fig,
                            levels=[0.5, 0.9],
                            quantiles=[0.25, 0.75],
                            color='tab:blue',
                            scale_hist=True,
                            plot_datapoints=False,
                            show_titles=True,
                            weights=weights * len(bilby_samples) / len(samples_df),
                            range=domain,
                        )

                        fig.legend(
                            handles=[
                                mpatches.Patch(color='tab:blue', label='Neural Spline Flow'),
                                mpatches.Patch(color='tab:orange', label='Bilby')],
                            loc='upper right',
                            fontsize=16, 
                        )

                    else:
                        samples_df = pd.DataFrame(samples[i].numpy(), columns=parameters)
                        weights = cosmoprior.prob(samples_df['distance'])
                        weights = weights / np.mean(weights)

                        corner.corner(
                            samples_df,
                            fig=fig,
                            labels=labels,
                            levels=[0.5, 0.9],
                            quantiles=[0.25, 0.75],
                            color='tab:blue',
                            truth_color='tab:orange',
                            scale_hist=True,
                            plot_datapoints=False,
                            show_titles=True,
                            truths=val_gts[i].numpy() if val_gts is not None else None,
                            weights=weights * len(bilby_samples) / len(samples_df),
                            range=domain,
                        )
                        
                        fig.legend(
                            handles=[
                                mpatches.Patch(color='tab:blue', label='Neural Spline Flow'),
                                mpatches.Patch(color='tab:orange', label='Ground Truth')],
                            loc='upper right',
                            fontsize=16, 
                        )

                    fig.suptitle(f'{figure_titles[i]} Parameter Estimation', fontsize=18)
                    
                    # fig.savefig(f'gwpe/figures/{figure_titles[i]}.png')

                    tb.add_figure(f'posteriors/{figure_titles[i]}', fig, epoch)

            tb.flush()

        except Exception as e:
            # warning: assertions may not trigger exception to exit process
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGSTOP)  # to do: check kill command

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
    # profile: bool=False,
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
    val_dir = Path('/mnt/datahole/daniel/gravflows/datasets/validation/')
    
    noise_dir = Path('/mnt/datahole/daniel/gwosc/O1')
    psd_dir = Path("/mnt/datahole/daniel/gravflows/datasets/train/PSD/")
    basis_dir = Path('/mnt/datahole/daniel/gravflows/datasets/basis/')

    log_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{os.uname().nodename}"

    save_dir = Path('gwpe/model_weights/')
    experiment_dir = save_dir / log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # config files
    waveform_params_ini = str(data_dir / 'config_files/parameters.ini')
    extrinsics_ini = 'gwpe/config_files/extrinsics.ini'

    static_args_ini = str(data_dir / 'config_files/static_args.ini')

    # validation

    # training data
    # dataset = BasisCoefficientsDataset(
    #     data_dir=data_dir,
    #     basis_dir=basis_dir,
    #     static_args_ini=static_args_ini,
    #     parameters_ini=waveform_params_ini,
    # )

    dataset = BasisEncoderDataset(
        n=num_basis,
        data_dir=data_dir,
        basis_dir=basis_dir,
        static_args_ini=static_args_ini,
        intrinsics_ini=waveform_params_ini,
        extrinsics_ini=extrinsics_ini,
        psd_dir=psd_dir,
        ifos=['H1','L1'],
        ref_ifo='H1',
        downcast=True,
        add_noise=True,
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
        prefetch_factor=2,
        worker_init_fn=dataset._worker_init_fn,
        collate_fn=dataset._collate_fn,
    )

    # validation data
    val_dataset = BasisCoefficientsDataset(
        data_dir=val_dir,
        basis_dir=basis_dir,
        static_args_ini=static_args_ini,
        parameters_ini=[waveform_params_ini, extrinsics_ini],
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
        prefetch_factor=4,
        worker_init_fn=val_dataset._worker_init_fn,
        collate_fn=val_dataset._collate_fn,
    )


    # validation data
    if interval != 0:

        # specify indices in validation dataset to validate samples
        min_idx = val_dataset.parameters.distance.argmin()
        max_idx = val_dataset.parameters.distance.argmax()
        median_idx = val_dataset.parameters.loc[
            val_dataset.parameters.distance == val_dataset.parameters.distance.quantile(interpolation='nearest')
        ].index[0]

        if rank == 0:
            figure_titles = ['GW150914', 'Min Distance', f'Median Distance', f'Max Distance']

            # validation ground truths for posterior sampling
            val_gts = torch.stack([
                torch.zeros(len(val_dataset.parameters.columns), dtype=torch.float32),  # gw150914 dummy gt
                torch.tensor(val_dataset.parameters.iloc[min_idx].values, dtype=torch.float32),  # rank 1
                torch.tensor(val_dataset.parameters.iloc[median_idx].values, dtype=torch.float32),  # rank 2
                torch.tensor(val_dataset.parameters.iloc[max_idx].values, dtype=torch.float32),  # rank 3
            ])

        with torch.no_grad():
            # load data from file manually (rather than using val_dataset._worker_init_fn)
            val_coefficients = np.load(val_dataset.data_dir / val_dataset.data_file, mmap_mode='c')

            # generate coefficients on cpu - we want to send this to tensorboard (rank 0) before sending to gpus
            val_coefficients = torch.cat([
                torch.from_numpy(generate_gw150914_context(num_basis, noise_dir, psd_dir, basis_dir, static_args_ini))[None],
                torch.tensor(val_coefficients[[min_idx, median_idx, max_idx]]),
            ], dim=0).to(dtype=torch.complex64)

            # place one of each stacked tensor onto corresponding gpu rank
            val_context = val_coefficients[rank] * torch.from_numpy(val_dataset.standardization[:, :num_basis])
            val_context = val_context.to(device=rank)
            val_context = torch.cat([val_context.real, val_context.imag], dim=0)
            val_context = val_context.reshape(val_context.shape[0]*val_context.shape[1])[None]

    else:
        figure_titles = None
        val_gts = None
        val_coefficients = None

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
            args=(
                queue,
                f'gwpe/runs/{log_dir}',
                val_dataset.generator.parameters,
                val_dataset.generator.latex,
                static_args_ini,
                num_basis,
                val_coefficients,
                val_gts,
                figure_titles,
            )
        )
        tb_process.start()

    # instantiate neural spline coupling flow
    flow = flows.create_NDE_model(
        input_dim=14,  # we do not predict coalescence time 
        context_dim=4*num_basis,
        num_flow_steps=15,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 10,
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
            iterator = iter(dataloader)
            coefficients, parameters = next(iterator)
            
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
                    coefficients, parameters = next(iterator)
                    
                    coefficients = coefficients.to(rank, non_blocking=True)
                    parameters = parameters.to(rank, non_blocking=True)

                except StopIteration:
                    # exit while loop if iterator is complete
                    complete = True
                    
                loss.backward()
                optimizer.step()
                # if profile: profiler.step()

                # total loss summed over each sample in batch
                train_loss += loss.detach() * coefficients.shape[0]
                if rank == 0: progress.update(1)

            scheduler.step()

            # gather total loss during epoch between each GPU worker as list of tensors
            world_loss = [torch.ones_like(train_loss) for _ in range(world_size)]
            distributed.all_gather(world_loss, train_loss)
            train_loss *= 0.0  # reset loss for next epoch
            
            if (interval != 0) and (epoch % interval == 0):
                with torch.no_grad():
                    # evaluate model on validation dataset
                    iterator = iter(enumerate(val_loader))
                    step, (coefficients, parameters) = next(iterator)
                    coefficients = coefficients.to(rank, non_blocking=True)
                    parameters = parameters.to(rank, non_blocking=True)

                    if rank==0:
                        val_progress = int(100*step/len(val_loader))
                        progress.set_description(f'[{log_dir}] Validating ({val_progress}%)', refresh=True)

                    complete = False
                    while not complete:

                        # negative log-likelihood conditional on strain over mini-batch
                        loss = -flow.module.log_prob(parameters, context=coefficients).mean()

                        try:
                            # async get data from CPU and move to GPU during model forward
                            step, (coefficients, parameters) = next(iterator)
                            coefficients = coefficients.to(rank, non_blocking=True)
                            parameters = parameters.to(rank, non_blocking=True)

                            if rank==0:
                                val_progress = int(100*step/len(val_loader))
                                progress.set_description(f'[{log_dir}] Validating ({val_progress}%)', refresh=True)

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
                        n=10000,
                        context=val_context,
                        output_device='cuda',
                        dtype=torch.float32,
                    )[0]

                    # gather samples from all gpus
                    world_samples = [torch.ones_like(samples) for _ in range(world_size)]
                    distributed.all_gather(world_samples, samples)

            if (rank == 0):
                progress.set_description(f'[{log_dir}] Sending to TensorBoard', refresh=True)
                
                scalars = {
                    'loss/train': torch.cat(world_loss).sum().item() / len(dataloader.dataset)
                } 

                # every "interval" we generate samples for vis, else None
                corner_samples = None   # reset to None for epochs where there is no corner plot
                if (interval != 0) and (epoch % interval == 0):

                    scalars['loss/validation'] = torch.cat(world_val_loss).sum().item() / len(val_loader.dataset)

                    # convert gw150914 samples to cpu and undo standardization
                    corner_samples = torch.stack(world_samples).cpu()
                    corner_samples *= torch.from_numpy(val_dataset.std)
                    corner_samples += torch.from_numpy(val_dataset.mean)

                # send data to async process to generate matplotlib figures
                queue.put((epoch, scalars, corner_samples))

                if (save != 0) and (epoch % save == 0):
                    # save checkpoint and write computationally expensive data to tb
                    torch.save(flow.module.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                    torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')

    # destroy processes from distributed training
    if rank == 0:
        # to do - graceful way to shutdown workers
        # need to send message back from child process
        sleep_time = 120
        for i in range(sleep_time):
            progress.set_description(f'[{log_dir}] Shutting down in {sleep_time - i}s', refresh=True)
            time.sleep(1)
            
        tb_process.terminate()
        
    cleanup_nccl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=6, help="Number of workers for dataloader.")
    parser.add_argument('--num_basis', type=int, default=100, help="Number of SVD reduced basis elements to use")
    parser.add_argument('--verbose', default=False, action="store_true")
    # parser.add_argument('--profile', default=False, action="store_true")


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