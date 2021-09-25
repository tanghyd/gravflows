import os
import signal
import traceback
# from numpy.lib.function_base import _median_dispatcher

from pathlib import Path
from typing import Optional
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime, timedelta

import numpy as np

import corner

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# data distributed parallel
import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import pycbc.psd
from pycbc.catalog import Merger

# local imports
from flows import nde_flows

from data.config import read_ini_config
from data.noise import NoiseTimeline, load_psd_from_file, get_tukey_window
from data.parameters import ParameterGenerator
from data.pytorch import WaveformDataset
from basis.pytorch import BasisEncoder

def setup_nccl(rank, world_size):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(0, 180))

def cleanup_nccl():
    distributed.destroy_process_group()

def tensorboard_writer(queue: mp.Queue, log_dir: Optional[str]):
    if log_dir is None:
        tb = SummaryWriter()
    else:
        tb = SummaryWriter(log_dir)
    while True:
        try:
            epoch, scalars, samples, labels = queue.get()
            
            for key, value in scalars.items():
                tb.add_scalar(key, value, epoch)
            
            if samples is not None:
                fig = corner.corner(
                    samples,
                    levels=[0.5, 0.9],
                    scale_hist=True,
                    plot_datapoints=False,
                    labels=labels,
                    show_titles=True,
    #                 fontsize=18,
    #                 truths=ground_truths[0].numpy()
                #     range=corner_range,
                )

                fig.suptitle('Gravitational Wave Parameter Estimates', fontsize=22)
                tb.add_figure('corner', fig, epoch)
            tb.flush()

        except Exception as e:
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGSTOP)

# Training with PyTorch native DataDistributedParallel (DDP)
def train_distributed(
    rank: int,
    world_size: int,
    epochs: int=500,
    interval: int=100,
    verbose: bool=False,
):
    
    assert (0 < interval) and (interval < epochs), "Interval must be a positive integer between 0 and epochs."

    # setup data distributed parallel training
    setup_nccl(rank, world_size)  # world size is total gpus
    torch.cuda.set_device(rank)  # rank is gpu index
    device = torch.device('cuda')

    # setup asynchronous figure generation and tensorboard writer process
    log_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{os.uname().nodename}"
    if rank == 0:  
        queue = mp.SimpleQueue()
        tb_process = mp.Process(target=tensorboard_writer, args=(queue, f'runs/{log_dir}'))
        tb_process.start()

    # directories
    noise_dir = Path('/mnt/datahole/daniel/gwosc/O1')
    psd_dir = Path("/mnt/datahole/daniel/gravflows/datasets/train/PSD/")
    waveform_dir = Path('/mnt/datahole/daniel/gravflows/datasets/train/')
    basis_dir = Path('/mnt/datahole/daniel/gravflows/datasets/basis/')

    save_dir = Path('model_weights/')
    experiment_dir = save_dir / log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # config files
    waveform_params_ini = str(waveform_dir / 'config_files/parameters.ini')
    projection_params_ini = 'config_files/extrinsics.ini'
    static_args_ini = str(waveform_dir / 'config_files/static_args.ini')
    event_args_ini = 'config_files/event_args.ini'  # GW150914 data
    _, static_args = read_ini_config(static_args_ini)
    _, event_args = read_ini_config(event_args_ini)

    # parameter generator and standardization
    generator = ParameterGenerator(config_files=[waveform_params_ini, projection_params_ini])
    mean = torch.tensor(generator.statistics['mean'].values, device=device)
    std = torch.tensor(generator.statistics['std'].values, device=device)

    # power spectral density
    ifos = ('H1', 'L1')
    # interferometers = {'H1': 'Hanford', 'L1': 'Livingston', 'V1': 'Virgo', 'K1': 'KAGRA'}

    psds = {}
    for ifo in ifos:
        # coloured noise from psd
        psd_file = Path(psd_dir) / f'{ifo}_PSD.npy'
        assert psd_file.is_file(), f"{psd_file} does not exist."
        psds[ifo] = load_psd_from_file(psd_file, delta_f=event_args['delta_f'])

    # get GW150914 Test data
    timeline = NoiseTimeline(data_dir=noise_dir, ifos=ifos)
    strains = timeline.get_strains(
        int(Merger('GW150914').time - event_args['seconds_before_event']),
        int(event_args['waveform_length'])
    )

    event_strain = {}
    for ifo in strains:
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
        event_strain[ifo] = strain.to_frequencyseries(delta_f=static_args['delta_f'])[:static_args['fd_length']]

    # reduced basis encoder
    n = 100  # number of reduced basis elements
    encoder = BasisEncoder(basis_dir, n)
    encoder.to(device)

    with torch.no_grad():
        # generate GW150914 reduced basis coefficients
        gw150914 = encoder(torch.tensor(np.stack(list(event_strain.values()))[None], device=device))
        gw150914 = torch.cat([gw150914.real, gw150914.imag], dim=1).to(torch.float32)
        gw150914 = gw150914.reshape(gw150914.shape[0], gw150914.shape[1]*gw150914.shape[2])
        

    # training data set
    batch_size = 1000

    dataset = WaveformDataset(
        data_dir=waveform_dir,
        static_args_ini=static_args_ini,
        intrinsics_ini=waveform_params_ini,
        extrinsics_ini=projection_params_ini,
        psd_dir=psd_dir,
        ifos=ifos,
    )

    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
        seed=rank,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        # prefetch_factor=4,
        worker_init_fn=dataset._worker_init_fn,
        collate_fn=dataset._collate_fn,
    )

    # load neural spline flow model
    flow = nde_flows.create_NDE_model(
        input_dim=15, 
        context_dim=4*n,
        num_flow_steps=15,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 10,
            'activation': 'elu',
        }
    )

    # sync_bn_flow = nn.SyncBatchNorm.convert_sync_batchnorm(flow, process_group=None)

    # train
    flow = DDP(flow.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(flow.parameters(), lr=2e-4)

    flow.train()
    train_loss = torch.zeros((1,), device=device, requires_grad=False)

    # tqdm progress bar
    disable = False if verbose and (rank == 0) else True

    with tqdm(total=len(dataloader)*epochs, desc='Training', disable=disable) as progress:
        for epoch in range(1, 1+epochs):
            if rank == 0: progress.set_postfix({'epoch': epoch})
            iterator = iter(dataloader)
            projections, parameters = next(iterator) 

            projections = projections.to(device, non_blocking=True)
            parameters = parameters.to(device, torch.float32, non_blocking=True)
            
            complete = False
            while not complete:
                optimizer.zero_grad()

                # project to reduced basis and flatten for 1-d residual network input
                coefficients = encoder(projections)
                coefficients = torch.cat([coefficients.real, coefficients.imag], dim=1).to(torch.float32)
                coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])

                # negative log-likelihood conditional on strain over mini-batch
                loss = -flow.module.log_prob(parameters, context=coefficients).mean()

                try:
                    # async get data from CPU and move to GPU during model forward
                    projections, parameters = next(iterator) 
                    projections = projections.to(device, non_blocking=True)
                    parameters = parameters.to(device, torch.float32, non_blocking=True)

                except StopIteration:
                    # exit while loop if iterator is complete
                    complete = True

                loss.backward()
                optimizer.step()

                # total loss summed over each sample in batch
                train_loss += loss.detach() * coefficients.shape[0]  # * 15 ??
                if rank == 0: progress.update(1)

            # gather total loss during epoch between each GPU worker as list of tensors
            world_loss = [torch.ones_like(train_loss) for _ in range(world_size)]
            distributed.all_gather(world_loss, train_loss)
            train_loss *= 0.0  # reset loss for next epoch
                
            with torch.no_grad():
                # validation with corner plot
                n_samples = 50000
                sample_size = (n_samples // world_size, parameters.shape[-1])
                world_samples = [
                    torch.ones(sample_size, dtype=torch.float32, device=device)
                    for _ in range(world_size)
                ]

                # generate samples for corner plots etc.
                samples = nde_flows.sample_flow(
                    flow.module,
                    n=n_samples// world_size,
                    context=gw150914,
                    output_device='cuda'
                )[0]
                samples = (samples * std) + mean

                print(f'rank {rank} samples.shape: {samples.shape}')
                for key, value in zip(generator.latex, samples.T):
                    print(f'rank {rank} {key} min: {value.min()} max: {value.max()} mean: {value.mean()}')

                distributed.all_gather(world_samples, samples)
                parameter_samples = torch.cat(world_samples, dim=0).cpu()#.numpy()

            if rank == 0:
                # log average epoch loss across DDP to tensorboard
                scalars = {
                    'loss': torch.cat(world_loss).mean().item() / len(dataloader.dataset)
                } 
                
                # send data to async process to generate matplotlib figures
                print(f'paramter_samples.shape: {parameter_samples.shape}')
                for key, value in zip(generator.latex, parameter_samples.T):
                    print(f'{key} min: {value.min()} max: {value.max()} mean: {value.mean()}')

                queue.put((epoch, scalars, parameter_samples.numpy(), generator.latex))
                parameter_samples = None  # reset to None for epochs where there is no corner plot

                if (interval != 0) and (epoch % interval == 0):
                    # save checkpoint and write computationally expensive data to tb
                    torch.save(flow.module.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                    torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')

            distributed.barrier()

        # destroy processes from distributed training
        cleanup_nccl()
        tb_process.terminate()
        tb_process.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    # training settings
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)

    # data directories
    # parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load parameter files.')
    # parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated waveform files.')
    # parser.add_argument('--psd_dir', dest='psd_dir', type=str, help='The output directory to save generated waveform files.')

    # logging
    parser.add_argument('--verbose', default=False, action="store_true")


    args = parser.parse_args()

    assert isinstance(args.num_gpus, int), "num_gpus argument must be an integer."
    assert args.num_gpus > 0 and args.num_gpus <= torch.cuda.device_count(), f"{args.num_gpus} not a valid number of GPU devices."


    # run pytorch training
    if args.num_gpus == 1:
        raise NotImplementedError()
        train()

    else:
        # data distributed parallel
        mp.spawn(
            train_distributed,
            args=(args.num_gpus, args.epochs, args.interval, args.verbose),
            nprocs=args.num_gpus,
            join=True
        )