import os
import signal
import traceback
import logging

from pathlib import Path
from typing import List
from argparse import ArgumentParser

import time
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

import pycbc.psd
from pycbc.catalog import Merger

import bilby

# local imports
from models import flows

from gwpe.utils import read_ini_config
from gwpe.noise import NoiseTimeline, load_psd_from_file, get_tukey_window
from gwpe.parameters import ParameterGenerator
from gwpe.pytorch.waveforms import WaveformDataset
from gwpe.pytorch.basis import BasisEncoder

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

    labels=labels[:13]
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
        [0.005,0.055],  # tc
        [100,800],  # distance
    ]

    cosmoprior = bilby.gw.prior.UniformSourceFrame(
        name='luminosity_distance',
        minimum=1e2,
        maximum=1e3,
    )

    while True:
        try:
            epoch, scalars, figures = queue.get()
            
            for key, value in scalars.items():
                tb.add_scalar(key, value, epoch)
            
            if figures is not None:
                for key, value in figures.items():
                    
                    if key == 'posteriors/gw150914':
                        assert isinstance(value, torch.Tensor)
                        samples_df = pd.DataFrame(value.numpy(), columns=parameters)
                        weights = cosmoprior.prob(samples_df['distance'])
                        weights = weights / np.mean(weights)

                        fig = corner.corner(
                            bilby_df.iloc[:, :13],
                            levels=[0.5, 0.9],
                            scale_hist=True,
                            plot_datapoints=False,
                            labels=labels,
                            color='red',
                        )

                        corner.corner(
                            samples_df.iloc[:, :13],
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
                        tb.add_figure(key, fig, epoch)

                    elif key in ('posteriors/validation', 'posteriors/test', 'posteriors/train'):
                        assert isinstance(value, tuple)
                        samples, ground_truth = value  # unpack a tuple
                        samples_df = pd.DataFrame(samples.numpy(), columns=parameters)
                        weights = cosmoprior.prob(samples_df['distance'])
                        weights = weights / np.mean(weights)

                        fig = corner.corner(
                            samples_df.iloc[:, :13],
                            levels=[0.5, 0.9],
                            scale_hist=True,
                            plot_datapoints=False,
                            labels=labels,
                            show_titles=True,
                            fontsize=18,
                            # range=domain,
                            truths=ground_truth[0].numpy()[:13],
                            weights=weights * len(bilby_samples) / len(samples_df),
                        )
                        
                        fig.suptitle(f"{key.split('/')[-1].replace('_', ' ').title()} Validation Sample: NSF (black) vs. Bilby (red)", fontsize=22)
                        tb.add_figure(key, fig, epoch)

            tb.flush()

        except Exception as e:
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGSTOP)

# Training with PyTorch native DataDistributedParallel (DDP)
def train_distributed(
    rank: int,
    world_size: int,
    epochs: int=500,
    interval: int=10,
    save: int=100,
    verbose: bool=False,
):
    
    assert (0 <= interval) and (interval <= epochs), "Interval must be a positive integer between 0 and epochs."

    # setup data distributed parallel training
    setup_nccl(rank, world_size)  # world size is total gpus
    torch.cuda.set_device(rank)  # rank is gpu index
    device = torch.device('cuda')

    # directories
    noise_dir = Path('/mnt/datahole/daniel/gwosc/O1')
    psd_dir = Path("/mnt/datahole/daniel/gravflows/datasets/train/PSD/")
    basis_dir = Path('/mnt/datahole/daniel/gravflows/datasets/basis/')
    waveform_dir = Path('/mnt/datahole/daniel/gravflows/datasets/train/')
    validation_dir = Path('/mnt/datahole/daniel/gravflows/datasets/validation/')
    test_dir = Path('/mnt/datahole/daniel/gravflows/datasets/test/')
    log_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{os.uname().nodename}"  # tb

    save_dir = Path('gwpe/model_weights/')
    experiment_dir = save_dir / log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # config files
    waveform_params_ini = str(waveform_dir / 'config_files/parameters.ini')
    projection_params_ini = 'gwpe/config_files/extrinsics.ini'
    static_args_ini = str(waveform_dir / 'config_files/static_args.ini')
    event_args_ini = 'gwpe/config_files/event_args.ini'  # GW150914 data
    _, static_args = read_ini_config(static_args_ini)
    _, event_args = read_ini_config(event_args_ini)

    # parameter generator and standardization
    generator = ParameterGenerator(config_files=[waveform_params_ini, projection_params_ini])
    mean = torch.tensor(generator.statistics['mean'].values, device=device, dtype=torch.float32)
    std = torch.tensor(generator.statistics['std'].values, device=device, dtype=torch.float32)

    # handle parameter scaling for no time shift/distance scaling
    # mean[generator.parameters.index('distance')] = 0.
    # std[generator.parameters.index('distance')] = 1.
    # mean[generator.parameters.index('time')] = 0.
    # std[generator.parameters.index('time')] = 1.

    # power spectral density
    ifos = ('H1', 'L1')
    # interferometers = {'H1': 'Hanford', 'L1': 'Livingston', 'V1': 'Virgo', 'K1': 'KAGRA'}

    # setup asynchronous figure generation and tensorboard writer process
    if rank == 0:  
        queue = mp.SimpleQueue()
        tb_process = mp.Process(
            target=tensorboard_writer,
            args=(queue, f'gwpe/runs/{log_dir}', generator.parameters, generator.latex)
        )
        tb_process.start()

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

    # training data set
    batch_size = 1000

    dataset = WaveformDataset(
        data_dir=waveform_dir,
        static_args_ini=static_args_ini,
        intrinsics_ini=waveform_params_ini,
        extrinsics_ini=projection_params_ini,
        psd_dir=psd_dir,
        ifos=ifos,
        downcast=True,
        scale=False,
        shift=False,
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
        worker_init_fn=dataset._worker_init_fn,
        collate_fn=dataset._collate_fn,
    )

    val_dataset = WaveformDataset(
        data_dir=validation_dir,
        static_args_ini=static_args_ini,
        data_file='projections.npy',
        psd_dir=psd_dir,
        ifos=ifos,
        downcast=True,
        scale=False,
        shift=False,
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
        num_workers=4,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=val_dataset._worker_init_fn,
    )

    test_dataset = WaveformDataset(
        data_dir=test_dir,
        static_args_ini=static_args_ini,
        data_file='projections.npy',
        psd_dir=psd_dir,
        ifos=ifos,
        downcast=True,
        scale=False,
        shift=False,
    )


    # reduced basis encoder
    n = 100  # number of reduced basis elements
    encoder = BasisEncoder(basis_dir, n, torch.complex64)  # .pt checkpoint must be single precision
    encoder.to(device=device)
    encoder.load_state_dict(torch.load(basis_dir / f'basis_encoder_{n}.pt', map_location=device))
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # validation data
    if interval != 0:
        if rank != 3:  # gpu3 will sample from training samples
            with torch.no_grad():
                if rank == 0:
                    # generate samples for gw150914
                    context = torch.tensor(np.stack(list(event_strain.values()))[None], dtype=torch.complex64, device=device)
                    gts = torch.zeros((1,len(val_dataset.parameters.columns)), dtype=torch.float32, device=device)  # dummy tensor

                elif rank == 1:
                    idx = val_dataset.parameters.distance.argmin()

                    # manually load projected waveforms and ground truths from file
                    # this is to circumvent calling ._worker_init_fn() on the DataSet object before training
                    context = np.load(val_dataset.data_dir / val_dataset.data_file, mmap_mode='r')[[idx]]
                    context = torch.tensor(np.array(context), dtype=torch.complex64, device=device)
                        
                    # generate noise for whitened waveform above lowpass filter (i.e. >20Hz)
                    lowpass = int(dataset.static_args['f_lower'] / dataset.static_args['delta_f'])
                    size = (context.shape[0], context.shape[1], context.shape[2] - lowpass)
                    context[:, :, lowpass:] += torch.randn(size, dtype=context.dtype, device=context.device)
                        
                    gts = val_dataset.parameters.iloc[[idx]].values
                    gts = torch.tensor(val_dataset.parameters.iloc[[idx]].values, dtype=torch.float32, device=device)

                    # idx = val_dataset.parameters.loc[
                    #     val_dataset.parameters.distance == val_dataset.parameters.distance.quantile(interpolation='nearest')
                    # ].index[0]
                
                elif rank == 2:
                    
                    idx = test_dataset.parameters.distance.argmin()
                    context = np.load(test_dataset.data_dir / test_dataset.data_file, mmap_mode='r')[[idx]]
                    context = torch.tensor(np.array(context), dtype=torch.complex64, device=device)

                    gts = test_dataset.parameters.iloc[[idx]].values
                    gts = torch.tensor(test_dataset.parameters.iloc[[idx]].values, dtype=torch.float32, device=device)
                    
                # context = encoder(context)
                context = torch.einsum('bij, bijk -> bik', context, encoder.basis) * encoder.scaler
                context = torch.cat([context.real, context.imag], dim=1)
                context = context.reshape(context.shape[0], context.shape[1]*context.shape[2])


    flow = flows.create_NDE_model(
        input_dim=15, 
        context_dim=4*n,
        num_flow_steps=30,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 5,
            'activation': 'elu',
        }
    )

    # sync_bn_flow = nn.SyncBatchNorm.convert_sync_batchnorm(flow)
    flow = DDP(flow.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(flow.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # run training loop
    flow.train()
    train_loss = torch.zeros((1,), device=device, requires_grad=False)
    val_loss = torch.zeros((1,), device=device, requires_grad=False)

    disable_pbar = False if verbose and (rank == 0) else True  # tqdm progress bar
    with tqdm(total=len(dataloader)*epochs, disable=disable_pbar) as progress:
        for epoch in range(1, 1+epochs):
            if rank == 0:
                progress.set_postfix({'epoch': epoch})
                progress.set_description(f'[{log_dir}] Training', refresh=True)

            # let all processes sync up before starting with a new epoch of training
            distributed.barrier()

            iterator = iter(dataloader)
            projections, parameters = next(iterator) 

            projections = projections.to(device, non_blocking=True)
            parameters = parameters.to(device, non_blocking=True)
            
            complete = False
            while not complete:
                optimizer.zero_grad()

                # generate noise for whitened waveform above lowpass filter (i.e. >20Hz)
                lowpass = int(dataset.static_args['f_lower'] / dataset.static_args['delta_f'])
                size = (projections.shape[0], projections.shape[1], projections.shape[2] - lowpass)
                projections[:, :, lowpass:] += torch.randn(size, dtype=projections.dtype, device=projections.device)

                # project to reduced basis and flatten for 1-d residual network input
                coefficients = encoder(projections)
                coefficients = torch.cat([coefficients.real, coefficients.imag], dim=1)
                coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])


                # negative log-likelihood conditional on strain over mini-batch
                loss = -flow.module.log_prob(parameters, context=coefficients).mean()
                # print(f"rank {rank}: train loss: {loss.detach().item()}")

                try:
                    # async get data from CPU and move to GPU during model forward
                    projections, parameters = next(iterator) 
                    projections = projections.to(device, non_blocking=True)
                    parameters = parameters.to(device, non_blocking=True)

                except StopIteration:
                    # exit while loop if iterator is complete
                    complete = True

                loss.backward()
                optimizer.step()
                scheduler.step()

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
                    # evaluation on noisy validation set 
                    iterator = iter(val_loader)
                    projections, parameters = next(iterator)
                    projections = projections.to(device, dtype=torch.complex64, non_blocking=True)
                    parameters = parameters.to(device, dtype=torch.float32, non_blocking=True)
            
                    complete = False
                    while not complete:
                        optimizer.zero_grad()

                        # generate noise for whitened waveform above lowpass filter (i.e. >20Hz)
                        lowpass = int(dataset.static_args['f_lower'] / dataset.static_args['delta_f'])
                        size = (projections.shape[0], projections.shape[1], projections.shape[2] - lowpass)
                        projections[:, :, lowpass:] += torch.randn(size, dtype=projections.dtype, device=projections.device)

                        # project to reduced basis and flatten for 1-d residual network input
                        coefficients = encoder(projections)
                        coefficients = torch.cat([coefficients.real, coefficients.imag], dim=1)
                        coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])

                        # validation does not use collate_fn so we need to manually standardize
                        parameters = (parameters - mean ) / std

                        # negative log-likelihood conditional on strain over mini-batch
                        loss = -flow.module.log_prob(parameters, context=coefficients).mean()
                        # print(f"rank {rank}: val loss: {loss.detach().item()}")

                        try:
                            # async get data from CPU and move to GPU during model forward
                            projections, parameters = next(iterator) 
                            projections = projections.to(device, dtype=torch.complex64, non_blocking=True)
                            parameters = parameters.to(device, dtype=torch.float32, non_blocking=True)

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

                    # sample posteriors from training
                    if rank==3:
                        # first sample in final batch (random)
                        context = coefficients[0]
                        gts = parameters[0]

                    samples = flows.sample_flow(
                        flow.module,
                        n=25000,
                        context=context,
                        output_device='cuda',
                        dtype=torch.float32,
                    )

                    # undo standardized sampled parameters
                    samples = (samples[0] * std) + mean

                    # gather samples from all gpus
                    world_samples = [torch.ones_like(samples) for _ in range(world_size)]
                    distributed.all_gather(world_samples, samples)

                    world_gts = [torch.ones_like(gts) for _ in range(world_size)]
                    distributed.all_gather(world_gts, gts)


            if rank == 0:
                progress.set_description(f'[{log_dir}] Sending to TensorBoard', refresh=True)
                
                scalars = {
                    'loss/train': torch.cat(world_loss).sum().item() / len(dataloader.dataset)
                } 
                
                figures = None  # reset to None for epochs where there is no corner plot
                if (interval != 0) and (epoch % interval == 0):
                    scalars['loss/validation'] = torch.cat(world_val_loss).sum().item() / len(val_loader.dataset)

                    # to do - set up this process for num_gpus < 4
                    figures = {}
                    world_gts = [gts.cpu() for gts in world_gts]
                    world_samples = [world_sample.cpu() for world_sample in world_samples]
                    for i, (world_sample, gt) in enumerate(zip(world_samples, world_gts)):
                        # data is sent as tuples - .share_memory_() must be called manually
                        world_sample.share_memory_()
                        gt.share_memory_()

                        # i should correspond to gpu rank of origin
                        if i == 0:
                            figures['posteriors/gw150914'] = world_sample
                        elif i == 1:
                            figures['posteriors/validation'] = (world_sample, gt)
                        elif i == 2:
                            figures['posteriors/test'] = (world_sample, gt)
                        elif i == 3:
                            figures['posteriors/train'] = (world_sample, gt)

                # send data to async process to generate matplotlib figures
                queue.put((epoch, scalars, figures))

                if (save != 0) and (epoch % save == 0):
                    # save checkpoint and write computationally expensive data to tb
                    torch.save(flow.module.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                    torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')

        # destroy processes from distributed training
        if rank == 0:
            tb_process.terminate()
            
        cleanup_nccl()

if __name__ == '__main__':
    parser = ArgumentParser()

    # training settings
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--verbose', default=False, action="store_true")

    # data directories
    # parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load parameter files.')
    # parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated waveform files.')
    # parser.add_argument('--psd_dir', dest='psd_dir', type=str, help='The output directory to save generated waveform files.')

    # logging

    logging.getLogger('bilby').setLevel(logging.INFO)


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
            args=(args.num_gpus, args.epochs, args.interval, args.save, args.verbose),
            nprocs=args.num_gpus,
            join=True
        )