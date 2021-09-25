import os
import shutil
import argparse

# TO DO: Implement logging over print statements
import logging

from pathlib import Path
from typing import Optional, Union, Dict, List
from functools import partial
from data.noise import load_psd_from_file
from tqdm import tqdm
from datetime import datetime

import multiprocessing
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.lib.format import open_memmap

from pycbc.detector import Detector
from pycbc.types.frequencyseries import FrequencySeries

# local imports
from data.config import read_ini_config
from data.noise import (
    load_psd_from_file, frequency_noise_from_psd,
    get_noise_std_from_static_args
)
from data.waveforms import (
    generate_intrinsic_waveform,
    batch_project, get_sample_frequencies,
)


def validate_waveform(
    waveform: np.ndarray,
    static_args: Dict[str, float],
    out_dir: Union[str, os.PathLike],
    name: Optional[str]=None,
):
    # create out dir if it does not exist
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Generate waveform
    assert waveform.shape[0] == 2, "First dimension of waveform sample must be 2 (+ and x polarizations)."
    assert waveform.shape[1] == static_args['fd_length'], "Waveform length not expected given provided static_args."
    
    hp, hc = waveform
    plus = FrequencySeries(hp, delta_f=static_args['delta_f'], copy=True)
    cross = FrequencySeries(hc, delta_f=static_args['delta_f'], copy=True)
    
    # ifft to time domain
    sp = plus.to_timeseries(delta_t=static_args['delta_t'])
    sc = cross.to_timeseries(delta_t=static_args['delta_t'])

    # plot plus / cross polarizations
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
    ax.plot(sp.sample_times, sp, label='Plus', color='tab:pink')  # pink for plus
    ax.plot(sc.sample_times, sc, label='Cross', color='tab:cyan')  # cyan for cross
    ax.set_ylabel('Strain')
    ax.set_xlabel('Time (s)')
    ax.grid('both')
        
    aux_desc='' if name is None else f' ({name})'
    ax.set_title(f"IFFT of Simulated {static_args['approximant']} Waveform Polarizations{aux_desc}", fontsize=14)
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(out_dir / 'intrinsic_polarizations.png')


def validate_projections(
    projections: np.ndarray,
    static_args: Dict[str, float],
    out_dir: Union[str, os.PathLike],
    ifos: Optional[List[str]]=None,
    name: Optional[str]=None,
):
    # full names for plotting
    interferometers = {'H1': 'Hanford', 'L1': 'Livingston', 'V1': 'Virgo', 'K1': 'KAGRA'}

    # create out dir if it does not exist
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Generate waveform
    if ifos is not None:
        assert projections.shape[0] == len(ifos), "First dimension of waveform sample must be 2 (+ and x polarizations)."
    assert projections.shape[1] == static_args['fd_length'], "Waveform length not expected given provided static_args."

    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows))

    # ifft to time domain
    for i in range(projections.shape[0]):
        label = None if ifos is None else interferometers[ifos[i]]
        h = FrequencySeries(projections[i, :], delta_f=static_args['delta_f'], copy=True)
        strain = h.to_timeseries(delta_t=static_args['delta_t'])
        ax.plot(strain.sample_times, strain, label=label, alpha=0.6)
        
    ax.set_ylabel('Strain')
    ax.set_xlabel('Time (s)')
    ax.grid('both')
    ax.legend(loc='upper left')

    aux_desc='' if name is None else f' ({name})'
    fig.suptitle(f"IFFT of Simulated {static_args['approximant']} Projected Waveforms at Interferometers{aux_desc}", fontsize=16) 
    fig.tight_layout()
    fig.savefig(out_dir / 'projection_examples.png')

def generate_waveform_dataset(
    static_args_ini: str,
    data_dir: str,
    out_dir: Optional[str]=None,
    params_file: str='parameters.csv',
    ifos: Optional[List[str]]=None,
    add_noise: bool=False,
    gaussian: bool=False,
    psd_dir: Optional[str]=None,
    whiten: bool=True,
    bandpass: bool=True,
    projections_only: bool=False,
    overwrite: bool=False,
    metadata: bool=True,
    downcast: bool=False,
    verbose: bool=True,
    validate: bool=False,
    workers: int=1,
    chunk_size: int=2500,
):
    """Function to generate a dataset of intrinsic waveform parameters using Python multiprocessing.
    
    Arguments:
        static_args: Dict[str, float]
            A dictionary of static args....? But we load this
        data_dir: str
            The dataset directory to load configuration files and save waveforms.
        params_file: str
            The .csv file to load params from - TO DO : should this be "parameters" or "parameters.csv"?  
        file_name: str
            The base name of the intrinsic waveform numpy file.

    """
    # load parameters
    data_dir = Path(data_dir)
    parameters = pd.read_csv(data_dir / params_file, index_col=0)  # assumes index column
    n_samples = len(parameters)
    chunks = int(np.ceil(n_samples/chunk_size))

    # TO DO: if parameters are not provided - do we sample / call generate_parameters automatically?

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)

    # check output numpy arrays
    waveform_file = out_dir / 'waveforms.npy'
    projections_file = out_dir / 'projections.npy'
    if not overwrite:
        if waveform_file.is_file():
            logging.debug(f'Aborting - {waveform_file} exists but overwrite is False.')
            return
        if projections_file.is_file():
            logging.debug(f'Aborting - {projections_file} exists but overwrite is False.')
            return

    if metadata:
        # load static_args.ini and save as .json in dataset
        in_args_path = Path(static_args_ini)
        config_dir = out_dir / 'config_files'
        config_dir.mkdir(exist_ok=True)
        out_args_path = config_dir / in_args_path.name
        shutil.copy(in_args_path, out_args_path)

    # specify precision of output waveforms
    dtype = np.complex128 if not downcast else np.complex64

    # whether to project waveforms onto detectors or keep as intrinsic
    if ifos is not None:
        required_extrinsics = ('ra', 'dec', 'psi')
        for extrinsic in required_extrinsics:
            assert extrinsic in parameters.columns, f"{extrinsic} not in {required_extrinsics}."

        detectors = {ifo: Detector(ifo) for ifo in ifos}
        waveform_desc = f'({ifos})'
        sample_frequencies = get_sample_frequencies(
            f_final=static_args['f_final'],
            delta_f=static_args['delta_f']
        )

        # load PSD files for noise generation or whitening
        if psd_dir is not None:
            psds = {}
            for ifo in ifos:
                # coloured noise from psd
                psd_file = Path(psd_dir) / f'{ifo}_PSD.npy'
                assert psd_file.is_file(), f"{psd_file} does not exist."
                psds[ifo] = load_psd_from_file(psd_file)

        # https://numpy.org/devdocs/reference/generated/numpy.lib.format.open_memmap.html#numpy.lib.format.open_memmap
        projection_memmap = open_memmap(
            filename=projections_file,
            mode='w+',  # create or overwrite file
            dtype=dtype,
            shape=(n_samples, len(ifos), static_args['fd_length'])
        )

        if add_noise:
            # noise_std = get_noise_std_from_static_args(static_args)
            noise_file = out_dir / 'noise.npy'
            noise_memmap = open_memmap(
                filename=noise_file,
                mode='w+',  # create or overwrite file
                dtype=dtype,
                shape=(n_samples, len(ifos), static_args['fd_length'])
            )

    else:
        # plus and cross polarizations
        detectors = None
        waveform_desc = '(intrinsic)'
        sample_frequencies = None  # only used for projected time shifts
        projections_only = False

    # intrinsic waveforms
    if not projections_only:
        waveform_memmap = open_memmap(
            filename=waveform_file,
            mode='w+',  # create or overwrite file
            dtype=dtype,
            shape=(n_samples, 2, static_args['fd_length'])
        )

    # multiprocessing generation
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # create buffer in memory to temporarily store before writing to desk
        waveforms = np.empty((chunk_size, 2, static_args['fd_length']), dtype=dtype)

        # loop through samples at approx. 10GB at a time and append to numpy array
        progress_desc = f"[{datetime.now().strftime('%H:%M:%S')}] Generating {static_args['approximant']} {waveform_desc} waveforms"
        saved = 0  # track number of saved arrays for progrss bar
        with tqdm(desc=progress_desc, total=n_samples, miniters=1, postfix={'saved': saved}, disable=not verbose) as progress:
            for i in range(chunks):
                # get index positions for chunks
                start = i*chunk_size
                end = (i+1)*chunk_size

                # check if chunk_size goes over length of samples
                if end > n_samples:
                    # overflow batch may not have a full chunk size - we need to re-instantiate waveform array
                    end = end - chunk_size + (n_samples % chunk_size)
                    waveforms = np.empty((end - start, 2, static_args['fd_length']), dtype=dtype)
                
                # get a chunk of samples
                saved += end - start
                samples = parameters.iloc[start:end].to_records()

                # submit waveform generation jobs to multiple processes while tracking source parameters by index
                waveform_generation_job = partial(
                    generate_intrinsic_waveform,
                    spins=True,  # hard-coded
                    spins_aligned=False,  # hard-coded
                    static_args=static_args,
                    downcast=downcast,
                )

                # store waveform polarisations in correct order while updating progress bar as futures complete
                ordered_futures = {executor.submit(waveform_generation_job, params): i for i, params in enumerate(samples)}
                for future in concurrent.futures.as_completed(ordered_futures):
                    waveforms[ordered_futures[future]] = np.stack(future.result())  #  assign (2, num_freq_bins) to array idx
                    progress.update(1)
                progress.refresh()

                if detectors is not None:
                    # output should be (batch, ifo, length)
                    projections = np.empty((end - start, len(ifos), static_args['fd_length']), dtype=dtype)
                    if add_noise:
                        noise = np.empty((end - start, len(ifos), static_args['fd_length']), dtype=dtype)
                    
                    for i, ifo in enumerate(ifos):
                        # batch project for each detector
                        projections[:, i, :] = batch_project(detectors[ifo], samples, waveforms, static_args, sample_frequencies)
                        
                        if add_noise:
                            if gaussian or psd_dir is None:
                                # gaussian white noise in frequency domain
                                size = (end-start, static_args['fd_length'])  # gaussian for each batch for each freq bin
                                noise[:, i, :] = (np.random.normal(0., 1., size) + 1j*np.random.normal(0., 1., size)).astype(dtype)
                            else:
                                # coloured noise from psd -- cut to fd_length (bandpass filter for higher frequencies)
                                noise[:, i, :] = frequency_noise_from_psd(psds[ifo], n=end-start)[:, :static_args['fd_length']]
                        
                        if whiten:
                            projections[:, i, :] /= psds[ifo][:static_args['fd_length']] ** 0.5
                            if add_noise:
                                noise[:, i, :] /= psds[ifo][:static_args['fd_length']] ** 0.5

                        if bandpass:
                            # filter out values less than f_lower (e.g. 20Hz) - to do: check truncation vs. zeroing
                            projections[:, i, :int(static_args['f_lower'] / static_args['delta_f'])] = 0.0
                            if add_noise:
                                noise[:, i, :int(static_args['f_lower'] / static_args['delta_f'])] = 0.0

                    if add_noise:
                        noise_memmap[start:end, :, :] = noise
                        projections += noise

                    projection_memmap[start:end, :, :] = projections
                    
                if not projections_only:
                    waveform_memmap[start:end, :, :] = waveforms

                # notify timer that batch has been saved
                progress.set_postfix(saved=saved)

    if validate:
        # this should get the final sample in the full dataset
        validate_waveform(waveforms[-1], static_args, out_dir / 'figures', name=f'sample_idx={n_samples}')
        if detectors is not None:
            validate_projections(projections[-1], static_args, out_dir / 'figures', ifos=ifos, name=f'sample_idx={n_samples}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for waveform generation code.')

    # configuration
    parser.add_argument('-i', '--ifos', type=str, nargs='+', help='The interferometers to project data onto - assumes extrinsic parameters are present.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-p', '--params_file', dest='params_file', default='parameters.csv', type=str, help='The input .csv file of generated parameters to load.')
    parser.add_argument('--add_noise', default=False, action="store_true", help="Whether to add frequency noise - if PSDs are provided we add coloured noise, else Gaussian.")
    
    # data directories
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load parameter files.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('--psd_dir', dest='psd_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    parser.add_argument('--projections_only', default=False, action="store_true", help="Whether to only save projections.npy files and ignore intrinsic waveforms.npy.")
    # to do: add functionality to save noise so we can reconstruct original waveform in visualisations, training, etc.

    # signal processing
    parser.add_argument('--gaussian', default=False, action="store_true", help="Whether to generate white gaussian nois when add_noise is True. If False, coloured noise is generated from a PSD.")
    parser.add_argument('--whiten', default=False, action="store_true", help="Whether to whiten the data with the provided PSD before fitting a reduced basis.")
    parser.add_argument('--bandpass', default=False, action="store_true", help="Whether to truncate the frequency domain data below 'f_lower' specified in the static args.")

    # logging
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode to display progress bars.")
    parser.add_argument('--validate', default=False, action="store_true", help='Whether to validate a sample of the data to check for correctness.')

    # multiprocessing
    parser.add_argument('-c', '--chunk_size', type=int, default=2500, help='The number of workers to use for Python multiprocessing.')
    parser.add_argument('-w', '--workers', type=int, default=8, help='The number of workers to use for Python multiprocessing.')
    
    # # generation
    # parser.add_argument('--seed', type=int, help="Random seed.")  # to do

    args = parser.parse_args()
    # logging.basicConfig(
    #     format='%(process)d-%(levelname)s-%(message)s',
    #     level=getattr(logging, args.logging)
    # )
    
    # if args.workers == -1: args.workers = multiprocessing.cpu_count()
    assert 1 <= args.workers <= multiprocessing.cpu_count(), f"{args.workers} workers are not available."

    # kwargs = {key: val for key, val in args.__dict__.items() if key not in 'logging'}
    generate_waveform_dataset(**args.__dict__)