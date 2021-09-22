import os
import json
import argparse
import logging

from pathlib import Path
from typing import List, Dict, Optional, Union
from functools import partial
from tqdm import tqdm

import multiprocessing
import concurrent.futures

import numpy as np
import pandas as pd


import pycbc.psd
from pycbc.catalog import Merger
from pycbc.detector import Detector

# local imports
from utils.config import read_ini_config
from utils.waveforms import (
    generate_intrinsic_waveform,
    batch_project, get_sample_frequencies,
)

def generate_waveform_dataset(
    # n_samples: int,
    static_args_ini: str,
    data_dir: str,
    out_dir: Optional[str]=None,
    params_file: str='parameters.csv',
    file_name: str='waveforms.npy',
    ifos: Optional[List[str]]=None,
    downcast: bool=False,
    workers: int=1,
    chunk_size: int=2000,
    verbose: bool=True,
    overwrite: bool=False,
    metadata: bool=True,
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

    # TO DO: if parameters are not provided

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)
    waveform_file = out_dir / file_name

    if waveform_file.is_file() and not overwrite:
        logging.debug('Overwrite exists. Aborting.')
        return

    if metadata:
        # load static_args.ini and save as .json in dataset
        in_args_path = Path(static_args_ini)
        out_args_path = out_dir / f'{in_args_path.stem}.json'
        if overwrite: out_args_path.unlink(missing_ok=True)
        with open(out_args_path, 'w') as f:
            json.dump(static_args, f)

    # specify precision of output waveforms
    dtype = np.complex128 if not downcast else np.complex64

    # whether to project waveforms onto detectors or keep as intrinsic
    if ifos is not None:
        required_extrinsics = ('ra', 'dec', 'psi')
        for extrinsic in required_extrinsics:
            assert extrinsic in parameters.columns, f"{extrinsic} not in {required_extrinsics}."

        detectors = [Detector(ifo) for ifo in ifos]
        waveform_desc = f'projected ({ifos})'
        sample_frequencies = get_sample_frequencies(
            f_final=static_args['f_final'],
            delta_f=static_args['delta_f']
        )
    else:
        # plus and cross polarizations
        detectors = None
        waveform_desc = 'intrinsic'
        sample_frequencies = None  # only used for projected time shifts

    # multiprocessing generation
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # create buffer in memory to temporarily store before writing to desk
        waveforms = np.empty((chunk_size, 2, static_args['fd_length']), dtype=dtype)

        # loop through samples at approx. 10GB at a time and append to numpy array
        progress_desc = f'Generating {static_args["approximant"]} {waveform_desc} waveforms with {workers} workers'
        saved = 0  # track number of saved arrays for progrss bar
        with tqdm(desc=progress_desc, total=n_samples, miniters=1, postfix={'saved': saved}, disable=not verbose) as progress:
            for i in range(chunks):

                # check if chunk_size goes over length of samples
                if (i+1)*chunk_size <= n_samples:
                    saved += chunk_size
                    samples = parameters.loc[i*chunk_size:((i+1)*chunk_size)-1].to_records()

                else:
                    # overflow batch may not have a full chunk size - we need to re-instantiate waveform array
                    overflow_chunk_size = n_samples % chunk_size
                    saved += overflow_chunk_size
                    waveforms = np.empty((overflow_chunk_size, 2, static_args['fd_length']), dtype=dtype)
                    samples = parameters.loc[i*chunk_size:(i*chunk_size)-1+overflow_chunk_size].to_records()

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

                # batch project for each detector - output should be (batch, ifo, length)
                if detectors is not None:
                    projections = np.stack([
                        batch_project(
                            detector,
                            samples,
                            waveforms,
                            static_args,
                            sample_frequencies,
                        ) for detector in detectors],
                        axis=1
                    )

                    # append to .npy file as binary with no header
                    # https://stackoverflow.com/a/51089256
                    with waveform_file.open(mode='ab') as f:
                        projections.tofile(f)

                else:
                    # TO DO: do we want the option to save intrinsic
                    # waveforms AND projected waveforms in the same run?
                    # consider data storage constraints
                    with waveform_file.open(mode='ab') as f:
                        waveforms.tofile(f)

                # notify timer that batch has been saved
                progress.set_postfix(saved=saved)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for waveform generation code.')

    # configuration files
    parser.add_argument('-d, --data_dir', dest='data_dir', type=str, help='The input directory to load parameter files.')
    parser.add_argument('-o, --out_dir', dest='out_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-p, --params_file', dest='params_file', default='parameters.csv', type=str, help='The input .csv file of generated parameters to load.')
    parser.add_argument('-f, --file_name', dest='file_name', default='waveforms.npy', type=str, help='The output .npy file name to save the generated parameters.')

    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")

    # logging
    parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode to display progress bars.")

    # multiprocessing
    parser.add_argument('--workers', type=int, default=1, help='The number of workers to use for Python multiprocessing.')
    
    # # generation
    
    # parser.add_argument('--seed', type=int, help="Random seed.")  # to do

    parser.add_argument(
        '-i', '--ifos', type=str, nargs='+',
        help='The interferometers to project data onto - assumes extrinsic parameters are present.'
    )

    # check outputs - maybe generate a test plot to see if data was generated correctly?
    # parser.add_argument(
    #     '--check', type=int, nargs='+',
    #     help='Whether to generate visualise plots for samples selected by index position.'
    # )



    args = parser.parse_args()
    logging.basicConfig(
        format='%(process)d-%(levelname)s-%(message)s',
        level=getattr(logging, args.logging)
    )

    assert 1 <= args.workers <= multiprocessing.cpu_count(), f"{args.workers} workers are not available."

    kwargs = {key: val for key, val in args.__dict__.items() if key not in 'logging'}
    generate_waveform_dataset(**kwargs)