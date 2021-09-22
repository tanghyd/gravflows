import os
import argparse
import logging
import time

from pathlib import Path
from typing import Optional, List
from datetime import datetime

import numpy as np
from numpy.lib.format import open_memmap

from sklearn.utils.extmath import randomized_svd

from pycbc.types import load_frequencyseries

# local imports
from utils.config import read_ini_config

# TO DO: Implement logging over print statements
import logging

def fit_reduced_basis(
    static_args_ini: str,
    num_basis: int=1000,
    data_dir: str='data',
    psd_dir: str='data/PSD',
    out_dir: Optional[str]=None,
    file_name: str='reduced_basis.npy',
    ifos: List[str]=['H1', 'L1'],
    whiten: bool=True,
    bandpass: bool=True,
    overwrite: bool=False,
    verbose: bool=True,
):
    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # specify output directory and file
    data_dir = Path(data_dir)
    assert not data_dir.is_file(), f"{data_dir} is a file. It should either not exist or be a directory."
    if overwrite and data_dir.is_file(): data_dir.unlink(missing_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)
    basis_file = out_dir / file_name
    
    # load projections from disk as copy-on-write memory mapped array
    projections = np.load(str(data_dir / 'projections.npy'), mmap_mode='c')
    bandpassed_frequencies = static_args['fd_length'] - int(static_args['f_lower'] / static_args['delta_f'])
    basis = np.empty((len(ifos), bandpassed_frequencies, num_basis), dtype=projections.dtype)
    
    basis = open_memmap(
        filename=basis_file,
        mode='w+',  # create or overwrite file
        dtype=projections.dtype,
        shape=(len(ifos), bandpassed_frequencies, num_basis),
    )

    for i, ifo in enumerate(ifos):
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Fitting randomized SVD for {ifo} with {num_basis} reduced elements.")
            start = time.perf_counter()

        # get projected waveform data
        data = projections[:, i, :]

        # whiten data
        if whiten:
            psd_file = Path(psd_dir) / f'{ifo}_PSD.npy'
            assert psd_file.is_file(), f"{psd_file} does not exist."
            psd = load_frequencyseries(psd_file)
            data /= psd[:static_args['fd_length']] ** 0.5
    
        # filter out values less than f_lower (20Hz)
        if bandpass:
            data = data[:, int(static_args['f_lower'] / static_args['delta_f']):]

        # reduced basis for projected waveforms
        _, _, Vh = randomized_svd(data, num_basis)
        basis[i, :, :] = Vh.T.conj()  # write to memmap array

        if verbose:
            end = time.perf_counter()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {ifo} in {round(end-start, 4)}s.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for randomized SVD fitting code.')
    
    parser.add_argument('-n', '--num_basis', dest='num_basis', default=1000, type=int, help="Number of reduced basis elements to fit.")
    parser.add_argument('-i', '--ifos', type=str, nargs='+', default=['H1', 'L1'], help='The interferometers to project data onto - assumes extrinsic parameters are present.')

    parser.add_argument('-d', '--data_dir', default='datasets/basis', dest='data_dir', type=str, help='The output directory to load generated waveform files.')
    parser.add_argument('-p', '--psd_dir', default='datasets/basis/PSD', dest='psd_dir', type=str, help='The output directory to load generated PSD files.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated reduced basis files.')
    parser.add_argument('-f', '--file_name', default='parameters.csv', dest='file_name', type=str, help='The output .csv file name to save the generated parameters.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')

    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if they already exists.")
    
    # random seed
    # parser.add_argument('--seed', type=int")  # to do
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode to display progress bars.")

    args = parser.parse_args()
    
    fit_reduced_basis(**args.__dict__)