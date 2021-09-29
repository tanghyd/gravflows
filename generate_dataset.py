import os
import shutil
import argparse
import time

import multiprocessing

from typing import Union, Optional, List
from pathlib import Path

# local impotrs
from gwpe.parameters import generate_parameters
from gwpe.waveforms import generate_waveform_dataset
from gwpe.noise import generate_psd
# from gwpe.basis import fit_reduced_basis
from gwpe.basis import SVDBasis


def generate_dataset(
    n: int,
    data_dir: Union[str, os.PathLike],
    config_files: Union[List[str], str],
    static_args_ini: Union[List[str], str],
    ifos: List[str]=['H1','L1'],
    overwrite: bool=True,
    verbose: bool=True,
    workers: int=1,
):
    """Function to generate a dataset of parameters and waveforms with associated metadata.
    
    Arguments:
        n: int
            The number of samples to generate.
        data_dir
            The repository to store saved data files.
        config_files
            A collection of PyCBC .ini files with details on variable_args (i.e. prior distributions).
        overwrite: bool
            Whether to overwrite the pre-existing data directory.
            """
    # specify output directory and file
    # data_dir = Path(data_dir)
    # assert not data_dir.is_file(), f"{data_dir} is a file. It should either not exist or be a directory."
    # if overwrite and data_dir.is_dir():
    #     shutil.rmtree(data_dir)  # does not remove parents if nested
    # data_dir.mkdir(parents=True, exist_ok=True)

    generate_parameters(
        n,
        config_files=config_files,
        data_dir=data_dir,
        file_name='parameters.csv',
        overwrite=overwrite,
        metadata=True,
        verbose=verbose,
    )

    generate_waveform_dataset(
        static_args_ini=static_args_ini,
        data_dir=data_dir,
        params_file='parameters.csv',
        ifos=ifos,
        whiten=True,
        whiten_intrinsics=False,
        projections_only=False,
        downcast=False,
        overwrite=overwrite,
        metadata=True,
        verbose=verbose,
        validate=True,
        time_shift=False,
        workers=workers,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for waveform generation code.')

    # configuration
    parser.add_argument('-i', '--ifos', type=str, nargs='+', help='The interferometers to project data onto - assumes extrinsic parameters are present.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    # parser.add_argument('-p', '--params_file', dest='params_file', default='parameters.csv', type=str, help='The input .csv file of generated parameters to load.')
    # parser.add_argument('--add_noise', default=False, action="store_true", help="Whether to add frequency noise - if PSDs are provided we add coloured noise, else Gaussian.")
    
    # # data directories
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load parameter files.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    # parser.add_argument('--projections_only', default=False, action="store_true", help="Whether to only save projections.npy files and ignore intrinsic waveforms.npy.")
    # # to do: add functionality to save noise so we can reconstruct original waveform in visualisations, training, etc.

    # # signal processing
    # parser.add_argument('--gaussian', default=False, action="store_true", help="Whether to generate white gaussian nois when add_noise is True. If False, coloured noise is generated from a PSD.")
    # parser.add_argument('--whiten', default=False, action="store_true", help="Whether to whiten the data with the provided PSD before fitting a reduced basis.")
    # parser.add_argument('--whiten_intrinsics', default=False, action="store_true", help="Whether to save a version of the intrinsic waveforms that have been whitened by IFO data.")
    # parser.add_argument('--bandpass', default=False, action="store_true", help="Whether to truncate the frequency domain data below 'f_lower' specified in the static args.")
    # parser.add_argument('--shift', default=False, action="store_true", help="Whether to shift the waveform by applying the sampled time shift.")

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

    # if args.workers == -1: args.workers = multiprocessing.cpu_count()
    assert 1 <= args.workers <= multiprocessing.cpu_count(), f"{args.workers} workers are not available."

    # kwargs = {key: val for key, val in args.__dict__.items() if key not in 'logging'}
    generate_waveform_dataset(**args.__dict__)