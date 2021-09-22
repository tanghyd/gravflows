import os
import argparse
import logging

from pathlib import Path
from typing import List, Union


def fit_reduced_basis(
    n: int,
    config_files: Union[List[str], str],
    data_dir: str='data',
    psd_dir: str='data',
    out_dir: str='data',
    file_name: str='reduced_basis.npy',
    overwrite: bool=False,
    # metadata: bool=True,
):
    # specify output directory and file
    data_dir = Path(data_dir)
    assert not data_dir.is_file(), f"{data_dir} is a file. It should either not exist or be a directory."
    if overwrite and data_dir.is_file(): data_dir.unlink(missing_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for randomized SVD fitting code.')

    parser.add_argument('-n', type=int, help="Number of reduced basis elements to fit..")
    parser.add_argument('-d, --data_dir', default='data', dest='data_dir', type=str, help='The output directory to load generated waveform files.')
    parser.add_argument('-d, --data_dir', default='data', dest='data_dir', type=str, help='The output directory to load generated PSD files.')
    parser.add_argument('-o, --out_dir', dest='out_dir', type=str, help='The output directory to save generated reduced basis files.')
    parser.add_argument('-f, --file_name', default='parameters.csv', dest='file_name', type=str, help='The output .csv file name to save the generated parameters.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    
    # random seed
    # parser.add_argument('--seed', type=int")  # to do
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    # parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode to display progress bars.")

    args = parser.parse_args()
    
    # generate_parameters(**args.__dict__)