import os
import argparse
import logging

from pathlib import Path
from typing import List, Union

# local imports
from data.parameters import ParameterGenerator

# to do - put this in on repo
def rm_tree(pth: Union[str, os.PathLike]):
    """Recursively removes all files and folders in a directory."""
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def generate_parameters(
    n: int,
    config_files: Union[List[str], str],
    data_dir: str='data',
    file_name: str='parameters.csv',
    overwrite: bool=False,
    metadata: bool=True,
):
    """Convenience function to generate parameters from a ParameterGenerator
    object and save them to disk as a .csv file.
    
    Can also copy corresponding PyCBC prior distribution metadata for completeness.

    Arguments:
        n: int
            Number of samples to generate.
        config_files: List[str] or str.
            A file path to a compatible PyCBC params.ini config files.
            This can also be a list of config_files (to do: check case if there are duplicates?).
        data_dir: str
            The output directory to save generated parameter data.
        file_name: str
            The output .csv file name to save the generated parameters.
        overwrite: bool=True
            If true, completely overwrites the previous directory specified at data_dir.
        metadata: bool=False
            Whether to copy config file metadata as .ini file to data_dir with same file base name.

    """
    # specify output directory and file
    data_dir = Path(data_dir)
    assert not data_dir.is_file(), f"{data_dir} is a file. It should either not exist or be a directory."
    if overwrite and data_dir.is_dir(): rm_tree(data_dir)  # recursively delete directory -- does not remove parents if nested
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # if not overwrite:
    #     assert not csv_path.is_file(), f"{csv_path} exists but argument overwrite is set to False."
        
    # generate intrinsic parameters
    generator = ParameterGenerator(config_files=config_files, seed=None)
    parameters = generator.draw(n, as_dataframe=True)
    
    # save parameters and metadata
    csv_path = data_dir / file_name  # filename should include .csv extension
    parameters.to_csv(csv_path, index_label='index')
    if metadata:
        with open(data_dir / f'{csv_path.stem}.ini', 'w') as file:
            generator.config_parser.write(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for parameter generation code.')

    parser.add_argument('-n', type=int, help="Number of parameter samples to generate.")
    parser.add_argument('-d', '--data_dir', default='data', dest='data_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('-f', '--file_name', default='parameters.csv', dest='file_name', type=str, help='The output .csv file name to save the generated parameters.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    parser.add_argument(
        '-c', '--config_files', dest='config_files', type=str, action='append', #nargs='+',
        help='A file path to a compatible PyCBC ini config files. Can be called multiple times if there are no duplicates.'
    )
    
    # random seed
    # parser.add_argument('--seed', type=int")  # to do

    # logging
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    # parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode to display progress bars.")
    # parser.add_argument('--validate', default=False, action="store_true", help='Whether to validate a sample of the data to check for correctness.')

    args = parser.parse_args()
    assert args.config_files is not None, ".ini config files must be provided with -c or --config_files."
    
    generate_parameters(**args.__dict__)