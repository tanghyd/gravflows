import os
import argparse

from pathlib import Path
from typing import Union, Optional, List

import pycbc.psd
# from pycbc.catalog import Merger

# local imports
from utils.config import read_ini_config
from utils.noise import NoiseTimeline, get_tukey_window

# TO DO: Better incorporate real data handling
# DATA_DIR = '/mnt/datahole/daniel/gwosc/O1'

# TO DO: Better incorporate gps time selection
# gps_time = int(Merger('GW150914').time + 2 - psd_window - static_args['waveform_length'])
GPS_TIME = 1126258436  # hard-coded time prior to GW150914

def generate_psd(
    data_dir: Union[str, os.PathLike],
    static_args_ini: str,
    gps_time: int=GPS_TIME,
    psd_window: int=1024,
    ifos: Union[str, List[str]]=['H1','L1'],
    out_dir: Optional[str]=None,
    verbose: bool=False,
):
    """Generates Power Spectral Densities (PSDs) using a welch estimate.
    
    Future work:
    To do: check PSD generation for V1
    To do: Enable multiple PSDs for the same ifo.
    """

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # retrieve strain data from valid windows from .hdf files
    data_dir = Path(data_dir)
    timeline = NoiseTimeline(data_dir, ifos)
    strains = timeline.get_strains(gps_time, psd_window)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ifo in strains:
        psd = pycbc.psd.estimate.welch(
            strains[ifo],
            avg_method='median',
            seg_len=static_args['td_length'], 
            seg_stride=static_args['td_length'],
            window=get_tukey_window(
                static_args['waveform_length'],
                static_args['target_sampling_rate'],
            )
        )

        out_file = out_dir / f'{ifo}_PSD.txt'
        psd.save(out_file)

    if verbose:
        print(f'Saved {list(strains.keys())} PSD .txt files to {str(out_file)}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for power spectral density generation code.')

    parser.add_argument('-d, --data_dir', dest='data_dir', type=str, help='The input directory to load .hdf5 strain data files.')
    parser.add_argument('-o, --out_dir', dest='out_dir', type=str, help='The output directory to save generated power spectral density .txt files.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-i', '--ifos', type=str, nargs='+', default=['H1', 'L1'], help='The interferometers to project data onto - assumes extrinsic parameters are present.')
    # parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    # parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    
    # random seed
    # parser.add_argument('--seed', type=int")  # to do
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode.")

    args = parser.parse_args()
    
    generate_psd(**args.__dict__)