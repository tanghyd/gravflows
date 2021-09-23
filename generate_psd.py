import os
import argparse

from pathlib import Path
from typing import Union, Optional, List, Dict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import pycbc.psd
from pycbc.catalog import Merger

# local imports
from data.config import read_ini_config
from data.noise import NoiseTimeline, get_tukey_window

# TO DO: Implement logging over print statements
import logging

# TO DO: Better incorporate gps time selection
# gps_time = int(Merger('GW150914').time + 2 - psd_window - static_args['waveform_length'])
GPS_TIME = 1126258436  # hard-coded time prior to GW150914

def validate_psds(
    strains: Dict[str, np.ndarray],
    psds: Dict[str, np.ndarray],
    out_dir: Union[str, os.PathLike],
    gps_time: Optional[float]=None,
):
    # full names for plotting
    interferometers = {'H1': 'Hanford', 'L1': 'Livingston', 'V1': 'Virgo', 'K1': 'KAGRA'}

    # output figure directory
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # strain timeseries at point in time
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
    for ifo in strains:
        ax.plot(strains[ifo].sample_times, strains[ifo], label=interferometers[ifo], alpha=0.6)
        
    ax.set_title(f'Strain Data for Power Spectral Density Estimation')
    ax.set_xlabel('GPS Time (s)')
    ax.set_ylabel('Strain')  # units?
    ax.legend(loc='upper left')
    ax.grid('both')

    if gps_time is not None:
        ax.axvline(gps_time, color='r', linestyle='--')  # GW150914 merger time marker
        ax.set_xticks([gps_time], minor=True)  # add low frequency cutoff to ticks
        ax.set_xticklabels(['$t_{c}$'], minor=True, color='r')

    fig.tight_layout()
    fig.savefig(out_dir / 'strain.png')
    fig.show()

    # visualise power spectral densities
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    
    for ifo in strains:
        ax.plot(psds[ifo].sample_frequencies, psds[ifo], label=interferometers[ifo], alpha=0.6)
        
    ax.set_title(f'Estimated Power Spectral Density Prior to GW150914')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude $Hz^{-1}$')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid('both')

    ax.set_xlim((20, 1024))
    ax.set_ylim((1e-47, 1e-39))

    fig.tight_layout()
    fig.savefig(out_dir / 'power_spectral_density.png')
    fig.show()


def generate_psd(
    data_dir: Union[str, os.PathLike],
    static_args_ini: str,
    gps_time: int=GPS_TIME,
    psd_window: int=1024,
    ifos: Union[str, List[str]]=['H1','L1'],
    out_dir: Optional[str]=None,
    verbose: bool=False,
    validate: bool=False,
):
    """Generates Power Spectral Densities (PSDs) using a welch estimate.
    
    Future work:
    To do: check PSD generation for V1
    To do: Enable multiple PSDs for the same ifo.
    """

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # specify output directory
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving {ifos} PSDs to {out_dir}/") 

    # retrieve strain data from valid windows from .hdf files
    timeline = NoiseTimeline(data_dir, ifos)
    strains = timeline.get_strains(gps_time, psd_window)

    psds = {}
    for ifo in strains:
        psds[ifo] = pycbc.psd.estimate.welch(
            strains[ifo],
            avg_method='median',
            seg_len=static_args['td_length'], 
            seg_stride=static_args['td_length'],
            window=get_tukey_window(
                static_args['waveform_length'],
                static_args['target_sampling_rate'],
            )
        )

        out_file = out_dir / f'{ifo}_PSD.npy'
        psds[ifo].save(out_file)

        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {ifo} PSD to {str(out_file)}.")

    if validate:
        validate_psds(strains, psds, out_dir=out_dir / 'figures', gps_time=Merger('GW150914').time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for power spectral density generation code.')

    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load .hdf5 strain data files.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated power spectral density .npy files.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-i', '--ifos', type=str, nargs='+', default=['H1', 'L1'], help='The interferometers to project data onto - assumes extrinsic parameters are present.')
    # parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    # parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    
    # random seed
    # parser.add_argument('--seed', type=int")  # to do
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode.")
    parser.add_argument('--validate', default=False, action="store_true", help='Whether to validate a sample of the data to check for correctness.')

    args = parser.parse_args()
    
    generate_psd(**args.__dict__)