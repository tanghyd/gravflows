import os

from pathlib import Path
from typing import Union, List

import numpy as np
from pycbc.catalog import Merger

# local imports
from gwpe.utils import read_ini_config
from gwpe.noise import NoiseTimeline, get_tukey_window, load_psd_from_file
from gwpe.basis import SVDBasis

def generate_gw150914_context(
    n: int,
    noise_dir: Union[str, os.PathLike],
    psd_dir: Union[str, os.PathLike],
    basis_dir: Union[str, os.PathLike],
    static_args_ini: str,
    ifos: List[str]=['H1','L1'],
    verbose: bool=False,
):
    """Function loads GW150914 segment from O1 dataset, applies signal processing steps
    such as whitening and low-pass filtering, then projects to reduced basis coefficients."""

    _, static_args = read_ini_config(static_args_ini)

    basis = SVDBasis(basis_dir, static_args_ini, ifos, file=None, preload=False)
    basis.load(time_translations=False, verbose=verbose)
    if n is not None: basis.truncate(n)

    # get GW150914 Test data
    timeline = NoiseTimeline(data_dir=noise_dir, ifos=ifos)
    strains = timeline.get_strains(
        int(Merger('GW150914').time - static_args['seconds_before_event'] - 1),
        int(static_args['waveform_length'] + 2)
    )

    psds = {}
    for ifo in ifos:
        # coloured noise from psd
        psd_file = Path(psd_dir) / f'{ifo}_PSD.npy'
        assert psd_file.is_file(), f"{psd_file} does not exist."
        psds[ifo] = load_psd_from_file(psd_file, delta_f=static_args['delta_f'])

    start_time = Merger('GW150914').time - static_args['seconds_before_event']
    end_time = Merger('GW150914').time + static_args['seconds_after_event']
    
    coefficients = []
    for i, ifo in enumerate(strains):   
        strains[ifo] = strains[ifo].time_slice(start_time, end_time)
        
        # whiten with settings associated to longer strain
        strains[ifo] = strains[ifo] * get_tukey_window(static_args['sample_length'])  # hann window
        strains[ifo] = strains[ifo].to_frequencyseries(delta_f=static_args['delta_f'])  # fft
        strains[ifo] /= psds[ifo]**0.5    # whiten
        strains[ifo][:int(static_args['f_lower'] / static_args['delta_f'])] = 0.  # lowpass below 20Hz
        strains[ifo] = strains[ifo][:static_args['fd_length']]  # truncate to 1024Hz
        
        # project gw150914 strain to reduced basis
        V = basis.V[0] if basis.V.shape[0] == 1 else basis.V[i]
        coefficients.append(strains[ifo] @ V)
        
    coefficients = np.stack(coefficients)

    # flatten for 1-d residual network input
#     coefficients = np.concatenate([coefficients.real, coefficients.imag], axis=0)
#     coefficients = coefficients.reshape(coefficients.shape[0]*coefficients.shape[1])
    
    return coefficients