from typing import Union, Optional, Tuple, Dict

import numpy as np

from lal import MSUN_SI#, LIGOTimeGPs
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.waveform import (
    get_td_waveform, get_fd_waveform,
    td_approximants, fd_approximants,
)


def source_frame_to_radiation(
    mass_1: float, mass_2: float, phase: float, theta_jn: float, phi_jl: float,
    tilt_1: float, tilt_2: float, phi_12: float, a_1: float, a_2: float, f_ref: float, 
) -> Tuple[float]:
    """Simulates a precessing inspiral of a Binary Black Hole (BBH) system given
    the specified input parameters. Spins are given in the source frame co-ordinates
    and returned in the (x,y,z) radiation frame co-ordinates.
    
    """
    # convert masses from Mpc to SI units
    mass_1_SI = mass_1 * MSUN_SI
    mass_2_SI = mass_2 * MSUN_SI

    # Following bilby code
    if (
        (a_1 == 0.0 or tilt_1 in [0, np.pi])
        and (a_2 == 0.0 or tilt_2 in [0, np.pi])
    ):
        spin_1x, spin_1y, spin_1z = 0.0, 0.0, a_1 * np.cos(tilt_1)
        spin_2x, spin_2y, spin_2z, = 0.0, 0.0, a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1_SI, mass_2_SI, f_ref, phase
            )
        )
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

# from .pycbc import match_precision

def match_precision(data: np.ndarray, real: bool=True):
    """Convenience function returns matching types.
    
    Works for np.ndarrays rather than only pycbc types.

    Arguments:
        real: bool
            If true, returns np.float; else np.complex.
    """
    if data.dtype in (np.float32, np.complex64):
        if real:
            return np.float32
        else:
            return np.complex64
    elif data.dtype in (np.float64, np.complex128):
        if real: 
            return np.float64
        else:
            return np.complex128
    else:
        raise TypeError("Input data array is neither single/double precision or real/complex.")


def generate_intrinsic_waveform(
    sample: Union[np.record, Dict[str, float]],
    static_args: Dict[str, Union[str, float]],
    spins: bool=True,
    spins_aligned: bool=False,
    # inclination: bool=True,
    downcast: bool=True,
    as_pycbc: bool=False,
) -> Union[Tuple[TimeSeries], Tuple[FrequencySeries]]:
    """Function generates a waveform in either time or frequency domain using PyCBC methods.
    
    Arguments:
        intrinsic: Union[np.record, Dict[str, float]]
            A one dimensional vector (or dictionary of scalars) of intrinsic parameters that parameterise a given waveform.
            We require sample to be one row in a because we want to process one waveform at a time,
            but we want to be able to use the PyCBC prior distribution package which outputs numpy.records.
        static_args: Dict[str, Union[str, float]]
            A dictionary of type-casted (from str to floats or str) arguments loaded from an .ini file.
            We expect keys in this dictionary to specify the approximant, domain, frequency bands, etc.
        inclination: bool
        spins: bool        
        spins_aligned: bool
        downcast: bool
            If True, downcast from double precision to full precision.
            E.g. np.complex124 > np.complex64 for frequency, np.float64 > np.float32 for time.
        
    Returns:
        (hp, hc)
            A tuple of the plus and cross polarizations as pycbc Array types
            depending on the specified waveform domain (i.e. time or frequency).
    """
    
    # type checking inputs
    if isinstance(sample, np.record):
        assert sample.shape == (), "input array be a 1 dimensional record array. (PyCBC compatibility)"
        
    for param in ('mass_1', 'mass_2', 'phase'):
        if isinstance(sample, np.record):
            assert param in sample.dtype.names, f"{param} not provided in sample."
        if isinstance(sample, dict):
            assert param in sample.keys(), f"{param} not provided in sample."

    for arg in ('approximant', 'domain','f_lower', 'f_final', 'f_ref'):
        assert arg in static_args, f"{arg} not provided in static_args."
        if arg == 'domain':
            assert static_args['domain'] in ('time', 'frequency'), (
                f"{static_args['domain']} is not a valid domain."
            )
            
    # reference distance - we are generating intrinsic parameters and we scale distance later
    distance = static_args.get('distance', 1)  # func default is 1 anyway
    
    # determine parameters conditional on spin and inclination of CBC
    if spins:
        if spins_aligned:
            iota = sample['theta_jn']  # we don't want to overwrite theta_jn if spins not aligned
            spin_1x, spin_1y, spin_1z = 0., 0., sample['chi_1']
            spin_2x, spin_2y, spin_2z = 0., 0., sample['chi_2']
        else:
            # convert from event frame to radiation frame
            iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = source_frame_to_radiation(
                sample['mass_1'], sample['mass_2'], sample['phase'], sample['theta_jn'], sample['phi_jl'],
                sample['tilt_1'], sample['tilt_2'], sample['phi_12'], sample['a_1'], sample['a_2'], static_args['f_ref'],
            )
    else:
        iota = sample['theta_jn']
        spin_1x, spin_1y, spin_1z = 0., 0., 0.
        spin_2x, spin_2y, spin_2z = 0., 0., 0.
        
    if static_args['domain'] in ('time',):
        # generate time domain waveform
        assert 'delta_t' in static_args, "delta_t not provided in static_args."
        assert static_args['approximant'] in td_approximants(), (
            f"{static_args['approximant']} is not a valid time domain waveform"
        )

        # TO DO: Handle variable length td_waveform generation.
        # ValueError: Time series does not contain a time as early as -8.
        raise NotImplementedError('time domain waveforms not yet implemented.')
        
        # Make sure f_min is low enough
        # if static_args['waveform_length'] > get_waveform_filter_length_in_time(
        #     mass1=sample['mass_1'], mass2=sample['mass_2'],
        #     spin1x=spin_1x, spin2x=spin_2x,
        #     spin1y=spin_1y, spin2y=spin_2y,
        #     spin1z=spin_1z, spin2z=spin_2z,
        #     inclination=iota,
        #     f_lower=static_args['f_lower'],
        #     f_ref=static_args['f_ref'],
        #     approximant=static_args['approximant'],
        #     distance=distance,
        # ):
        #     print('Warning: f_min not low enough for given waveform duration')

        # get plus polarisation (hp) and cross polarisation (hc) as time domain waveforms
        hp, hc = get_td_waveform(
            mass1=sample['mass_1'], mass2=sample['mass_2'],
            spin1x=spin_1x, spin2x=spin_2x,
            spin1y=spin_1y, spin2y=spin_2y,
            spin1z=spin_1z, spin2z=spin_2z,
            coa_phase=sample['phase'],
            inclination=iota,  #  "Check this!" - Stephen Green
            delta_t=static_args['delta_t'],
            f_lower=static_args['f_lower'],
            f_ref=static_args['f_ref'],
            approximant=static_args['approximant'],
            distance=distance,
        )
        
        # Apply the fade-on filter to them - should be timeseries only?
        #     if static_arguments['domain'] == 'time':
        #         h_plus = fade_on(h_plus, alpha=static_arguments['tukey_alpha'])
        #         h_cross = fade_on(h_cross, alpha=static_arguments['tukey_alpha'])

        # waveform coalesces at t=0, but t=0 is at end of array
        # add to start_time s.t. (t=l-ength, t=0) --> (t=0, t=length)
        
        hp = hp.time_slice(-int(static_args['waveform_length']), 0.0)
        hc = hc.time_slice(-int(static_args['waveform_length']), 0.0)
        hp.start_time += static_args['waveform_length']
        hc.start_time += static_args['waveform_length']

        # Resize the simulated waveform to the specified length  ??
        # need to be careful here
        # hp.resize(static_args['original_sampling_rate'])
        # hc.resize(static_args['original_sampling_rate'])
        
        if downcast:
            hp = hp.astype(np.float32)
            hc = hc.astype(np.float32)

    elif static_args['domain'] in ('frequency',):
        # generate frequency domain waveform
        assert 'delta_t' in static_args, "delta_t not provided in static_args."
        assert static_args['approximant'] in fd_approximants(), (
            f"{static_args['approximant']} is not a valid frequency domain waveform"
        )
        
        hp, hc = get_fd_waveform(
            mass1=sample['mass_1'], mass2=sample['mass_2'],
            spin1x=spin_1x, spin2x=spin_2x,
            spin1y=spin_1y, spin2y=spin_2y,
            spin1z=spin_1z, spin2z=spin_2z,
            coa_phase=sample['phase'],
            inclination=iota,  #  "Check this!" - Stephen Green
            delta_f=static_args['delta_f'],
            f_lower=static_args['f_lower'],
            f_final=static_args['f_final'],
            f_ref=static_args['f_ref'],
            approximant=static_args['approximant'],
            distance=distance,
        )
        
        if downcast:
            hp = hp.astype(np.complex64)
            hc = hc.astype(np.complex64)
            
    if as_pycbc:
        return hp, hc
    return hp.data, hc.data

# https://pycbc.org/pycbc/latest/html/_modules/pycbc/detector.html
def project_onto_detector(
    detector: Detector,
    sample: Union[np.recarray, Dict[str, float]],
    hp: Union[np.ndarray, FrequencySeries],
    hc: Union[np.ndarray, FrequencySeries],
    static_args: Dict[str, Union[str, float]],
    sample_frequencies: Optional[np.ndarray]=None,
    as_pycbc: bool=True,
) -> Union[np.ndarray, FrequencySeries]:
    """Takes a plus and cross waveform polarization (i.e. generated by intrinsic parameters)
    and projects them onto a specified interferometer using a PyCBC.detector.Detector.
    """
    # input handling
    assert type(hp) == type(hc), "Plus and cross waveform types must match."
    if isinstance(hp, FrequencySeries):
        assert np.all(hp.sample_frequencies == hc.sample_frequencies), "FrequencySeries.sample_frequencies do not match."
        sample_frequencies = hp.sample_frequencies
    assert sample_frequencies is not None, "Waveforms not FrequencySeries type or frequency series array not provided."
    
    # project intrinsic waveform onto detector
    fp, fc = detector.antenna_pattern(sample['ra'], sample['dec'], sample['psi'], static_args.get('ref_time', 0.))
    h = fp*hp + fc*hc

    # scale waveform amplitude according to ratio to reference distance
    h *= static_args.get('distance', 1)  / sample['distance']  # default d_L = 1

    # Calculate time shift at detector and add to geocentric time
    time_shift_earth_center = sample['time'] - static_args.get('ref_time', 0.)  # default ref t_c = 0
    dt = detector.time_delay_from_earth_center(sample['ra'], sample['dec'], static_args.get('ref_time', 0.))
    time_shift = time_shift_earth_center + dt
    time_shift += (static_args['waveform_length'] / 2)  # put event at centre of window

    time_shift = time_shift.astype(match_precision(sample_frequencies))
    h *= np.exp(- 2j * np.pi * time_shift * sample_frequencies)
        
    # output desired type
    if isinstance(h, FrequencySeries):
        if as_pycbc:
            return h
        else:
            return h.data
    else:
        if as_pycbc:
            return FrequencySeries(h, delta_f=static_args['delta_f'], copy=False)
        else:
            return h

def get_sample_frequencies(f_lower: float=0, f_final: float=1024, delta_f: float=0.25):
    """Constructs sample frequency bins for frequency domain data."""
    num = int((f_final - f_lower) / delta_f)
    return np.linspace(f_lower, f_final, num + 1)

def batch_project(
    detector: Detector,
    extrinsics: Union[np.recarray, Dict[str, np.ndarray]],
    waveforms: np.ndarray,
    static_args: Dict[str, Union[str,float]],
    sample_frequencies: Optional[np.ndarray]=None,
):
    if sample_frequencies is None:
        sample_frequencies = get_sample_frequencies(
            f_final=static_args['f_final'],
            delta_f=static_args['delta_f']
        )
    
    # check waveform matrix inputs
    assert waveforms.shape[-1] == sample_frequencies.shape[0], "delta_f and fd_length do not match."
    assert waveforms.shape[1] == 2, "2nd dim in waveforms must be plus and cross polarizations."
    assert waveforms.shape[0] == len(extrinsics), "waveform batch and extrinsics length must match."
    
    fp, fc = detector.antenna_pattern(
        extrinsics['ra'], extrinsics['dec'], extrinsics['psi'],
        static_args.get('ref_time', 0.)
    )

    projections = fp[:, None]*waveforms[:, 0, :] + fc[:, None]*waveforms[:, 1, :]

    # scale waveform amplitude according to ratio to reference distance
    distance_scale = static_args.get('distance', 1.)  / extrinsics['distance']  # default d_L = 1
    projections *= distance_scale[:, None]

    # Calculate time shift at detector and add to geocentric time
    dt = extrinsics['time'] - static_args.get('ref_time', 0.)  # default ref t_c = 0
    dt += (static_args['sample_length'] / 2)  # put event at centre of window
    dt += detector.time_delay_from_earth_center(
        extrinsics['ra'], extrinsics['dec'], static_args.get('ref_time', 0.)
    )
    dt = dt.astype(match_precision(sample_frequencies, real=True))

    time_shift = np.exp(- 2j * np.pi * dt[:, None] * sample_frequencies[None, :])
    projections *= time_shift

    return projections

# def batch_project(
#     detector: Detector,
#     sample: Union[np.recarray, Dict[str, float]],
#     hp: Union[np.ndarray, FrequencySeries],
#     hc: Union[np.ndarray, FrequencySeries],
#     static_args: Dict[str, Union[str, float]],
#     sample_frequencies: Optional[np.ndarray]=None,
# ) -> np.ndarray:
#     """Takes a plus and cross waveform polarization (i.e. generated by intrinsic parameters)
#     and projects them onto a specified interferometer using a PyCBC.detector.Detector.
#     """
#     # input handling
#     assert type(hp) == type(hc), "Plus and cross waveform types must match."
#     if isinstance(hp, FrequencySeries):
#         assert np.all(hp.sample_frequencies == hc.sample_frequencies), "FrequencySeries.sample_frequencies do not match."
#         sample_frequencies = hp.sample_frequencies
#     assert sample_frequencies is not None, "Waveforms not FrequencySeries type or frequency series array not provided."
    
#     # project intrinsic waveform onto detector
#     fp, fc = detector.antenna_pattern(
#         sample['ra'],
#         sample['dec'],
#         sample['psi'],
#         static_args.get('ref_time', 0.)
#     )
#     h = fp*hp + fc*hc

#     # scale waveform amplitude according to ratio to reference distance
#     h *= static_args.get('distance', 1)  / sample['distance']  # default d_L = 1

#     # Calculate time shift at detector and add to geocentric time
#     time_shift_earth_center = sample['time'] - static_args.get('ref_time', 0.)  # default ref t_c = 0
#     dt = detector.time_delay_from_earth_center(sample['ra'], sample['dec'], static_args.get('ref_time', 0.))
#     time_shift = time_shift_earth_center + dt
#     time_shift += (static_args['waveform_length'] / 2)  # put event at centre of window

#     time_shift = time_shift.astype(match_precision(sample_frequencies))
#     h *= np.exp(- 2j * np.pi * time_shift * sample_frequencies)

