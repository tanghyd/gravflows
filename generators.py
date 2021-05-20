# import packages
import math
import numpy as np
import multiprocessing

from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

# gravitational wave science
import pycbc

# from pycbc.detector import Detector
# from pycbc.waveform import get_waveform_filter_length_in_time, get_td_waveform, 
from pycbc.waveform import get_fd_waveform

# local imports
from utils.lal import source_frame_to_radiation, is_fd_waveform
from utils.priors import (
    generate_ordered_parameters, generate_psd,
    uniformity_transform, inverse_uniformity_transform,
)

class PSDGenerator:
    
    def __init__(
        self,
        detectors: List[str],
        f_max: float,
        f_min: float,
        event_dir: Optional[Union[Path, str]]=None,
    ):
        detector_list = ('H1', 'L1', 'V1')
        assert all([detector in detector_list for detector in detectors]), (
            f"Provided detectors {detectors} must be in {detector_list}."
        )
        self.detectors = detectors
        self.data = {detector: {} for detector in self.detectors}
        
    def get(
        self,
        detector: str,
        delta_f: float,
    ) -> pycbc.types.FrequencySeries:
        """Return a Power Spectral Density (PSD) at a detector with given delta_f.

        PSDs are stored in a dictionary where keys are integer values corresponding
        to the width of the frequency bin. If the PSD already exists, we retrieve
        it from the dictionary store without re-computing it, otherwise we generate
        it and save it to the PSD dictionary.
        
        Note that PSDs with the same delta_f but which different f_max or f_min
        values will overwrite as this information is not used when saving to dict.

         Arguments:
            detector {str} -- detector name
            delta_f {float} -- frequency spacing for PSD

        Returns:
            psd -- generated PSD
            
        DANIEL TO-DO: 
            Implement a more generalised structure for caching power spectral density data.
        """

        key = int(1.0/delta_f)

        if key not in self.data[detector]:
            self.data[detector][key] = generate_psd(detector, delta_f, self.f_min, self.f_max, self.event_dir)

        return self.data[detector][key]

class WaveformGenerator:
    
    def __init__(
        self,
        spins: bool=True,
        inclination: bool=True,
        spins_aligned: bool=True,
        mass_ratio: bool=False,
        detectors: List[str]=['H1', 'L1', 'V1'],
        domain: str='TD',
        extrinsic_at_train: bool=False,
        num_workers: int=1,
        priors: Optional[Dict[str, Tuple[float]]]=None,
        
    ):  
        "Waveform dataset generator."
        # multiprocessing
        assert num_workers <= multiprocessing.cpu_count()
        self.num_workers = num_workers
        
        # waveform parameterisation
        self.approximant = approximant = 'IMRPhenomPv2' # LAL simulated waveform approximant
        self.spins = spins
        self.spins_aligned = spins_aligned
        self.inclination = inclination
        self.mass_ratio = mass_ratio
        
        # waveforms generated in time/frequency domain or reduced basis via randomised SVD
        domains = ('TD', 'FD', 'RB')
        assert domain in domains, f"{domain} provided not in {domains}"
        self.domain = domain
        
        # whether to apply extrinsic parameters at train or data prep time
        self.extrinsic_at_train = extrinsic_at_train
        self.extrinsic_params = ['time', 'distance', 'psi', 'ra', 'dec']
        
        # Set up indices for parameters
        self.parameters = generate_ordered_parameters(spins, spins_aligned, inclination, mass_ratio)
        
        # default prior ranges if priors not provided
        priors = priors or dict(
            mass_1=[10.0, 80.0],  # solar masses
            mass_2=[10.0, 80.0],
            M=[25.0, 100.0],
            q=[0.125, 1.0],
            phase=[0.0, 2*math.pi],
            time=[-0.1, 0.1],  # seconds
            distance=[100.0, 4000.0],  # Mpc
            chi_1=[-1.0, 1.0],
            chi_2=[-1.0, 1.0],
            a_1=[0.0, 0.99],
            a_2=[0.0, 0.99],
            tilt_1=[0.0, math.pi],
            tilt_2=[0.0, math.pi],
            phi_12=[0.0, 2*math.pi],
            phi_jl=[0.0, 2*math.pi],
            theta_jn=[0.0, math.pi],
            psi=[0.0, math.pi],
            ra=[0.0, 2*math.pi],
            dec=[-math.pi/2.0, math.pi/2.0]
        )
        
        # create dictionary of prior ranges and instantiate them as numpy arrays
        self.priors = {
            key: np.array(value, dtype=np.float64)
            for key, value in priors.items()
            if key in self.parameters
        }
        
        self.psd = PSDGenerator(
            detectors=detectors
        )
        
    def sample_priors(
        self,
        n: int,
        stack: bool=False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Function draws n samples from the prior distributions specified.
        
        Parameter prior ranges are defined on a per-parameter basis. All prior bounds
        are converted via specifics functions so they can be sampled from a uniform
        distribution, and are transformed back with the corresponding inverse function.
        
        Args:
            n: The number of samples.
            stack: Whether to stack the samples in a single 2-D numpy array, or
                whether to return a dictionary of 1-D samples for each parameter.
        
        """
        # generate samples over from a transformed uniform domain and transform back
        samples = inverse_uniformity_transform({
            key: np.random.uniform(*value, size=n).astype(np.float64)
            for key, value in uniformity_transform(self.priors).items()
        })

        # handle mass and mass ratios
        if ('M' in samples.keys()) and ('q' in samples.keys()):
            # reparameterise M and Q to be component masses (unordered)
            samples['mass_1'] = samples['M'] * samples['q']
            samples['mass_2'] = samples['M'] * (1 - samples['q'])

            # recreate samples dictinoary without M and Q (and inserts mass_1 and mass_2 at the front)
            samples = {
                'mass_1': samples['M'] * samples['q'],
                'mass_2': samples['M'] * (1 - samples['q']),
                **{key: val for key, val in samples.items() if key not in ('M','q')}
            }

        # uphold constraint that m1 >= m2 by sorting along the concat dim then splitting
        # note: this approach won't work for OOP-based probability distribution objects
        samples['mass_1'], samples['mass_2'] = np.sort(
            np.stack([samples['mass_1'], samples['mass_2']]),
            axis=0,
        )
        
        if stack:
            samples = np.stack(tuple(samples.values()), axis=1)  # NxF array
        
        return samples
    
    def _generate_whitened_waveform(self, sample: Dict[str, float]):
        assert len(sample) == len(parameters)
        
        # get index position of stored parameters list to retrieve sample
        params = ('mass_1', 'mass_2', 'phase', 'time', 'distance', 'ra', 'dec')
        indices = [self.parameters.index(param) for param in params]
        mass_1, mass_2, phase, coalesce_time, distance, ra, dec = sample[indices]
        
        if self.inclination:
            inclination_indices = [self.parameters.index(param) for param in ('theta_jn', 'psi')]
            theta_jn, psi = sample[inclination_indices]
        else:
            theta_jn, psi = 0., 0.
            
        if self.spins:
            if self.spins_aligned:
                spin_indices = [self.parameters.index(param) for param in ('chi_1', 'chi_2')]
                chi_1, chi_2 = sample[spin_indices]
                iota = theta_jn 
                spin_1x, spin_1y, spin_1z = 0., 0., chi_1
                spin_2x, spin_2y, spin_2z = 0., 0., chi_2
            else:
                spin_params = ('a_1', 'a_2', 'tilt_2', 'tilt_2', 'phi_jl', 'phi_12')
                spin_indices = [self.parameters.index(param) for param in spin_params]
                a_1, a_2, tilt_1, tilt_2, phi_jl, phi_12 = sample[phase_indices]
                
                # use bilby/LAL to simulate an inspiral given intrinsic parameters
                iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = source_frame_to_radiation(
                    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, f_ref, phase
                )
        else:
            iota = theta_jn
            spin_1x, spin_1y, spin_1z = 0.0, 0.0, 0.0
            spin_2x, spin_2y, spin_2z = 0.0, 0.0, 0.0
    
        if self.domain == 'TD':
            # Make sure f_min is low enough
            if time_duration > get_waveform_filter_length_in_time(
                mass1=mass_1, mass2=mass_2,
                spin1x=spin_1x, spin2x=spin_2x,
                spin1y=spin_1y, spin2y=spin_2y,
                spin1z=spin_1z, spin2z=spin_2z,
                inclination=iota,
                f_lower=self.f_min,
                f_ref=self.f_ref,
                approximant=self.approximant
            ):
                print('Warning: f_min not low enough for given waveform duration')

            # get plus polarisation (hp) and cross polarisation (hc) as time domain waveforms
            hp_TD, hc_TD = get_td_waveform(
                mass1=mass_1, mass2=mass_2,
                spin1x=spin_1x, spin2x=spin_2x,
                spin1y=spin_1y, spin2y=spin_2y,
                spin1z=spin_1z, spin2z=spin_2z,
                distance=distance,
                coa_phase=phase,
                inclination=iota,  # CHECK THIS!!!
                delta_t=delta_t,
                f_lower=f_min,
                f_ref=f_ref,
                approximant=approximant
            )
            
            # convert from timet domain to pycbc frequency series
            hp = hp_TD.to_frequencyseries()
            hc = hc_TD.to_frequencyseries()
            
            return hp, hc
            
        elif self.domain in ('FD', 'RB'):
            # LAL refers to approximants by an index 
            if is_fd_waveform(self.approximant):
                # "Use the pycbc waveform generator; change this later" - Stephen Green
                # returns plus and cross phases ("hp", "hc") of the waveform in frequency domain
                plus_waveform, cross_waveform = get_fd_waveform(
                    mass1=mass_1, mass2=mass_2,
                    spin1x=spin_1x, spin2x=spin_2x,
                    spin1y=spin_1y, spin2y=spin_2y,
                    spin1z=spin_1z, spin2z=spin_2z,
                    distance=distance,
                    coa_phase=phase,
                    inclination=iota,
                    f_lower=f_min,
                    f_final=f_max,
                    delta_f=self.delta_f,  # frequency bin width
                    f_ref=f_ref,
                    approximant=self.approximant,
                )

                return plus_waveform, cross_waveform