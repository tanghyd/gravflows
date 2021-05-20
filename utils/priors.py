# Source code adapted from Stephen Green's work in lfigw.
# Available: https://github.com/stephengreen/lfi-gw/blob/master/lfigw/waveform_generator.py
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import numpy as np

# gravitational wave science
import pycbc

def generate_ordered_parameters(spins: bool, spins_aligned: bool, inclination: bool, mass_ratio: bool=False) -> Tuple[str]:
    """Function genereates an ordered tuple of parameters.
    
    The index positions of each parameter in this list must be kept static for downstream tasks.
    """
    if mass_ratio:
        parameters = ['M', 'q']
    else:
        parameters = ['mass_1', 'mass_2']
        
    parameters.extend(['phase', 'time', 'distance'])

    if spins:
        if spins_aligned:
            parameters.extend(['chi_1', 'chi_2'])
        else:
            if not inclination:
                raise Exception('Precession requires nonzero inclination.')
            parameters.extend(['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'])

    if inclination:
        parameters.extend(['theta_jn', 'psi'])

    parameters.extend(['ra', 'dec'])
    
    return tuple(parameters)

def get_parameter_latex_labels():
    return dict(
        mass_1=r'$m_1$',
        mass_2=r'$m_2$',
        M=r'$M$',
        q=r'$q$',
        phase=r'$\phi_c$',
        time=r'$t_c$',
        distance=r'$d_L$',
        chi_1=r'$\chi_1$',
        chi_2=r'$\chi_2$',
        a_1=r'$a_1$',
        a_2=r'$a_2$',
        tilt_1=r'$t_1$',
        tilt_2=r'$t_2$',
        phi_12=r'$\phi_{12}$',
        phi_jl=r'$\phi_{jl}$',
        theta_jn=r'$\theta_{JN}$',
        psi=r'$\psi$',
        ra=r'$\alpha$',
        dec=r'$\delta$'
    )

def uniformity_transform(priors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Function maps a dictionary of list values (typically prior bounds) from
    their original domain to a domain suitable for sampling from a uniform distribution.
    
    Returns a new dictionary of numpy arrays.
    """
    uniform_priors = {}
    
    for key in priors:
        # logic to implement uniformity transformations
        if key == 'distance':
            uniform_priors[key] = priors[key] ** 3
        elif key == 'dec':
            uniform_priors[key] = np.sin(priors[key])
        elif key in ['theta_jn', 'tilt_1', 'tilt_2']:
            uniform_priors[key] = np.cos(priors[key])
        else:
            uniform_priors[key] = priors[key]
            
        # ensure array is sorted with lower and upper bound
        uniform_priors[key] = np.sort(uniform_priors[key])
        
    return uniform_priors

def inverse_uniformity_transform(
    uniform: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Function maps a dictionary of list values (typically prior bounds) from
    a domain suitable for uniform sampling to their original prior domain.
    
    Returns a new dictionary of numpy arrays.
    """
    priors = {}
    
    for key in uniform:
        if key == 'distance':
            priors[key] = uniform[key] ** (1/3)
        elif key == 'dec':
            priors[key] = np.arcsin(uniform[key])
        elif key in ['theta_jn', 'tilt_1', 'tilt_2']:
            priors[key] = np.arccos(uniform[key])
        else:
            priors[key] = uniform[key]
    
        # ensure array is sorted with lower and upper bound
        priors[key] = np.sort(priors[key])
        
    return priors