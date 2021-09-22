import os
from typing import Tuple, Dict, Union, List, Optional
import numpy as np
import pandas as pd

# gravitational wave science
import pycbc
from pycbc.workflow import WorkflowConfigParser
from pycbc.transforms import apply_transforms, read_transforms_from_config
from pycbc.distributions import (
    JointDistribution, read_params_from_config,
    read_constraints_from_config, read_distributions_from_config
)

 # using customised pycbc constraint import
from .pycbc import constraints

class ParameterGenerator:
    
    def __init__(
        self,
        config_files: Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]],
        seed: Optional[int]=None,
    ):
        """Class to generate gravitational waveform parameters using PyCBC workflow and distribution packages.
        """ 
        if seed is not None: raise NotImplementedError("Reproducible random seed not yet implemented.")

        self.config_files = config_files if isinstance(config_files, list) else [config_files]
        self.config_parser = WorkflowConfigParser(configFiles=self.config_files)
        
        self.var_args, self.static_args = read_params_from_config(self.config_parser)
        self.constraints = read_constraints_from_config(self.config_parser)
        self.transforms = read_transforms_from_config(self.config_parser)
        self.distribution = JointDistribution(
            self.var_args,
            *read_distributions_from_config(self.config_parser),
            **{'constraints': self.constraints}
        )
        
        
    def draw(self, n: int=1, as_dataframe: bool=False) -> Union[pd.DataFrame, np.record]:
        """
        Draw a sample from the joint distribution and construct a
        dictionary that maps the parameter names to the values
        generated for them.
        
        Arguments:
            n: int
                The number of samples to draw.
            as_dataframe: bool
                If True, returns a pd.DataFrame; else a numpy.recarray


        Returns:
            A `dict` containing a the names and values of a set of
            randomly sampled waveform parameters (e.g., masses, spins,
            position in the sky, ...).
        """
        assert n >= 1, "n must be a positive integer."
        samples = apply_transforms(self.distribution.rvs(size=n), self.transforms)
        if as_dataframe:
            return pd.DataFrame(samples)
        return samples

def generate_ordered_parameters(
    spins: bool,
    spins_aligned: bool,
    inclination: bool,
    mass_ratio: bool=False
) -> Tuple[str]:
    """Function genereates an ordered tuple of parameters.
    
    The index positions of each parameter in this list must be kept static for downstream tasks.

    TO DO:
    Remove any dependency on hard-coded positions for parameters.
    Everything should be as key-value pairs or have automated ordering with error handling.
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
    """Parameter descriptions sourced from LALInference.
    
    See: Veich et al. (2015) @ https://arxiv.org/pdf/1409.7215.pdf
    """
    return dict(
        # mass ratio = False
        mass_1=r'$m_1$',  # mass of compact binary 1 (m1 > m2 by convention)
        mass_2=r'$m_2$',  # mass of compact binary 2 (m1 > m2 by convention)
        
        # mass ratio = True
        M=r'$M$',  # chirp mass
        q=r'$q$',  # asymmetric mass ratio
        
        phase=r'$\phi_c$',  # The orbital phase φ_c of the binary at the reference time t_c
        time=r'$t_c$',  # An arbitrary reference time tc, e.g. time of binary coalescence
        distance=r'$d_L$',  # the luminosity distance to the source d_L
        psi=r'$\psi$',  # polarisation angle - orientation of the projection of the binary's orbital momentum vector onto plane on sky
        ra=r'$\alpha$',  # right ascension of the source
        dec=r'$\delta$',  # declination of the source
    
        ## spin aligned
        chi_1=r'$\chi_1$',  # angle each spin vector - orientation w.r.t plane defined by line of sight and initial orbital angular momentum
        chi_2=r'$\chi_2$',  # angle each spin vector - orientation w.r.t plane defined by line of sight and initial orbital angular momentum
        
        # inclination (precession?)
        theta_jn=r'$\theta_{JN}$',  # the inclination of the system's total angular momentum with respect to the line of sight

        ## spins not aligned
        a_1=r'$a_1$',  # dimensionless spin magnitude a = spin / m ^2 in the range [0, 1]
        a_2=r'$a_2$',  # dimensionless spin magnitude a = spin / m ^2 in the range [0, 1]
        tilt_1=r'$t_1$',  # tilt angle between the compact objects' spins and the orbital angular momentum
        tilt_2=r'$t_2$',  # tilt angle between the compact objects' spins and the orbital angular momentum
        phi_12=r'$\phi_{12}$',  #  the complimentary azimuthal angle separating the spin vectors
        phi_jl=r'$\phi_{jl}$',  # the azim of the orbital angular momentum on its cone of precession about the total angular momentum

    )


def compute_parameter_statistics(priors: Dict[str, pycbc.boundaries.Bounds]):
    """Compute analytical mean and standard deviations for physical parameters
    given (hard-coded) assumptions about their prior distributions.
    
    This follows from Green and Gair (2020) where parameters are standardized
    before being input to a neural network - see: https://arxiv.org/abs/2008.03312.

    TO DO: Enable this function to be compatible with PyCBC prior .ini files.
    """

    statistics = pd.DataFrame(columns=['mean','std'])
    
    # Use analytic expressions
    for param, bounds in priors.items():
        # get bounds for prior
        left, right = bounds.min, bounds.max

        if param == 'mass_1':
            m2left, m2right = priors['mass_2'].min, priors['mass_2'].max
            mean = (
                (
                    -3*m2left*(left + right)+ 2*(left**2 + left*right + right**2)
                ) 
                / (3.*(left - 2*m2left + right))
            )
            cov = (
                (
                    (left - right)**2*(
                        left**2 + 6*m2left**2
                        + 4*left*right + right**2
                        - 6*m2left*(left + right)
                    )
                )
                / (18.*(left - 2*m2left + right)**2)
            )
            std = np.sqrt(cov)

        elif param == 'mass_2':
            m1left, m1right = priors['mass_1'].min, priors['mass_1'].max
            mean = ((-3*left**2 + m1left**2 + m1left*m1right + m1right**2)
                    / (3.*(-2*left + m1left + m1right)))
            cov = ((-2*(-3*left**2 + m1left**2
                        + m1left*m1right + m1right**2)**2 +
                    3*(-2*left + m1left + m1right) *
                    (-4*left**3
                     + (m1left + m1right)*(m1left**2 + m1right**2))) /
                   (18.*(-2*left + m1left + m1right)**2))
            std = np.sqrt(cov)

        elif param in (
            'phase', 'time', 'distance',
            'chi_1', 'chi_2', 'a_1', 'a_2',
            'phi_12', 'phi_jk', 'ra', #'psi',
        ):
            # Uniform prior
            mean = (left + right)/2
            std = np.sqrt(((left - right)**2) / 12)

#         elif param == 'distance':
#             # Uniform in distance^3 rather than distance
#             mean = ((3/4) * (left + right) * (left**2 + right**2)
#                     / (left**2 + left*right + right**2))
#             std = np.sqrt((3*((left - right)**2)
#                            * (left**4 + 4*(left**3)*right
#                               + 10*(left**2)*(right**2)
#                               + 4*left*(right**3) + right**4))
#                           / (80.*((left**2 + left*right + right**2)**2)))

        elif param in ('tilt_1', 'tilt_2', 'theta_jn', 'psi'):
            # Uniform in cosine prior
            # Assume range is [0, pi]
            mean = np.pi / 2.0
            std = np.sqrt((np.pi**2 - 8) / 4)

        elif param == 'dec':
            # Uniform in sine prior
            # Assume range for declination is [-pi/2, pi/2]
            mean = 0.0
            std = np.sqrt((np.pi**2 - 8) / 4)

        statistics = statistics.append(pd.Series({'mean': mean, 'std': std}, name=param))

    # TO DO - fix hard coded position compatibility with pycbc .ini config files
    reordered_params = [
        'mass_1', 'mass_2', 'phase', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12',
        'phi_jl', 'theta_jn', 'psi', 'ra', 'dec', 'time', 'distance'
    ]

    return statistics.loc[reordered_params]
