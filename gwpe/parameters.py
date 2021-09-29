
import os
import shutil
import argparse
import time

from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict

# TO DO: Implement logging instead of print statements
# import logging

import numpy as np
import pandas as pd

import pycbc
from pycbc.workflow import WorkflowConfigParser
from pycbc.transforms import apply_transforms, read_transforms_from_config
from pycbc.distributions import (
    JointDistribution, read_params_from_config,
    read_constraints_from_config, read_distributions_from_config
)

# using customised pycbc constraint import
from .utils.pycbc import constraints

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
        
        self.parameters, self.static_args = read_params_from_config(self.config_parser)
        self.constraints = read_constraints_from_config(self.config_parser)
        self.transforms = read_transforms_from_config(self.config_parser)
        self.distribution = JointDistribution(
            self.parameters,
            *read_distributions_from_config(self.config_parser),
            **{'constraints': self.constraints}
        )

        # ensure statistics match output of self.parameters
        self.statistics = compute_parameter_statistics({
            parameter: self.distribution.bounds[parameter]
            for parameter in self.parameters
        })
        
    @property
    def latex(self):
        return [get_parameter_latex_labels()[param] for param in self.parameters]
        
    def draw(self, n: int=1, as_df: bool=False) -> Union[pd.DataFrame, np.record]:
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
        if as_df:
            return pd.DataFrame(samples)
        return samples

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

def compute_parameter_statistics(bounds: Dict[str, pycbc.boundaries.Bounds]):
    """Compute analytical mean and standard deviations for physical parameters
    given (hard-coded) assumptions about their prior distributions.
    
    We use analytic expressions defined by Green and Gair https://arxiv.org/abs/2008.03312.
    These are used to standardize parameters before being input to a neural network.
    """

    statistics = pd.DataFrame(columns=['mean','std'])
    
    # Use analytic expressions
    for param, (left, right) in bounds.items():

        if param == 'mass_1':
            m2left, m2right = bounds['mass_2'].min, bounds['mass_2'].max
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
            m1left, m1right = bounds['mass_1'].min, bounds['mass_1'].max
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

    return statistics


def generate_parameters(
    n: int,
    config_files: Union[List[str], str],
    data_dir: str='data',
    file_name: str='parameters.csv',
    overwrite: bool=False,
    metadata: bool=True,
    verbose: bool=True,
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
    if verbose:
        start = time.perf_counter()

    # specify output directory and file
    data_dir = Path(data_dir)
    assert not data_dir.is_file(), f"{data_dir} is a file. It should either not exist or be a directory."
    if overwrite and data_dir.is_dir():
        shutil.rmtree(data_dir)  # does not remove parents if nested
    data_dir.mkdir(parents=True, exist_ok=True)
        
    # generate intrinsic parameters
    generator = ParameterGenerator(config_files=config_files, seed=None)
    parameters = generator.draw(n, as_df=True)
    
    # save parameters and metadata
    csv_path = data_dir / file_name  # filename should include .csv extension
    parameters.to_csv(data_dir / file_name, index_label='index')
    if metadata:
        config_dir = data_dir / 'config_files'
        config_dir.mkdir(exist_ok=True)
        with open(config_dir / f'{csv_path.stem}.ini', 'w') as file:
            generator.config_parser.write(file)

    if verbose:
        end = time.perf_counter()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generated {n} parameters to {csv_path} in {round(end-start, 4)}s.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for parameter generation code.')

    parser.add_argument('-n', type=int, help="Number of parameter samples to generate.")
    parser.add_argument('-d', '--data_dir', default='data', dest='data_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('-f', '--file_name', default='parameters.csv', dest='file_name', type=str, help='The output .csv file name to save the generated parameters.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action="store_true", help="Sets verbose mode to track runtime.")
    # parser.add_argument('--validate', default=False, action="store_true", help='Whether to validate a sample of the data to check for correctness.')

    parser.add_argument(
        '-c', '--config_files', dest='config_files', type=str, action='append', #nargs='+',
        help='A file path to a compatible PyCBC ini config files. Can be called multiple times if there are no duplicates.'
    )
    
    # random seed
    # parser.add_argument('--seed', type=int")  # to do

    # logging
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")

    args = parser.parse_args()
    assert args.config_files is not None, ".ini config files must be provided with -c or --config_files."
    
    generate_parameters(**args.__dict__)