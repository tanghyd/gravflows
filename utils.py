import math

import torch
from torch.distributions.uniform import Uniform

from typing import Tuple, List

import pycbc.psd

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

class PriorDataset:
    """Contains a database of waveforms from which to train a model.
    """

    def __init__(
        self,
        spins: bool=True,
        inclination: bool=True,
        spins_aligned: bool=True,
        mass_ratio: bool=False,
        detectors: List[str]=['H1', 'L1', 'V1'],
        domain: str='TD',
        extrinsic_at_train: bool=False
    ):
        # configuration
        self.spins = spins
        self.spins_aligned = spins_aligned
        self.inclination = inclination
        self.mass_ratio = mass_ratio
        self.domain = domain
        
        # whether to apply extrinsic parameters at train or data prep time
        self.extrinsic_at_train = extrinsic_at_train
        self.extrinsic_params = ['time', 'distance', 'psi', 'ra', 'dec']
        
        # Set up indices for parameters
        self.parameters = generate_ordered_parameters(spins, spins_aligned, inclination, mass_ratio)
        self.param_idx = {param: i for i, param in enumerate(self.parameters)}
        self.nparams = len(self.parameters)
        
        # Default prior ranges
        self.priors = dict(
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
        
        self.latex = dict(
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
        
        self.psd_names = dict(
            H1='aLIGODesignSensitivityP1200087',
            L1='aLIGODesignSensitivityP1200087',
            V1='AdVDesignSensitivityP1200087',
            ref='aLIGODesignSensitivityP1200087'
        )
        
        self.psd = dict(
            H1={},
            L1={},
            V1={},
            ref={}
        )

        
    @property
    def parameter_labels(self):
        labels = []
        for param in self.param_idx.keys():
            labels.append(self.latex[param])
        return labels
   
    def _sample_prior(self, n):
        # create dictionary of prior ranges and instantiate them as torch tensors
        bounds = {
            param: torch.tensor(self.priors[param], dtype=torch.float64) # device=device
            for param in self.parameters
        }

        # transform prior domain in order to sample form uniform (must be sorted as [low, high])
        bounds = {
            param: (
                value.pow(3).sort().values if param == 'distance'
                else value.sin().sort().values if param == 'dec'
                else value.cos().sort().values if param in ['theta_jn', 'tilt_1', 'tilt_2']
                else value.sort().values
            ) for param, value in bounds.items()
        }

        # create a torch.distributions.uniform.Uniform object for each parameter
        uniform_priors = { param: Uniform(*value, validate_args=True) for param, value in bounds.items()}

        # sample from uniform distributions
        samples = {parameter: distribution.sample([n]) for parameter, distribution in uniform_priors.items()}

        # undo uniformity transformations
        samples = {
            param: (
                value.pow(1/3) if param == 'distance'
                else value.arcsin() if param == 'dec'
                else value.arccos() if param in ['theta_jn', 'tilt_1', 'tilt_2']
                else value
            ) for param, value in samples.items()
        }

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

        # uphold constraint that mass_1 >= mass_2 by sorting along the concatenated dimension then splitting
        # warning: this approach may have some unintended consequences regarding the prior bounds of m1 and m2
        samples['mass_1'], samples['mass_2'] = torch.stack([samples['mass_1'], samples['mass_2']]).sort(dim=0).values

        return samples
    
    def _generate_psd(self, delta_f, f_max, ifo):
        """Generate a PSD. This depends on the detector chosen.

        Arguments:
            delta_f {float} -- frequency spacing for PSD
            ifo {str} -- detector name

        Returns:
            psd -- generated PSD
        """

        # The PSD length should be the same as the length of FD
        # waveforms, which is determined from delta_f and f_max.

        psd_length = int(self.f_max / delta_f) + 1

        if self.event is None:
            psd = pycbc.psd.from_string(
                self.psd_names[ifo],
                psd_length,
                delta_f,
                self.f_min_psd,
            )
        else:
            psd = pycbc.psd.from_txt(
                self.event_dir / (self.psd_names[ifo] + '.txt'),
                psd_length,
                delta_f,
                self.f_min_psd,
                is_asd_file=False
            )

        # To avoid division by 0 when whitening, set the PSD values
        # below f_min and for f_max to the boundary values.
        lower = int(self.f_min_psd / delta_f)
        psd[:lower] = psd[lower]
        psd[-1:] = psd[-2]

        return psd

    def _get_psd(self, delta_f, ifo):
        """Return a PSD with given delta_f.

        Either get the PSD from the PSD dictionary or generate it and
        save it to the PSD dictionary.

         Arguments:
            delta_f {float} -- frequency spacing for PSD
            ifo {str} -- detector name

        Returns:
            psd -- generated PSD
        """

        key = int(1.0/delta_f)

        if key not in self.psd[ifo]:
            self.psd[ifo][key] = self._generate_psd(delta_f, ifo)

        return self.psd[ifo][key]
