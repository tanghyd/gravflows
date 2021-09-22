"""
samplegen - Original authors: (1) Timothy Gabbard; (2) Damon Beveridge.

Provide functions for reading and parsing configuration files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import json
import copy

from typing import Union, Tuple, Dict

from pycbc.workflow import WorkflowConfigParser
from pycbc.distributions import read_params_from_config

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------
"""
Provide tools that are needed for amending and typecasting the static
arguments from an `*.ini` configuration file, which controls the
waveform simulation process.
"""

def amend_static_args(static_args: Dict[str, str]):
    """Amend the static_args from the `*.ini` configuration file by adding
    the parameters that can be computed directly from others (more
    intuitive ones). Note that the static_args should have been
    properly typecast first; see :func:`typecast_static_args()`.

    Args:
        static_args: dict
        The static_args dict after it has been typecast
        by :func:`typecast_static_args()`.

    Returns:
        The amended `static_args`, where implicitly defined variables
        have been added.
    """

    # Create a copy of the original static_args
    args = copy.deepcopy(static_args)
    
    # If necessary, compute the sample length
    if 'sample_length' not in args.keys():
        args['sample_length'] = (
            args['seconds_before_event'] + args['seconds_after_event']
        )

    # If necessary, add delta_t = 1 / target_sampling_rate
    if 'delta_t' not in args.keys():
        args['delta_t'] = 1.0 / args['target_sampling_rate']

    # If necessary, add delta_f = 1 / waveform_length
    if 'delta_f' not in args.keys():
        args['delta_f'] = 1.0 / args['waveform_length']

    # If necessary, add td_length = waveform_length * target_sampling_rate
    if 'td_length' not in args.keys():
        args['td_length'] = int(
            args['waveform_length'] * args['target_sampling_rate']
        )

    # If necessary, add fd_length = td_length / 2 + 1
    if 'fd_length' not in args.keys():
        if 'f_final' in args.keys():
            args['fd_length'] = int(args['f_final'] / args['delta_f']) + 1
        else:
            args['fd_length'] = int(args['td_length'] / 2.0 + 1)

    return args


def typecast_static_args(static_args: Dict[str, str]):
    """
    Take the `static_args` dictionary as it is read in from the PyCBC
    configuration file (i.e., all values are strings) and cast the
    values to the correct types (`float` or `int`).

    Args:
        static_args (dict): The raw `static_args` dictionary as it is
            read from the `*.ini` configuration file.
            
    Returns:
        The `static_args` dictionary with proper types for all values.
    """

    # list variables that must be casted to integers
    # note: whitening_segment_duration was previously a float
    int_args = [
        'bandpass_lower', 'bandpass_upper',
        'waveform_length', 'noise_interval_width',
        'original_sampling_rate', 'target_sampling_rate',
        'whitening_segment_duration', 'whitening_max_filter_duration',
    ]


    # list variables that must be casted to floats
    float_args = [
        'distance', 'f_lower', 'seconds_before_event',
        'seconds_after_event', 'whitening_segment_duration'
    ]

    # copy dictionary with type cast conversions
    args = copy.deepcopy(static_args)

    for float_arg in float_args:
        if float_arg in args:
            args[float_arg] = float(args[float_arg])
        
    for int_arg in int_args:
        if int_arg in args:
            args[int_arg] = float(args[int_arg])

    return args

def read_ini_config(
    file_path: Union[str, os.PathLike],
) -> Tuple[dict, dict]:
    """
    Read in a `*.ini` config file, which is used mostly to specify the
    waveform simulation (for example, the waveform model, the parameter
    space for the binary black holes, etc.) and return its contents.
    
    Args:
        file_path (str): Path to the `*.ini` config file to be read in.

    Returns:
        A tuple `(variable_arguments, static_arguments)` where
            - `variable_arguments` should simply be a list of all the
            parameters which get randomly sampled from the specified
            distributions, usually using an instance of
            :class:`utils.waveforms.WaveformParameterGenerator`.
            - `static_arguments` should be a dictionary containing the keys
            and values of the parameters that are the same for each
            example that is generated (i.e., the non-physical parameters
            such as the waveform model and the sampling rate).
    """
    
    # Make sure the config file actually exists
    if not os.path.exists(file_path):
        raise IOError(f'Specified configuration file does not exist: {file_path}')
    
    # Set up a parser for the PyCBC config file
    config_parser = WorkflowConfigParser(configFiles=[file_path])
    
    # Read the variable_arguments and static_arguments using the parser
    variable_arguments, static_arguments = read_params_from_config(config_parser)
    
    # Typecast and amend the static arguments
    static_arguments = typecast_static_args(static_arguments)
    static_arguments = amend_static_args(static_arguments)
    
    return variable_arguments, static_arguments


def read_json_config(file_path: Union[str, os.PathLike]) -> Tuple[dict, dict]:
    """
    Read in a `*.json` config file, which is used to specify the
    sample generation process itself (for example, the number of
    samples to generate, the number of concurrent processes to use,
    etc.) and return its contents.
    
    Args:
        file_path: Union[str, os.PathLike]
            Path to the `*.json` config file to be read in.

    Returns:
        A `dict` containing the contents of the given JSON file.
    """
    
    # Make sure the config file actually exists
    if not os.path.exists(file_path):
        raise IOError(f'Specified configuration file does not exist: {file_path}')
    
    # Open the config while and load the JSON contents as a dict
    with open(file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define the required keys for the config file in a set
    required_keys = {
        'background_data_directory', 'dq_bits', 'inj_bits',
        'waveform_params_file_name', 'max_runtime',
        'n_injection_samples', 'n_noise_samples', 'n_processes',
        'random_seed', 'output_file_name', 'n_template_samples'
    }
    
    # Make sure no required keys are missing
    missing_keys = required_keys.difference(set(config.keys()))
    if missing_keys:
        raise KeyError('Missing required key(s) in JSON configuration file: '
                       '{}'.format(', '.join(list(missing_keys))))

    return config
