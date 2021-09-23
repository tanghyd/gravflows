import os
import h5py

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm

from scipy.signal import tukey

import numpy as np
import pandas as pd

from lal import LIGOTimeGPS

import pycbc
from pycbc.frame import read_frame
from pycbc.types import complex_same_precision_as
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.catalog import Catalog

    
def get_tukey_window(
    window_duration: int,
    sampling_rate: int=4096,
    roll_off: float=0.4
):
    alpha = 2 * roll_off / window_duration
    length = int(window_duration * sampling_rate)
    return tukey(length, alpha)

def get_noise_std(window_factor, delta_f):
    """Standard deviation of the whitened noise distribution.
    To have noise that comes from a multivariate *unit* normal
    distribution, you must divide by this factor.
    
    In practice, this means dividing the whitened waveforms by this factor.
    
    In the continuum limit in time domain, the standard deviation of white
    noise would at each point go to infinity, hence the delta_t factor.
    """
    return np.sqrt(window_factor) / np.sqrt(4.0 * delta_f)

def get_standardization_factor(basis: np.ndarray, static_args: Dict[str, float]):
    """ Given a whitened noisy waveform, we want to rescale each component to
    # have unit variance. This is to improve neural network training. The mean
    # should already be zero. - Green
    """
    # estimate noise standardization for reduced basis given windowed data
    tukey_window = get_tukey_window(static_args['sample_length'], static_args['target_sampling_rate'])
    window_factor = np.sum(tukey_window ** 2)
    window_factor /= (static_args['sample_length'] * static_args['target_sampling_rate'])
    noise_std = get_noise_std(window_factor=window_factor, delta_f=static_args['delta_f'])
    
    # Standard deviation of data.
    # Divide by sqrt(2) because we want real
    # and imaginary parts to have unit standard deviation.
    std = np.std(basis, axis=0) / np.sqrt(2)
    return 1 / np.sqrt(std**2 + noise_std**2)

def frequency_noise_from_psd(
    psd: Union[np.ndarray, FrequencySeries],
    n: int=1,
    seed: Optional[int]=None
) -> np.ndarray:
    """ Create noise with a given psd.

    Return noise coloured with the given psd. The returned noise
    FrequencySeries has the same length and frequency step as the given psd.
    Note that if unique noise is desired a unique seed should be provided.

    Parameters
    ----------
    psd : FrequencySeries
        The noise weighting to color the noise.
    n: int
        The number of samples to draw (i.e. batch_size).
    seed : {0, int} or None
        The seed to generate the noise. If None specified,
        the seed will not be reset.

    Returns
    --------
    noise : np.ndarray
        A np.ndarray containing gaussian noise colored by the given psd.
    """

    sigma = 0.5 * (psd / psd.delta_f) ** (0.5)
    if seed is not None: np.random.seed(seed)
    sigma = sigma.numpy()
    dtype = complex_same_precision_as(psd)

    not_zero = (sigma != 0)

    sigma_red = sigma[not_zero]

    # generate batch of frequency noise
    size = (n, sigma_red.shape[0])
    noise_re = np.random.normal(loc=0, scale=sigma_red, size=size)
    noise_co = np.random.normal(loc=0, scale=sigma_red, size=size)
    noise_red = noise_re + 1j * noise_co

    noise = np.zeros((n, len(sigma)), dtype=dtype)
    noise[:, not_zero] = noise_red

    return noise


def load_psd_from_file(
    file: Union[str, os.PathLike],
    delta_f: Optional[float]=None,
    f_lower: Optional[float]=None,
    f_final: Optional[float]=None,
) -> FrequencySeries:
    """Load a psd from file (i.e. from a .txt or .npy file file) and subset
    according to lower and final frequency bands"""
    psd = pycbc.types.load_frequencyseries(file)
    
    if delta_f is not None and psd.delta_f != delta_f:
        # interpolation can cause some issues near edges
        # if we have available data, we should ideally
        # generate an accurate psd from source
        psd = pycbc.psd.interpolate(psd, delta_f)
    
    start = int(f_lower/delta_f)if f_lower is not None else 0
    end = int(f_final/delta_f) if f_final is not None else len(psd)

    return psd[start:end]

def get_strain_from_gwf_files(
    gwf_files: Dict[str, List[Union[str, bytes, os.PathLike]]],
    gps_start: int,
    window: int,
    original_sampling_rate: int=4096,
    target_sampling_rate: int=4096,
    as_pycbc_timeseries: bool=True,
    channel: str='GDS-CALIB_STRAIN',
    check_integrity: bool=True,
):
    assert isinstance(gps_start, int), 'time is not an int'
    assert isinstance(window, int), 'interval_width is not an int'
    assert isinstance(original_sampling_rate, int), 'original_sampling_rate is not an int'
    assert isinstance(target_sampling_rate, int), 'target_sampling_rate is not an int'
    assert (original_sampling_rate % target_sampling_rate) == 0, (
        'Invalid target_sampling_rate: Not a divisor of original_sampling_rate!'
    )
    
    sampling_factor = int(original_sampling_rate / target_sampling_rate)
    samples = defaultdict(list)
    
    for ifo in gwf_files:
        detector_channel = f'{ifo}:{channel}'
        for file_path in gwf_files[ifo]:
            strain = read_frame(
                str(file_path),
                detector_channel,
                start_time=gps_start,
                end_time=gps_start+window,
                check_integrity=check_integrity,
            )
            
            samples[ifo].append(strain[::sampling_factor])
        
        samples[ifo] = np.ascontiguousarray(np.concatenate(samples[ifo]))
        
    if not as_pycbc_timeseries:
        return samples

    else:
        # Convert strain of both detectors to a TimeSeries object
        timeseries = {
            ifo: TimeSeries(
                initial_array=samples[ifo],
                delta_t=1.0/target_sampling_rate,
                epoch=LIGOTimeGPS(gps_start)
            ) for ifo in samples
        }

        return timeseries

def get_strain_from_hdf_files(
    hdf_files: Dict[str, List[Union[str, bytes, os.PathLike]]],
    gps_start: int,
    window: int,
    ifos: Union[str, List[str]]=['H1','L1'],
    original_sampling_rate: int=4096,
    target_sampling_rate: int=4096,
    as_pycbc_timeseries: bool=True,
):
    """
    For a given `gps_start`, select the interval of length
    `interval_width` (centered around `gps_time`) from the HDF files
    specified in `hdf_file_paths`, and resample them to the given
    `target_sampling_rate`.

    Args:
        hdf_file_paths (dict): A dictionary with keys `{'H1', 'L1'}`,
            which holds the paths to the HDF files containing the
            interval around `gps_start`.
        gps_start (int): A (valid) background noise time (GPS timestamp).
        interval_width (int): The length of the strain sample (in
            seconds) to be selected from the HDF files.
        original_sampling_rate (int): The original sampling rate (in
            Hertz) of the HDF files sample. Default is 4096.
        target_sampling_rate (int): The sampling rate (in Hertz) to
            which the strain should be down-sampled (if desired). Must
            be a divisor of the `original_sampling_rate`.
        as_pycbc_timeseries (bool): Whether to return the strain as a
            dict of numpy arrays or as a dict of objects of type
            `pycbc.types.timeseries.TimeSeries`.

    Returns:
        A dictionary with keys `{'H1', 'L1'}`. For each key, the
        dictionary contains a strain sample (as a numpy array) of the
        given length, centered around `gps_time`, (down)-sampled to
        the desired `target_sampling_rate`.
    """

    assert isinstance(gps_start, int), 'time is not an int'
    assert isinstance(window, int), 'interval_width is not an int'
    assert isinstance(original_sampling_rate, int), 'original_sampling_rate is not an int'
    assert isinstance(target_sampling_rate, int), 'target_sampling_rate is not an int'
    assert (original_sampling_rate % target_sampling_rate) == 0, (
        'Invalid target_sampling_rate: Not a divisor of original_sampling_rate!'
    )

    # Compute the resampling factor
    sampling_factor = int(original_sampling_rate / target_sampling_rate)

    # Store the sample we have selected from the HDF files
    samples = defaultdict(list)

    # Read in the HDF file(s) and select the noise sample
    for ifo in ifos:
        for file_path in hdf_files[ifo]:
            with h5py.File(file_path, 'r') as hdf_file:
                # compute array indices of segments given gps_time and metadata
                start_time = int(hdf_file['meta']['GPSstart'][()])  # to do: change variable names
                duration = int(hdf_file['meta']['Duration'][()])

                # to do: optimise nested control loops
                if start_time <= gps_start < (start_time + duration):
                    # if windowed segment starts in file...
                    start_idx = (gps_start - start_time) * original_sampling_rate
                    end_idx = (gps_start - start_time + window) * original_sampling_rate

                else:
                    # concat with the next file overlaps into second segment
                    assert start_time < (gps_start + window) < (start_time + duration)
                    start_idx = 0
                    end_idx = (gps_start - start_time + window) * original_sampling_rate

                # extract and down-sample the segment to the target_sampling_rate
                strain = np.array(hdf_file['strain']['Strain'])
                sample = strain[start_idx:end_idx][::sampling_factor]
                samples[ifo].append(sample)

        samples[ifo] = np.ascontiguousarray(np.concatenate(samples[ifo]))

    if not as_pycbc_timeseries:
        # return dictionary of as numpy arrays
        return samples

    else:
        # Convert strain of both detectors to a TimeSeries object
        timeseries = {
            ifo: TimeSeries(
                initial_array=samples[ifo],
                delta_t=1.0/target_sampling_rate,
                epoch=LIGOTimeGPS(gps_start)
            ) for ifo in ifos
        }

        return timeseries

class NoiseTimeline:

    def __init__(
        self,
        data_dir: Union[str, bytes, os.PathLike],
        ifos: Union[str, List[str]]=['H1','L1'],
        extensions: List[str]=['.hdf','.h5'],
        preload: bool=False,
        verbose: bool=False,
    ):
        """
        
        injection_bitmap = {
            0: np.array([0, 0, 0, 0, 0]),
            19: np.array([1, 0, 0, 1, 1]),  # 1 + 2 + 16
            21: np.array([1, 0, 1, 0, 1]),  # 1 + 4 + 16
            22: np.array([1, 0, 1, 1, 0]),  # 2 + 4 + 16
            23: np.array([1, 0, 1, 1, 1]),  # 1 + 2 + 4 + 16
            29: np.array([1, 1, 1, 0, 1]),  # 1 + 4 + 8 + 16  
            30: np.array([1, 1, 1, 1, 0]),  # 2 + 4 + 8 + 16
            31: np.array([1, 1, 1, 1, 1]),  # 1 + 2 + 4 + 8 + 16
        }
        
        """
        self.verbose = verbose
        self.ifos = [ifos] if isinstance(ifos, str) else ifos
    
        # load hdf files and data quality bits
        self.data_dir = Path(data_dir)  # base directory for observation runs
        if preload:
            self.load_files(extensions)
        else:
            self.hdf_files = None
            self.bit_masks = None

        # build and cache valid timelines depending on quality specification
        self._cache = pd.DataFrame(columns=['window', 'dq_bits', 'inj_bits'])
        self._masks = {}
        # self._default_window = window
        # self.build_timeline(window, dq_bits, inj_bits)  # use default settings

    def load_files(self, extensions: List[str]=['.hdf','.h5']):
        # TO DO: Lazy load hdf file metadata - don't need to read each one.
        self.hdf_files = self._get_hdf_files(extensions)
        self.bit_masks = self._get_bit_masks()

    @property
    def gps_start_time(self) -> int:
        """The GPS start time of the observation run."""
        return min([file['start_time'] for file in self.hdf_files])
    
    @property
    def gps_end_time(self) -> int:
        """The GPS end time of the observation run."""
        return max([file['start_time'] + file['duration'] for file in self.hdf_files])
    
    @property
    def n_entries(self) -> int:
        """The length of the timeline array, which should be equal
        to the number of rounded seconds spanning the dataset files."""
        return self.gps_end_time - self.gps_start_time
        
    def _get_hdf_files(self, extensions: List[str]=['.hdf','.h5']):
        """Read in all metadata in the base data directory (self.data_dir) conditional on file type"""
        hdf_files = []
        for ext in extensions:
            for path in self.data_dir.glob(f'**/*{ext}'):  # walk directory
                if path.is_file():
                    with h5py.File(path, 'r') as f:
                        # [()] (or [...]) extracts scalar quantities from h5 files
                        start_time = f['meta']['GPSstart'][()]
                        ifo = f['meta']['Detector'][()].decode('utf-8')
                        duration = f['meta']['Duration'][()]
                        inj_mask = np.array(f['quality']['injections']['Injmask'], dtype=np.int32)
                        dq_mask = np.array(f['quality']['simple']['DQmask'], dtype=np.int32)

                        assert duration == len(inj_mask) == len(dq_mask), (
                            f'Length of InjMask or DQMask does not match the duration!'
                        )

                        if ifo in self.ifos:
                            hdf_files.append({
                                'file_path': path,
                                'start_time': start_time,
                                'detector': ifo,
                                'duration': duration,
                                'inj_mask': inj_mask,
                                'dq_mask': dq_mask,
                            })

        return sorted(hdf_files, key=lambda file: file['start_time'])

    def _get_bit_masks(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Constructs an object to track data quality masks (dq_mask)
        and injection masks (inj_mask) for strain data at each detector.
        """
        # initialize an empty timeline for each detector
        bit_masks = {
            ifo: {
                'inj_mask': np.zeros(self.n_entries, dtype=np.int32),
                'dq_mask': np.zeros(self.n_entries, dtype=np.int32)
            } for ifo in self.ifos
        }
        
        for hdf_file in self.hdf_files:
            # Map start/end from GPS time to array indices
            idx_start = hdf_file['start_time'] - self.gps_start_time
            idx_end = idx_start + hdf_file['duration']
            
            # Add the information to the correct detector
            ifo = hdf_file['detector']
            bit_masks[ifo]['inj_mask'][idx_start:idx_end] = hdf_file['inj_mask']
            bit_masks[ifo]['dq_mask'][idx_start:idx_end] = hdf_file['dq_mask']

        return bit_masks
        
    def get_file_paths_in_window(self, gps_start: int, window: int):
        """Obtain the file paths that correspond to the time interval specified starting at gps_start."""
        # Keep track of the results, i.e., the paths to the HDF files
        file_paths = defaultdict(list)
        gps_end = gps_start + window

        # Loop over all *sorted* HDF files to find the ones containing the given time
        for hdf_file in self.hdf_files:

            # Get the start and end time for the current HDF file
            start_time = hdf_file['start_time']
            end_time = start_time + hdf_file['duration']

            # Check if the given GPS time falls into the interval of the
            # current HDF file, and if so, store the file path for it
            if start_time < gps_start < end_time:
                file_paths[hdf_file['detector']].append(hdf_file['file_path'])

            elif start_time < gps_end < end_time:
                file_paths[hdf_file['detector']].append(hdf_file['file_path'])

        # If both files were found, we are done!
        if all([det in file_paths.keys() for det in self.ifos]):
            return file_paths
    
    def idx2gps(self, idx: int) -> int:
        """Map an index to a GPS time by correcting for the start time of
        the observation run, as determined from the HDF files.

        Arguments:
            idx: int
                An index of a time series array (covering an observation run).

        Returns the corresponding GPS time.
        """

        return int(idx + self.gps_start_time)

    def gps2idx(self, gps: int) -> int:
        """
        Map an GPS time to an index by correcting for the start time of
        the observation run, as determined from the HDF files.

        Arguments:
            gps : int
                A GPS time belonging to a point in time between
                the start and end of an observation run.

        Returns the corresponding time series index.
        """

        return int(gps - self.gps_start_time)
    
    def build_masks(
        self,
        dq_bits: Tuple[int]=(0, 1, 2, 3),
        inj_bits: Tuple[int]=(0, 1, 2, 4),
        as_array: bool=False,
    ) -> Dict[str, np.ndarray]:
        """Builds boolean masks given data quality and injection bit masks."""
        min_dq = sum([2**i for i in dq_bits])
        ones = np.ones(self.n_entries, dtype=np.int32)
        masks = {
            ifo: np.stack([
                self.bit_masks[ifo]['dq_mask'] > min_dq,
                np.stack([
                    np.bitwise_and(self.bit_masks[ifo]['inj_mask'], np.left_shift(ones, i))
                    for i in inj_bits
                ]).all(axis=0)
            ]).all(axis=0)
            for ifo in self.bit_masks
        }
        
        if as_array:
            return np.stack(list(masks.values()))
        return masks
    
    def build_timeline(
        self,
        window: int=32,
        dq_bits: Tuple[int]=(0, 1, 2, 3),
        inj_bits: Tuple[int]=(0, 1, 2, 4),
        chunk_size: int=100000,
    ) -> np.ndarray:
#         For a given `gps_time`, check if is a valid time to sampleim 
#         noise from by checking if all data points in the interval
#         `[gps_time - window / 2, gps_time + window / 2]` have the specified
#         `dq_bits` and `inj_bits` set.
        """For more information about the `dq_bits` and
            `inj_bits`, check out the website of the GW Open Science
            Center, which explains these for the case of O1:

                https://www.gw-openscience.org/archive/dataset/O1

        Args:
            window : int
                The number of seconds around `gps_time`
                which we also want to be valid (because the sample will
                be an interval).
            dq_bits : Tuple[int]
                The Data Quality Bits which one would like to require 
                (see note above). *For example:* `dq_bits=(0, 1, 2, 3)`
                means that the data quality needs to pass all tests
                up to `CAT3`.
            inj_bits : Tuple[int]s
                The Injection Bits which one would like to require
                (see note above). *For example:* `inj_bits=(0, 1, 2, 4)`
                means that only continuous wave (CW) injections are permitted;
                all recordings containing any of other type of injection
                will be invalid for sampling.

        Returns:
            A boolean array - `True` if the data is valid, otherwise `False`.
        """

        assert isinstance(window, int) and window >= 0, 'Received an invalid int for window!'
        assert set(dq_bits).issubset(set(range(7))),'Invalid Data Quality bit specification!'
        assert set(inj_bits).issubset(set(range(5))), 'Invalid Injection bit specification!'
        
        match = self._cache[
            (self._cache['window'] == window) & 
            (self._cache['dq_bits'] == dq_bits) & 
            (self._cache['inj_bits'] == inj_bits)
        ]
        
        assert len(match) in (0, 1), "Duplicated detected in timeline cache!"

        if len(match) == 1:
            timeline = self._masks[match.index.item()]
        else:
            # build data quality masks for each detector (window independent)
            masks = self.build_masks(dq_bits, inj_bits, as_array=True)

            timeline = np.zeros((len(masks), self.n_entries-window+1), dtype=bool)
            assert 0 < chunk_size < (self.n_entries-window+1), "chunk_size must be smaller than length of timeline."
            n_chunks = int(np.ceil((self.n_entries-window+1) / chunk_size))

            mask_buffer = np.stack([np.arange(window, dtype=np.int32)+i for i in range(chunk_size)])
            with tqdm(
                total=self.n_entries,
                desc=f'Processing timeline windows',
                disable=not self.verbose,
            ) as progress:
                # loop through generator that chunks timeline array
                chunker = chunk_counter(self.n_entries, n_chunks, chunk_size, window)
                for start, end in chunker:
                    
                    # edit timeline mask for all (:) detectors
                    timeline[:, start:end] = masks[:, mask_buffer[:end-start, :]].all(axis=2)
                    
                    progress.update(end-start)  # update tqdm iters
                    progress.refresh()
                    mask_buffer += chunk_size  # increment buffer matrix for next chunk
                    
                # append with "deadzone" masks for completeness (dead_zone not long enough for full window)
                dead_zone = np.stack([np.array([False]*(window-1)) for _ in range(masks.shape[0])])
                timeline = np.concatenate([timeline, dead_zone], axis=1).all(axis=0)  # np.all down ifo dim
                progress.update(window-1)
                
            # Get GPS times of all confirmed mergers and filter if within delta_t of event time
            catalog = Catalog()
            real_event_times = [merger.time for merger in catalog.mergers.values()]
            event_mask = np.array([
                list(range(self.gps2idx(event_time - (window/2)), self.gps2idx(event_time + (window/2))))
                for event_time in real_event_times if self.gps_start_time < event_time < self.gps_end_time
            ])

            timeline[event_mask] = False

            # add timeline to cache
            metadata = [{'window': window, 'dq_bits': dq_bits, 'inj_bits': inj_bits}]
            self._cache = self._cache.append(metadata, ignore_index=True)
            self._masks[self._cache.index[-1]] = timeline
            
        return timeline

    def sample_times(
        self,
        n: int,
        window: Optional[int]=None,
        dq_bits: Tuple[int]=(0, 1, 2, 3),
        inj_bits: Tuple[int]=(0, 1, 2, 4),
        gps_start_time: Optional[int]=None,
        gps_end_time: Optional[int]=None,
        as_gps_times: bool=True,
    ) -> Dict[str, np.ndarray]:
        """
        Randomly sample a time (or index) from `[gps_start_time, gps_end_time]`
        which passes the :func:`NoiseTimeline.is_valid()` test.

        Args:
            delta_t (int): For an explanation, see
                :func:`NoiseTimeline.is_valid()`.
            dq_bits (tuple): For an explanation, see
                :func:`NoiseTimeline.is_valid()`.
            inj_bits (tuple): For an explanation, see
                :func:`NoiseTimeline.is_valid()`.
            gps (bool): Whether or not to return the gps_time
                for samples. If False, returns index position of timeline.
            return_paths (bool): Whether or not to return the paths to
                the HDF files containing the `gps_time`.

        Returns:
            An array specifying valid index positions (or GPS times)
            for the timeline of the dataset.
        """
        window = window
        timeline = self.build_timeline(window, dq_bits, inj_bits)

        # subset timeline based on optional (start, end) gps times
        start = gps_start_time or self.gps_start_time
        end = gps_end_time or self.gps_end_time
        assert start >= self.gps_start_time
        assert end <= self.gps_end_time

        timeline[:self.gps2idx(start)] = False
        timeline[self.gps2idx(end):] = False

        # uniformally sample from valid timeline indices
        indices = np.random.choice(timeline.nonzero()[0], n)

        if as_gps_times:
            return indices + self.gps_start_time

        return indices

    def get_strains(
        self,
        gps_start: int,
        window: int,
        original_sampling_rate: int=4096,
        target_sampling_rate: int=4096,
        as_pycbc_timeseries: bool=True,   
    ) -> Dict[str, Union[np.ndarray, TimeSeries]]:

        if self.hdf_files is None:
            self.load_files()

        hdf_files = self.get_file_paths_in_window(gps_start, window)

        return get_strain_from_hdf_files(
            hdf_files=hdf_files,
            gps_start=gps_start,
            window=window,
            ifos=self.ifos,
            original_sampling_rate=original_sampling_rate,
            target_sampling_rate=target_sampling_rate,
            as_pycbc_timeseries=as_pycbc_timeseries
        )


def chunk_counter(n_entries, n_chunks, chunk_size, window):
    """Utility function to yield (start, end) indices for chunked array windowing.
    
    To Do: Multiprocessing for noise timeline ?
    """
    for i in range(n_chunks):
        start = i*chunk_size
        if i == n_chunks-1:
            end = start + (n_entries-window+1) - (i*chunk_size)
        else:
            end = (i+1)*chunk_size
        yield start, end
