import os
import h5py
import argparse
import time

from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
from datetime import datetime
from tqdm import tqdm

import numpy as np
from numpy.lib.format import open_memmap

import pandas as pd

from sklearn.utils.extmath import randomized_svd

from lal import REARTH_SI, C_SI

# local imports
from .utils import read_ini_config
from .noise import get_standardization_factor
from .waveforms import get_sample_frequencies

# TO DO: Implement logging instead of print statements
# import logging

def chunked_projection(
    data: np.ndarray,
    basis: np.ndarray,
    chunk_size: int=2500,
    n: Optional[int]=None,
    verbose: bool=True,
) -> np.ndarray:

    # store results
    coefficients = []
    
    # specify n for basis truncation
    if n is not None:
        assert 1 <= n <= basis.shape[1], f"n must be 1 <= n <= {basis.shape[1]}"
    else:
        n = basis.shape[1]  # no truncation

    # batch process data (especially on GPU with memory limitations)
    chunks = int(np.ceil(len(data) / chunk_size))
    desc=f"[{datetime.now().strftime('%H:%M:%S')}] CPU: Reconstructing {n} basis elements"
    with tqdm(total=data.shape[0], desc=desc, disable=not verbose) as progress:
        for i in range(chunks):        
            # set up chunking indices
            start = i * chunk_size
            if i == chunks - 1:
                end = len(data)
            else:
                end = (i+1)*chunk_size
                
            # batch matrix multiplication
            waveform = data[start:end, :]
            coefficients.append((waveform @ basis[:, :n]))
            progress.update(end - start)

    return np.concatenate(coefficients, axis=0)
    
def basis_reconstruction(
    data: np.ndarray,
    basis: np.ndarray,
    chunk_size: int=2500,
    n: Optional[int]=None,
    verbose: bool=True,
) -> Tuple[np.ndarray]:

    # store results
    waveforms = []  # frequency domain waveforms
    reconstructions = []  # freq waveforms --> RB --> freq domain
    
    # specify n for basis truncation
    if n is not None:
        assert 1 <= n <= basis.shape[1], f"n must be 1 <= n <= {basis.shape[1]}"
    else:
        n = basis.shape[1]  # no truncation

    # batch process data (especially on GPU with memory limitations)
    chunks = int(np.ceil(len(data) / chunk_size))
    desc=f"[{datetime.now().strftime('%H:%M:%S')}] CPU: Reconstructing {n} basis elements"
    with tqdm(total=data.shape[0], desc=desc, disable=not verbose) as progress:
        for i in range(chunks):        
            # set up chunking indices
            start = i * chunk_size
            if i == chunks - 1:
                end = len(data)
            else:
                end = (i+1)*chunk_size
                
            # batch matrix multiplication
            waveform = data[start:end, :]
            reconstruction = (waveform @ basis[:, :n]) @ basis.T.conj()[:n, :]
            waveforms.append(waveform)
            reconstructions.append(reconstruction)

            progress.update(end - start)

    waveforms = np.concatenate(waveforms, axis=0)
    reconstructions = np.concatenate(reconstructions, axis=0)

    return waveforms, reconstructions

def compute_mismatch_statistics(waveforms: np.ndarray, reconstructions: np.ndarray) -> Dict[str, float]:
    """Compute statistics comparing the mismatch of values between two arrays.
    
    We compare the ratio of values for each length element (i.e. frequency bin)
    between the two arrays, following similar procedure to Green and Gair (2020).
    """
    norm1 = np.mean(np.abs(waveforms)**2, axis=1)
    norm2 = np.mean(np.abs(reconstructions)**2, axis=1)
    inner = np.mean(waveforms.conj()*reconstructions, axis=1).real

    # if ratio of values are similar matches should tend to 1.
    matches = inner / np.sqrt(norm1 * norm2)
    mismatches = 1 - matches

    statistics = {
        'mean': np.mean(mismatches),
        'std': np.std(mismatches),
        'max': np.max(mismatches),
        'median': np.median(mismatches),
        'perc99': np.percentile(mismatches, 99.),
        'perc99.9': np.percentile(mismatches, 99.9),
        'perc99.99': np.percentile(mismatches, 99.99),
    }

    return statistics

def evaluate_basis_reconstruction(
    data: np.ndarray,
    basis: np.ndarray,
    batch_size: int=5000,
    n: Optional[Union[List[int], int]]=None,
    verbose: bool=True,
) -> Dict[str, float]:
    
    statistics = []
    n = n if isinstance(n, list) else [n]  # loop through multiple truncations
    for num_basis in n:
        waveforms, reconstructions = basis_reconstruction(data, basis, batch_size, num_basis, verbose)
        statistics.append({
            'num_basis': num_basis,
            **compute_mismatch_statistics(waveforms, reconstructions)
        })

    return pd.DataFrame(statistics)

class SVDBasis:
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        static_args_ini: str,
        ifos: List[str]=['H1', 'L1'],
        projections_file: str='projections.npy',
        preload: bool=False,
    ):
        # load static argument file
        _, self.static_args = read_ini_config(static_args_ini)

        # specify output directory and file
        self.data_dir = Path(data_dir)
        assert not self.data_dir.is_file(), (
            f"{self.data_dir} is a file. It should either not exist or be a directory."
        )
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        assert (self.data_dir / projections_file).is_file()
        self.projections_file = projections_file
        self.ifos = ifos
        
        # saved reduced basis elements
        self.V = None
        self.Vh = None
        self.standardization = None
        
        # time translation matrices
        self.dt_grid = None
        self.T_matrices = None
        self.T_matrices_deriv = None
        
        if preload: self.load()
        
    def fit(
        self,
        n: int,
        t_min: float=-0.1,
        t_max: float=0.1,
        Nt: int=1001,
        verbose: bool=True,
        pytorch: bool=True,
    ):
        assert n > 0, f"Number of reduced basis elements n must be greater than 0."
        self.n = n
        self._fit_randomized_svd(n, verbose)
        coefficients = self._generate_coefficients()
        self.standardization = self._fit_standardization(coefficients)
        self._fit_time_translation(t_min, t_max, Nt, verbose=verbose, pytorch=pytorch)

    def _fit_randomized_svd(self, n: int=500, verbose: bool=True):
    
        # load projections from disk as copy-on-write memory mapped array
        projections = np.load(self.data_dir / self.projections_file, mmap_mode='c')  # copy-on-write
        
        self.V = []
        self.Vh = []
        for i, ifo in enumerate(self.ifos):
            if verbose:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Fitting randomized SVD for {ifo} with {n} reduced elements.")
                start = time.perf_counter()

            # reduced basis for projected waveforms
            _, _, Vh = randomized_svd(projections[:, i, :], n)
            
            self.Vh.append(Vh)
            self.V.append(Vh.T.conj())

            if verbose:
                end = time.perf_counter()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {ifo} in {round(end-start, 4)}s.")
                
        self.Vh = np.stack(self.Vh).astype(np.complex64)
        self.V = np.stack(self.V).astype(np.complex64)

    def _generate_coefficients(
        self,
        chunk_size: int=2000,
        verbose: bool=True,
    ):
        """Generate reduced coefficients by batch matrix multpliication of loaded
        projected waveforms and the saved reduced basis V."""
        projections = np.load(self.data_dir / self.projections_file, mmap_mode='c')
        chunks = int(np.ceil(len(projections) / chunk_size))
        
        coefficients = []
        for i in tqdm(
            range(chunks),
            f"[{datetime.now().strftime('%H:%M:%S')}] Projecting waveforms to reduced basis",
            disable=not verbose,
        ):        
            # set up chunking indices
            start = i * chunk_size
            if i == chunks - 1:
                end = len(projections)
            else:
                end = (i+1)*chunk_size

            # batch matrix multiplication
            waveform = np.array(projections[start:end])

            # np.einsum implementation is x100 slower than torch - we use list comp and stack
            coefficients.append(np.stack(
                [(waveform[:, i, :] @ self.V[i]) for i in range(waveform.shape[1])],
                axis=1
            ))

        return np.concatenate(coefficients, axis=0)
        
    def _fit_standardization(
        self,
        coefficients: np.ndarray,
    ):
        return get_standardization_factor(coefficients, self.static_args)

    def _fit_time_translation(
        self,
        t_min: float,
        t_max: float,
        Nt: int=1001,
        sample_frequencies: Optional[np.ndarray]=None,
        verbose: bool=True,
        pytorch: bool=True,
    ):
        """Initialize the time translation matrices according to Green and Gair (2020).
        
        We also follow the same process of adding earth-crossing time.

        The time translation in frequency domain corresponds to multiplication
        by e^{ - 2 pi i f dt }. If we only have waveforms in terms of basis
        coefficients, however, this is quite expensive: first one must
        transform to frequency domain, then time translate, then transform
        back to the reduced basis domain. Generally the dimensionality of
        FD waveforms will be much higher than the dimension of the reduced
        basis, so this is very costly.

        This function pre-computes N x N matrices in the reduced basis domain,
        where N is the dimension of the reduced basis. Matrices are computed
        at a discrete set of dt's. Later, interpolation is used to compute time
        translated coefficients away from these discrete points.

        Arguments:
            sample_frequencies: np.ndarray
                Frequencies at which frequency domain waveforms are evaluated.
            t_min: float
                Minimum value of dt time translation.
            t_max: float
                Maximum value of dt time translation.
            Nt: int
                Number of discrete points at which matrices are pre-computed.
        """
        if sample_frequencies is None:
            sample_frequencies = get_sample_frequencies(
                f_final=self.static_args['f_final'],
                delta_f=self.static_args['delta_f']
            ).astype(np.complex64)

        # Add the earth-crossing time buffer to each side of time
        earth_crossing_time = 2 * REARTH_SI / C_SI
        t_min -= earth_crossing_time
        t_max += earth_crossing_time
 
        # dt_grid are the discrete timesteps for which we want to precompute translations
        self.dt_grid = np.linspace(t_min, t_max, num=Nt, endpoint=True, dtype=np.float32)

        if pytorch:
            import gwpe.pytorch.basis
            # accelerated with torch.einsum (np.einsum is slow, possibly not multi-threaded properly)
            self.T_matrices, self.T_matrices_deriv = gwpe.pytorch.basis.fit_time_translation(
                self.V, self.Vh, self.dt_grid, sample_frequencies, verbose=verbose,
            )

        else:
            # Numpy for loop calculation of time translation matrix for basis elements and its corresponding derivative
            self.T_matrices = np.empty((len(self.ifos), Nt, self.n, self.n), dtype=np.complex64)
            self.T_matrices_deriv = np.empty((len(self.ifos), Nt, self.n, self.n), dtype=np.complex64)

            for i in tqdm(
                range(Nt),
                desc=f"[{datetime.now().strftime('%H:%M:%S')}] Building time translation matrices",
                disable=not verbose,
            ):

                # Translation by dt in FD is multiplication by e^{- 2 pi i f dt}
                # T_fd denotes an array applying time shift in frequency domain
                # T_fd_deriv is therefore the derivative of T_fd.
                T_fd = np.exp(- 2j * np.pi * self.dt_grid[i, None] * sample_frequencies[None])
                T_fd_deriv = - 2j * np.pi * sample_frequencies[None] * T_fd

                # Convert to FD, apply t translation, convert to reduced basis
                T_basis = (self.Vh * T_fd) @ self.V
                T_basis_deriv = (self.Vh * T_fd_deriv) @ self.V

                # insert calculated translation matrix at specified dt index
                self.T_matrices[:, i, :, :] = T_basis
                self.T_matrices_deriv[:, i, :, :] = T_basis_deriv
            
    def batched_time_translate(
        self,
        coefficients: np.ndarray,
        dt: float,
        interpolation: str='linear'
    ):
        """Time translation function based on code written by Green and Gair (2020) to approximate
        a transformation on coefficients of a reduced basis given pre-computed time translation matrices.
        This implementation implements broadcasting and np.einsum to do the operation in batch.

        If the provided dt timestep exists in the time translation matrices, we apply a matrix multiplication.

        Otherwise, we can interpolate the desired dt using the derivatives of the time translations as well.

        Returns a batch of coefficients that have been approximately shifted "dt" in time.

        ~~~~
        
        Calculate basis coefficients for a time-translated waveform.

        The new waveform h_new(t) = h_old(t - dt).

        In other words, if the original merger time is t=0, then the new merger time is t=dt.
        In frequency domain, this corresponds to multiplication by  e^{ - 2 pi i f dt }.

        This method is capable of linear or cubic interpolation.

        Arguments:
            coefficients {array} -- basis coefficients of initial waveform
            dt {float} -- time translation

        Keyword Arguments:
            interpolation {str} -- 'linear' or 'cubic' interpolation
                                   (default: {'linear'})
        Returns:
            array -- basis coefficients of time-translated waveform

        """
        # left-most index position in dt_grid array given provided dt
        pos = np.searchsorted(self.dt_grid, dt, side='right') - 1

        if self.dt_grid[pos] == dt:
            translated = np.einsum('bij, ijk -> bik', coefficients, self.T_matrices[:, pos])

        else:
            t_left, t_right = self.dt_grid[pos: pos+2]

            # Interpolation parameter u(dt) defined so that:
            # u(t_left) = 0, u(t_right) = 1
            u = (dt - t_left) / (t_right - t_left)

            # Require coefficients evaluated on boundaries of interval
            y_left = np.einsum('bij, ijk -> bik', coefficients, self.T_matrices[:, pos])
            y_right = np.einsum('bij, ijk -> bik', coefficients, self.T_matrices[:, pos+1])

            if interpolation == 'linear':

                translated = y_left * (1 - u) + y_right * u

            elif interpolation == 'cubic':

                # Also require derivative of coefficients wrt dt
                dydt_left = np.einsum('bij, ijk -> bik', coefficients, self.T_matrices_deriv[:, pos])
                dydt_right = np.einsum('bij, ijk -> bik', coefficients, self.T_matrices_deriv[:, pos+1])

                # Cubic interpolation over interval
                # See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
                h00 = 2*(u**3) - 3*(u**2) + 1
                h10 = u**3 - 2*(u**2) + u
                h01 = -2*(u**3) + 3*(u**2)
                h11 = u**3 - u**2

                translated = (
                    y_left * h00
                    + dydt_left * h10 * (t_right - t_left)
                    + y_right * h01
                    + dydt_right * h11 * (t_right - t_left)
                )

        return translated

    def save(
        self,
        basis_dir: Optional[Union[str, os.PathLike]]=None,
        filename='reduced_basis.hdf5',
        verbose: bool=True,
    ):
        # whether to save as single or double precision
#         complex_dtype = np.complex64 if downcast else np.complex128
#         real_dtype = np.float32 if downcast else np.float64
        
        # save computed basis statistics to file
        if basis_dir is None:
            basis_dir = self.data_dir
        basis_dir = Path(basis_dir)
        basis_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            start = time.perf_counter()

        with h5py.File(basis_dir / filename, 'w') as file:
            
            # V matrix and corresponding interoferometers for each ifo dim
            file.create_dataset("ifos", data=np.array(self.ifos).astype('S'))
            file.create_dataset(
                "V", data=self.V,#.astype(complex_dtype),
                compression='gzip', compression_opts=9
            )

            # standrdization matrix
            assert self.standardization is not None
            file.create_dataset(
                "standardization",
                data=self.standardization,#.astype(real_dtype),
                compression="gzip"
            )

            # time translation group of data
            assert self.T_matrices is not None
            group = file.create_group("time_translation")
            group.create_dataset(
                "dt_grid",
                data=self.dt_grid,#.astype(real_dtype),
                compression='gzip', compression_opts=9
            )
            
            group.create_dataset(
                "T_matrices",
                data=self.T_matrices,#.astype(complex_dtype),
                compression='gzip', compression_opts=9)
            
            group.create_dataset(
                "T_matrices_deriv",
                data=self.T_matrices_deriv,#.astype(complex_dtype),
                compression='gzip', compression_opts=9
            )   
        
        if verbose:
            end = time.perf_counter()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved reduced basis data to {basis_dir} in {round(end-start, 4)}s.")

    def load(
        self,
        basis_dir: Optional[Union[str, os.PathLike]]=None,
        filename='reduced_basis.hdf5',
        verbose: bool=True,
    ):
        # whether to load as single or double precision
        # complex_dtype = np.complex64 if downcast else np.complex128
        # real_dtype = np.float32 if downcast else np.float64
        
        if basis_dir is None:
            basis_dir = self.data_dir
        basis_dir = Path(basis_dir)
        
        if verbose:
            start = time.perf_counter()

        # load reduced basis elements
        with h5py.File(basis_dir / filename, 'r') as file:
            
            # V matrix and corresponding interoferometers for each ifo dim
            self.ifos = [ifo.decode('utf-8') for ifo in file['ifos'][:]]
            self.V = file['V'][:, :, :]#astype(complex_dtype)
            self.Vh = self.V.T.conj()
            self.n = self.V.shape[-1]
            
            # time translation matrices
            self.dt_grid = file['time_translation']['dt_grid'][:]#.astype(real_dtype)
            self.T_matrices = file['time_translation']['T_matrices'][:, :, :, :]#.astype(complex_dtype)
            self.T_matrices_deriv = file['time_translation']['T_matrices_deriv'][:, :, :, :]#.astype(complex_dtype)

            # standrdization matrix
            self.standardization = file['standardization'][:, :]#.astype(real_dtype)

        if verbose:
            end = time.perf_counter()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded reduced basis data from {basis_dir} in {round(end-start, 4)}s.")

def fit_randomized_svd(
    static_args_ini: str,
    num_basis: int=1000,
    data_dir: str='data',
    out_dir: Optional[str]=None,
    file_name: str='reduced_basis.npy',
    ifos: List[str]=['H1', 'L1'],
    overwrite: bool=False,
    verbose: bool=True,
    validate: bool=False,
    pytorch: bool=True
):
    # specify output directory and file
    data_dir = Path(data_dir)
    assert not data_dir.is_file(), f"{data_dir} is a file. It should either not exist or be a directory."
    if overwrite and data_dir.is_file(): data_dir.unlink(missing_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # fit randomised SVD and save .hdf5
    basis = SVDBasis(data_dir, static_args_ini, ifos, preload=False)
    basis.fit(num_basis, verbose=verbose, pytorch=True)
    basis.save(out_dir, file_name)

    # load projections from disk as copy-on-write memory mapped array
    projections = np.load(data_dir / 'projections.npy', mmap_mode='c')  # copy-on-write

    if validate:
        basis_dir = out_dir / 'reconstruction'
        interval = 100

        for i, ifo in enumerate(ifos):
            # get projected waveform data
            data = projections[:, i, :]

            # to do: we should be able to use np.einsum to accelerate this without GPU? to investigate
            if pytorch:
                import gwpe.pytorch.basis
                # GPU matrix multiplication is fast enough to try multiple basis truncations - is this just faster on torch?
                statistics = gwpe.pytorch.basis.evaluate_basis_reconstruction(data, basis.V[i], n=list(range(100, num_basis+1, interval)))
            else:
                statistics = evaluate_basis_reconstruction(data, basis.V[i], n=list(range(100, num_basis+1, interval)))
            
            # to do: consider better approach to store statistical results
            statistics.to_csv(basis_dir / f'{ifo}_reconstruction.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for randomized SVD fitting code.')
    
    # configuration
    parser.add_argument(
        '-n', '--num_basis', dest='num_basis', default=1000,
        type=int, help="Number of reduced basis elements to fit."
    )
    
    parser.add_argument(
        '-i', '--ifos', type=str, nargs='+', default=['H1', 'L1'],
        help='The interferometers to project data onto - assumes extrinsic parameters are present.'
    )
    
    parser.add_argument(
        '-s', '--static_args', dest='static_args_ini', action='store',
        type=str, help='The file path of the static arguments configuration .ini file.'
    )

    # data directories
    parser.add_argument(
        '-d', '--data_dir', default='datasets/basis', dest='data_dir',
        type=str, help='The output directory to load generated waveform files.'
    )
    
    parser.add_argument(
        '-o', '--out_dir', dest='out_dir',
        type=str, help='The output directory to save generated reduced basis files.'
    )
    
    parser.add_argument(
        '-f', '--file_name', default='parameters.csv', dest='file_name',
        type=str, help='The output .csv file name to save the generated parameters.'
    )
    
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if they already exists.")

    # random seed
    # parser.add_argument('--seed', type=int")  # to do
    # parser.add_argument(
#     "-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
#     help="Set the logging level"
#     )
    
    parser.add_argument(
        '-v', '--verbose', dest='verbose',
        default=False, action="store_true",
        help="Sets verbose mode to display progress bars."
    )

    # validation
    parser.add_argument(
        '--pytorch', default=False, action="store_true",
        help="Whether to use the PyTorch implementation to evaluate basis reconstruction error."
    )
    
    parser.add_argument(
        '--validate', default=False, action="store_true",
        help='Whether to validate a sample of the data to check for correctness.'
    )

    args = parser.parse_args()

    fit_randomized_svd(**args.__dict__)
