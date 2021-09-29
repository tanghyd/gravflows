import os

from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# local imports - this is not handled properly for generate_datasets.sh
from ..basis import SVDBasis

def basis_reconstruction(
    data: torch.Tensor,
    basis: torch.Tensor,
    chunk_size: int=5000,
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
    desc=f"[{datetime.now().strftime('%H:%M:%S')}] GPU: Reconstructing {n} basis elements"
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
            waveforms.append(waveform.cpu())
            reconstructions.append(reconstruction.cpu())

            progress.update(end - start)

    waveforms = torch.cat(waveforms, dim=0)
    reconstructions = torch.cat(reconstructions, dim=0)

    return waveforms, reconstructions


def compute_mismatch_statistics(
    waveforms: torch.Tensor,
    reconstructions: torch.Tensor,
) -> Dict[str, float]:
    """Compute statistics comparing the mismatch of values between two arrays.
    
    We compare the ratio of values for each length element (i.e. frequency bin)
    between the two arrays, following similar procedure to Green and Gair (2020).

    This version of the function utilises PyTorch for GPU programming.
    
    Note: the following should be equivalent to numpy.percentile.
    https://pytorch.org/docs/stable/generated/torch.quantile.html
    """
    norm1 = waveforms.pow(2).abs().mean(dim=1)
    norm2 = reconstructions.pow(2).abs().mean(dim=1)
    inner = (waveforms.conj()*reconstructions).mean(dim=1).real
    
    # if ratio of values are similar matches should tend to 1.
    matches = inner / (norm1 * norm2).sqrt()
    mismatches = 1 - matches

    statistics = {
        'mean': mismatches.mean().item(),
        'std': mismatches.std().item(),
        'max': mismatches.max().item(),
        'median': mismatches.median().item(),
        'perc99': torch.quantile(
            mismatches, torch.tensor([0.99], dtype=mismatches.dtype, device=mismatches.device)
        ).item(),
        'perc99.9': torch.quantile(
            mismatches, torch.tensor([0.999], dtype=mismatches.dtype, device=mismatches.device)
        ).item(),
        'perc99.99': torch.quantile(
            mismatches, torch.tensor([0.9999], dtype=mismatches.dtype, device=mismatches.device)
        ).item(),
        }

    return statistics

def evaluate_basis_reconstruction(
    data: torch.Tensor,
    basis: torch.Tensor,
    cuda: bool=False,
    batch_size: int=5000,
    n: Optional[Union[List[int], int]]=None,
    verbose: bool=True,
) -> Dict[str, float]:

    device = 'cuda' if cuda else 'cpu'
    data = torch.from_numpy(data).to(device)
    basis = torch.from_numpy(basis).to(device)

    statistics = []
    n = n if isinstance(n, list) else [n]  # loop through multiple truncations
    for num_basis in n:
        waveforms, reconstructions = basis_reconstruction(data, basis, batch_size, num_basis, verbose)
        statistics.append({
            'num_basis': num_basis,
            **compute_mismatch_statistics(waveforms, reconstructions)
        })

    return pd.DataFrame(statistics)

def fit_time_translation(
    V: Union[np.ndarray, torch.Tensor],
    Vh: Union[np.ndarray, torch.Tensor],
    dt_grid: np.ndarray,
    sample_frequencies: np.ndarray,
    chunk_size: int=7,  # 1001 % 143 = 0
    verbose: bool=True,
) -> Tuple[np.ndarray]:
    # we cast to single precision asdouble precision for 
    # a (3, 1001, 1000, 1000) double T_matrix alone is ~48GBGB
    Nt = len(dt_grid)
    time_shift = torch.from_numpy(np.exp(-2j * np.pi * dt_grid[:, None] * sample_frequencies[None])).to(torch.complex64)
    time_shift_deriv = (-2j * np.pi * torch.from_numpy(sample_frequencies[None]) * time_shift).to(torch.complex64)
    
    assert len(V.shape) == 3, "V should be of shape [ifo, freq_bin, basis elmenents]"
    n_ifos = V.shape[0]  # number of ifos
    n = V.shape[2]  # number of reduced basis elements
    time_translation = torch.empty((n_ifos, Nt, n, n), dtype=torch.complex64)
    time_translation_deriv = torch.empty((n_ifos, Nt, n, n), dtype=torch.complex64)
    
    # convert to torch tensors
    # torch.einsum is faster than np.einsum by orders of magnitude
    if isinstance(V, np.ndarray): V = torch.from_numpy(V)
    if isinstance(Vh, np.ndarray): Vh = torch.from_numpy(Vh)
    V = V.to(torch.complex64)
    Vh = Vh.to(torch.complex64)

    chunks = int(np.ceil(Nt / chunk_size))
    assert 1 <= chunk_size <= Nt
    for i in tqdm(
        range(chunks),
        desc=f"[{datetime.now().strftime('%H:%M:%S')}] Building time translation matrices with PyTorch",
        disable=not verbose,
    ):
        # setup chunking indices
        start = i * chunk_size
        if (i+1)*chunk_size >= Nt:
            end = Nt
        else:
            end = (i+1)*chunk_size

        # (Vh * time_shift) @ V : for each ifo, for each time step (dt, chunked for memory)
        time_translation[:, start:end] = torch.einsum(
            'tibf, itf, tifl -> itbl',
            Vh.expand(end-start, *Vh.shape),
            time_shift[start:end].expand(Vh.shape[0], end-start, time_shift.shape[-1]),
            V.expand(end-start, *V.shape),
        )

        # (Vh * time_shift_derivative) @ V : for each ifo, for each time step (dt, chunked for memory)
        time_translation_deriv[:, start:end] = torch.einsum(
            'tibf, itf, tifl -> itbl',
            Vh.expand(end-start, *Vh.shape),
            time_shift_deriv[start:end].expand(Vh.shape[0], end-start, time_shift_deriv.shape[-1]),
            V.expand(end-start, *V.shape),
        )

    return time_translation.numpy(), time_translation_deriv.numpy()

class BasisEncoder(nn.Module):
    
    """Custom nn.Module for SVD Reduced Basis layer.
    
    We implement a standardization factor to rescale basis
    coefficients to have unit variance as per Green (2020).
    
    To do:
        Investigate reduced basis details.
        Implement lowpass filtered frequency compatibility?
        Otherwise we can set values in bandpass region to zero.
        When we reproject data --> RB --> data we find non-zero
        values present in the originally lowpassed areas.
    
        This problem does not exist when using double precision!
    
    Arguments:
        basis_file:
            The path to the reduced basis file as a numpy array.
        n: int
            The number of reduced basis elements (used for truncation).
            Must be between 0 < n <= basis.shape[0].
    """
    def __init__(
    
        self,
        static_args_ini: str,
        data_dir: Union[str, os.PathLike],
        projections_file: str='projections.npy',
        ifos: List[str]=['H1', 'L1'],
        downcast: bool=True,
    ):
        super().__init__()
        
        self.basis = SVDBasis(static_args_ini, data_dir, ifos, projections_file, preload=False)
        self.basis.load(downcast=downcast)

        self.V = nn.Parameter(torch.from_numpy(self.basis.V))
        self.register_buffer('standardization', torch.from_numpy(self.basis.standardization))
        
    def time_translate(self, coefficients: torch.Tensor, dt: float, interpolation: str='cubic'):
        # left-most index position in dt_grid array given provided dt
        pos = np.searchsorted(self.basis.dt_grid, dt, side='right') - 1

        if self.basis.dt_grid[pos] == dt:
            translated = torch.einsum(
                'bij, ijk -> bik',
                coefficients, 
                torch.tensor(self.basis.T_matrices[:, pos], coefficients.device)
            )

        else:
            t_left, t_right = self.basis.dt_grid[pos: pos+2]

            # Interpolation parameter u(dt) defined so that:
            # u(t_left) = 0, u(t_right) = 1
            u = (dt - t_left) / (t_right - t_left)

            # Require coefficients evaluated on boundaries of interval
            y_left = torch.einsum(
                'bij, ijk -> bik',
                coefficients,
                torch.tensor(self.basis.T_matrices[:, pos], coefficients.device)
            )
            
            y_right = torch.einsum(
                'bij, ijk -> bik',
                coefficients,
                torch.tensor(self.basis.T_matrices[:, pos+1], coefficients.device)
            )

            if interpolation == 'linear':

                translated = y_left * (1 - u) + y_right * u

            elif interpolation == 'cubic':

                # Also require derivative of coefficients wrt dt
                dydt_left = torch.einsum(
                    'bij, ijk -> bik',
                    coefficients,
                    torch.tensor(self.basis.T_matrices_deriv[:, pos], coefficients.device)
                )

                dydt_right = torch.einsum(
                    'bij, ijk -> bik',
                    coefficients,
                    torch.tensor(self.basis.T_matrices_deriv[:, pos+1], coefficients.device)
                )

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
        
    def forward(self, x, scale: bool=True):
        coefficients = torch.einsum('bij, ijk -> bik', x, self.V)
        if scale: coefficients *= self.standardization
        return coefficients
    
    def inverse(self, x, scale: bool=True):
        if scale:
            x = x / self.standardization
        return torch.einsum('bik, ikj -> bij', x, torch.transpose(self.V, 1, 2).conj())