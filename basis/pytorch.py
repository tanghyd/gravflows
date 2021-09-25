import os

from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# local imports
from data.config import read_ini_config
from data.noise import get_standardization_factor

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
        basis_dir: Union[str, os.PathLike],
        n: Optional[int]=None,
        dtype=torch.complex128,
    ):
        super().__init__()
        # load reduced basis (randomized_svd: Vh = V.T.conj())
        self.basis_dir = Path(basis_dir)
        basis = np.load(self.basis_dir / 'reduced_basis.npy')
        
        if n is not None:
            # basis truncation
            assert 0 < n < basis.shape[-1]
            basis = basis[:, :, :n]
        
        # self.basis = nn.Parameter(torch.from_numpy(basis[None]))
        self.register_buffer('basis', torch.tensor(basis[None], dtype=dtype, requires_grad=False))
        self.register_buffer('scaler', torch.ones((self.basis.shape[1], self.basis.shape[3]), dtype=dtype, requires_grad=False)[None])
    
    def _generate_coefficients(
        self,
        projections_file: str='projections.npy'
    ):
        # batch process data (especially on GPU with memory limitations)A
        device = list(self.parameters())[0].device
        dtype = list(self.parameters())[0].dtype
        projections = np.load(self.basis_dir / projections_file, mmap_mode='r')
        coefficients = []

        chunk_size = 5000
        chunks = int(np.ceil(len(projections) / chunk_size))

        with torch.no_grad():
            for i in range(chunks):        
                # set up chunking indices
                start = i * chunk_size
                if i == chunks - 1:
                    end = len(projections)
                else:
                    end = (i+1)*chunk_size

                # batch matrix multiplication with pytorch
                waveform = torch.tensor(projections[start:end, :], dtype, device)
                coefficients.append(self(waveform).cpu().numpy())

        return np.concatenate(coefficients, axis=0)
        
    def _fit_scaler(
        self,
        static_args_ini: str,
        coefficients: Optional[np.ndarray]=None,
        projections_file: str='projections.npy',
    ):
        if coefficients is None:
            coefficients = self._generate_coefficients(projections_file)
        _, static_args = read_ini_config(static_args_ini)  # should we store args on nn.Module?
        
        device = list(self.parameters())[0].device
        dtype = list(self.parameters())[0].dtype
        standardization = get_standardization_factor(coefficients, static_args)
        self.scaler = torch.tensor(standardization, dtype=dtype, device=device)
        
    def forward(self, x):
        return torch.einsum('bij, bijk -> bik', x, self.basis) * self.scaler
                        
    def inverse(self, x):
        return torch.einsum(
            'bik, bikj -> bij',
            x/self.scaler,
            torch.transpose(self.basis, 2, 3).conj()
        )