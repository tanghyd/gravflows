from typing import Optional, Union, Tuple, List, Dict
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import torch

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