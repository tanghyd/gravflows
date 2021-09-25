from typing import Optional, Union, Tuple, List, Dict
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

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