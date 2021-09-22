import os

from typing import Union, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import pycbc.psd
from pycbc.detector import Detector

# local imports
from .noise import NoiseTimeline, frequency_noise_from_psd
from .waveforms import get_tukey_window

# PyTorch DataSet
class BasisDataset(Dataset):
    def __init__(
        self,
        basis_dir: Union[str, os.PathLike],
        static_args: Dict[str, Union[str, float]],
        file_name: str='basis.npy',
    ):
        # load settings and files
        self.basis_dir = Path(basis_dir)
        self.static_args = static_args
        
        # load ground truth parameters
        self.intrinsics = pd.read_csv(self.basis_dir / 'intrinsics.csv', index_col=0)
        self.extrinsics = pd.read_csv(self.basis_dir / 'extrinsics.csv', index_col=0)
        self.parameters = self.intrinsics.join(self.extrinsics)
        self.n = len(self.intrinsics)
        self.file_name = file_name
        self.data = None
        
    def _worker_init_fn(self, worker_id: int=None):
        self.data = np.load(self.basis_dir / self.file_name, mmap_mode='r')
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (np.array(self.data[idx]), self.parameters.iloc[idx].values)

# PyTorch DataSet
class StrainDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        waveform_dir: Union[str, os.PathLike],
        static_args: Dict[str, Union[str, float]],
        psd_time: Union[int, List[int]],
        psd_window: int=1024,
        intrinsics: Optional[pd.DataFrame]=None,
        extrinsics: Optional[pd.DataFrame]=None,
        ifos: List[str]=['H1','L1'],
        dtype: type=np.complex128,
    ):
        # load settings and files
        self.data_dir = Path(data_dir)
        self.waveform_dir = Path(waveform_dir)
        self.static_args = static_args
        self.dtype = dtype
        
        # load ground truth parameters, if provided
        if intrinsics is not None:
            self.intrinsics = intrinsics
        else:
            self.intrinsics = pd.read_csv(self.waveform_dir / 'intrinsics.csv', index_col=0)
        assert isinstance(self.intrinsics, pd.DataFrame)
        self.n = len(self.intrinsics)
            
        if extrinsics is not None:
            self.extrinsics = extrinsics
        else:
            self.extrinsics = pd.read_csv(self.waveform_dir / 'extrinsics.csv', index_col=0)
        
        # detector and PSD metadata
        self.ifos = ifos
        self.timeline = NoiseTimeline(data_dir=self.data_dir, ifos=ifos)
        self.psd_time = psd_time
        self.psd_window = psd_window
        self.sample_frequencies = np.linspace(
            start=0, stop=static_args['f_final'],
            num=int(static_args['f_final'] / static_args['delta_f']) + 1,
            dtype=np.float32 if self.dtype==np.complex64 else np.float64
        )
        
        # data is loaded from worker_init_fn
        self.psds = None 
        self.waveforms = None
        
    def _worker_init_fn(self, worker_id: int=None):
        # memory map intrinsic waveform data on disk
        self.detectors = {ifo: Detector(ifo) for ifo in self.ifos}
        self.waveforms = np.memmap(
            filename=str(self.waveform_dir / 'waveforms.npy'),
            dtype=self.dtype,  # header-less .npy bytes require dtype specification
            mode='r',
            shape=(self.n, 2, int(self.static_args['f_final'] / self.static_args['delta_f']) + 1)
        )
        
        # calculate PSDs using welch method
        strains = self.timeline.get_strains(self.psd_time, self.psd_window)

        # PSDs should be in double precision as we may have values
        # on the order of 1e-50.
        self.psds = {
            ifo: pycbc.psd.estimate.welch(
                strains[ifo],
                avg_method='median',
                seg_len=self.static_args['td_length'], 
                seg_stride=self.static_args['td_length'],
                window=get_tukey_window(
                    self.static_args['sample_length'],
                    self.static_args['target_sampling_rate'],
                )
             ) for ifo in strains
        }

    def _collate_fn(
        self,
        batch: List[Tuple[np.ndarray]],
        random_extrinsics: bool=False,
        add_noise: bool=True,
        whiten: bool=True,
        ref_psd: Optional[str]=None,
        flatten: bool=False,
        downcast: bool=False,
    ):
        """Function takes a ((batch, n_ifos, fd_length), (batch)) tuple of numpy arrays;
        representing plus and cross polarizations of a gravitational wave as well as
        intrinsic parameter labels, then project waveforms onto a set of detectors
        given a set of sampled extrinsic parameters.
        
        After projection, coloured noise is drawn from a specified PSD and added
        to the frequency domain data, which is then whitened before being returned.

        Arguments:
            batch: (nd.ndarray, nd.ndarray)
                A (strain, ground_truth) yielded from a dataset.__getitem__ method.
            random_extrinsics: bool
                Whether to generic random extrinsics live or use saved dataframe.
            add_noise: bool
                Whether to add noise according to the provided PSD.
            ref_psd: Optional[str]
                If provided, then the ref_psd corresponding to the specified
                detector is used, regardless of the origin of the strain data.
            flatten: bool
                Whether to stack real and imaginary components of the frequency
                domain strain for each into a single [B x L] real valued array.

        Returns: 
            A (batch, n, fd_length) array where n = the number of detectors;
            and an array of floats corresponding to intrinsic parameter values.
        """
        if ref_psd is not None:
            assert ref_psd in self.ifos, f"ref_psd: {ref_psd} if not an available interferometer."

        # sample extrinsic parameters on the fly and save as ground truth
        intrinsics = pd.DataFrame([labels for (_, labels) in batch])
        
        if random_extrinsics:
            extrinsics = self.extrinsics.sample(len(batch))
        else:
            assert isinstance(batch[0][1], pd.Series), (
                "Deterministic extrinsics require indexed intrinsics as pd.DataFrame/pd.Series."
            )
            extrinsics = self.extrinsics.loc[intrinsics.index]
            
        ground_truth = np.concatenate([intrinsics.values, extrinsics.values], axis=1)
        extrinsics = extrinsics.to_records(index=False)
        
        # create tensor in memory (batch x ifo x freq_bin)
        waveforms = np.stack([waveforms for (waveforms, _) in batch])
        strains = np.empty(
            (len(batch), len(self.detectors), self.static_args['fd_length']),
            dtype=self.dtype
        )
        
        # loop through each detector to write in batched tensor
        for i, ifo in enumerate(self.detectors):
            
            # get antenna beam pattern function for extrinsic parameters in batch
            fp, fc = self.detectors[ifo].antenna_pattern(
                extrinsics['ra'], extrinsics['dec'], extrinsics['psi'],
                self.static_args.get('ref_time', 0.)
            )
            
            # project intrinsic waveform (plus and cross polariizations) onto detector
            strains[:, i, :] = fp[: ,None]*waveforms[:, 0, :] + fc[:,None]*waveforms[:, 1, :]

            # scale waveform amplitude according to ratio to reference distance
            distance_scale = self.static_args.get('distance', 1.)  / extrinsics['distance']  # default d_L = 1
            strains[:, i, :] *= distance_scale[:, None]

            # Calculate time shift at detector and add to geocentric time
            dt = extrinsics['time'] - self.static_args.get('ref_time', 0.)  # default ref t_c = 0
            dt += (self.static_args['sample_length'] / 2)  # put event at centre of window
            dt += self.detectors[ifo].time_delay_from_earth_center(
                extrinsics['ra'], extrinsics['dec'], self.static_args.get('ref_time', 0.)
            )
            time_shift = np.exp(- 2j * np.pi * dt[:, None] * self.sample_frequencies[None, :])
            strains[:, i, :] *= time_shift

            # change to reference psd, if provided
            if ref_psd is not None: ifo = ref_psd
                 
            # add coloured noise from psd
            if add_noise:
                noise = frequency_noise_from_psd(self.psds[ifo], n=len(batch))
                strains[:, i, :] += noise[:, :self.static_args['fd_length']]
            
            if whiten:
                strains[:, i, :] /= (self.psds[ifo][:self.static_args['fd_length']] ** 0.5)
        
        # downcast from double to full precision and wrap with pytorch tensor
        if downcast:
            strains = strains.astype(np.complex64)
            ground_truth = ground_truth.astype(np.float32)
            
        strains = torch.from_numpy(strains)
        ground_truth = torch.from_numpy(ground_truth)

        if flatten:
            strains = (
                torch.stack([strains.real, strains.imag], dim=1)
                .reshape((strains.shape[0], 2*strains.shape[1]*strains.shape[2]))
            )

        return strains, ground_truth
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.waveforms[idx], self.intrinsics.iloc[idx])