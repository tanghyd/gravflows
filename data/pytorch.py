import os

from typing import Optional, Union, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from pycbc.detector import Detector

# local imports# local imports
from .config import read_ini_config
from .parameters import ParameterGenerator
from .noise import load_psd_from_file
from .waveforms import batch_project, get_sample_frequencies

class WaveformDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        static_args_ini: str,
        intrinsics_ini: Optional[str]=None,
        extrinsics_ini: Optional[str]=None,
        psd_dir: Optional[Union[str, os.PathLike]]=None,
        ifos: List[str]=['H1','L1'],
        downcast: bool=False,
    ):
        # load static argument file
        self.downcast = downcast
        _, self.static_args = read_ini_config(static_args_ini)

        # configure data and settings
        self.data_dir = Path(data_dir)
        self.psd_dir = Path(psd_dir) if psd_dir is not None else self.data_dir / 'PSD'
        
        self.ifos = ifos
#         self.psds = {ifo: load_psd_from_file(self.psd_dir / f'{ifo}_PSD.npy') for ifo in self.ifos}

        # ground truth parameters
        self.parameters = pd.read_csv(self.data_dir / 'parameters.csv', index_col=0)
        
        # loaded on each worker
        self.intrinsics = ParameterGenerator(config_files=intrinsics_ini, seed=None)
        self.extrinsics = ParameterGenerator(config_files=extrinsics_ini, seed=None)
        
        # the following are loaded with worker_init_fn on each process
        self.basis = None
        self.detectors = None
        self.data = None
        
    def _worker_init_fn(self, worker_id: int=None):
        self.detectors = {ifo: Detector(ifo) for ifo in self.ifos}
        self.data = np.load(self.data_dir / 'waveforms.npy', mmap_mode='r')
        self.sample_frequencies = get_sample_frequencies(
            f_final=self.static_args['f_final'],
            delta_f=self.static_args['delta_f']
        )
        
        # save asds as stacked numpy array for faster compute
        asds = []
        for ifo in self.ifos:
            psd = load_psd_from_file(self.psd_dir / f'{ifo}_PSD.npy')
            asds.append(psd[:self.static_args['fd_length']] ** 0.5)
        self.asds = np.stack(asds)
        
    def _collate_fn(self, batch: Tuple[np.ndarray]):
        # get data
        waveforms = np.stack([item[0] for item in batch])
        intrinsics = np.stack([item[1] for item in batch])
        extrinsics = self.extrinsics.draw(waveforms.shape[0])  # generate in real time

        # indices to set to zero for bandpass (and avoid noise generation)
        lowpass = int(self.static_args['f_lower'] / self.static_args['delta_f'])
        
        projections = np.empty((waveforms.shape[0], len(self.ifos), self.static_args['fd_length']), dtype=waveforms.dtype)
        for i, ifo in enumerate(self.ifos):
            # batch project for each detector -  1s per call (batch_size=2000)
            projections[:, i, :] = batch_project(
                self.detectors[ifo],
                extrinsics,  # np.recarray
                waveforms,
                self.static_args,
                self.sample_frequencies
            )
            
        # filter out values less than f_lower (e.g. 20Hz) - to do: check truncation vs. zeroing
        projections[:, : :lowpass] = 0.0
            
        # whiten - PSD division best done with double precision - 0.07s per call (batch_size=2000)
        projections /= self.asds[None]  # broadcasted over batch

        # parameter standardization
        parameters = np.concatenate([intrinsics, np.array(extrinsics.tolist())], axis=1)
        mean, std = np.concatenate([self.intrinsics.statistics.values, self.extrinsics.statistics.values]).T
        parameters = (parameters - mean) / std
        
        # build and typecast data - 0.07s per call (0.14s total) (batch_size=2000)
        if self.downcast:
            projections = projections.astype(np.complex64)
            parameters = parameters.astype(np.float32)
        
        # send to torch tensors
        projections = torch.from_numpy(projections)  # (batch, ifo, length)
        parameters = torch.from_numpy(parameters)  # (batch, 15)
        return projections, parameters
        
    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), self.parameters.iloc[idx].values