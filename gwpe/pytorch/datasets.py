import os

from typing import Optional, Union, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from pycbc.detector import Detector

# local imports# local imports
from ..utils import read_ini_config
from ..parameters import ParameterGenerator
from ..noise import get_noise_std_from_static_args, load_psd_from_file
from ..waveforms import batch_project

from .basis import BasisEncoder


class BasisCoefficientsDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        basis_dir: Union[str, os.PathLike],
        static_args_ini: str,
        parameters_ini: str,
        data_file: str='coefficients.npy',
        coefficient_noise: bool=False,
    ):
        # load static argument file
        _, self.static_args = read_ini_config(static_args_ini)

        # configure data and settings
        self.data_file = data_file
        self.data_dir = Path(data_dir)

        # load basis coefficient standardization scaler
        self.basis_dir = Path(basis_dir)
        self.standardization = torch.from_numpy(np.load(basis_dir / 'reduced_basis'/ 'standardization.npy'))
        
        self.noise_std = get_noise_std_from_static_args(self.static_args)
        self.coefficient_noise = coefficient_noise

        # ground truth parameters
        self.parameters = pd.read_csv(self.data_dir / 'parameters.csv', index_col=0).astype(np.float32)
        self.generator = ParameterGenerator(config_files=parameters_ini, seed=None)
        self.mean, self.std = self.generator.statistics.values.astype(np.float32).T

        # the following are loaded with worker_init_fn on each process
        self.data = None
        
    def _worker_init_fn(self, worker_id: int=None):
        self.data = np.load(self.data_dir / self.data_file, mmap_mode='r')

    def _collate_fn(self, batch: Tuple[np.ndarray]):
        # get data
        coefficients = np.stack([item[0] for item in batch])
        parameters = np.stack([item[1] for item in batch])

        # parameter standardization
        parameters = (parameters - self.mean) / self.std
        
        coefficients = torch.from_numpy(coefficients)  # (batch, ifo, length)
        parameters = torch.from_numpy(parameters)  # (batch, 15)

        if not self.coefficient_noise:
            # add noise to reduced basis as per: https://github.com/stephengreen/lfi-gw/blob/f4a8aceb80965eb2ad8bf59b1499b93a3c7b9194/lfigw/waveform_generator.py#L1580
            coefficients += torch.normal(0, self.noise_std, coefficients.shape, dtype=coefficients.dtype, device=coefficients.device)

        # coefficient standardization truncated to match input data 
        coefficients *= self.standardization[:, :coefficients.shape[2]]

        # flatten for 1-d residual network input
        coefficients = torch.cat([coefficients.real, coefficients.imag], dim=1)
        coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])


        return coefficients, parameters

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        
        return np.array(self.data[idx]), self.parameters.iloc[idx].values


class BasisEncoderDataset(Dataset):
    def __init__(
        self,
        n: int,
        data_dir: Union[str, os.PathLike],
        basis_dir: Union[str, os.PathLike],
        static_args_ini: str,
        data_file: str='waveforms.npy',
        intrinsics_ini: Optional[str]=None,
        extrinsics_ini: Optional[str]=None,
        psd_dir: Optional[Union[str, os.PathLike]]=None,
        ifos: List[str]=['H1','L1'],
        ref_ifo: Optional[str]=None,
        downcast: bool=False,
        add_noise: bool=False,
        coefficient_noise: bool=False,
        # time_shift: bool=False,
    ):
        # load static argument file
        self.downcast = downcast
        self.complex_dtype = np.complex64 if self.downcast else np.complex128
        self.real_dtype = np.float32 if self.downcast else np.float64
        
        self.add_noise = add_noise
        # self.time_shift = time_shift

        _, self.static_args = read_ini_config(static_args_ini)

        # configure data and settings
        self.data_file = data_file
        self.data_dir = Path(data_dir)
        self.basis_dir = Path(basis_dir)
        self.psd_dir = Path(psd_dir) if psd_dir is not None else None
        
        if ref_ifo is not None:
            assert ref_ifo in ifos, f"ref_ifo {ref_ifo} not in {ifos}."

        self.ref_ifo = ref_ifo
        self.ifos = ifos
        
        self.noise_std = get_noise_std_from_static_args(self.static_args)
        self.coefficient_noise = coefficient_noise

        # ground truth parameters
        self.parameters = pd.read_csv(self.data_dir / 'parameters.csv', index_col=0)
        
        # loaded on each worker
        if intrinsics_ini is not None:
            self.intrinsics = ParameterGenerator(config_files=intrinsics_ini, seed=None)
        else:
            self.intrinsics = None
        if extrinsics_ini is not None:
            self.extrinsics = ParameterGenerator(config_files=extrinsics_ini, seed=None)
        else:
            self.extrinsics = None

        self.mean, self.std = np.concatenate([self.intrinsics.statistics.values, self.extrinsics.statistics.values]).T

        # reduced basis
        self.n = n  # number of reduced basis elements
        self.encoder = BasisEncoder(self.basis_dir, static_args_ini, preload=False, verbose=False)

        # the following are loaded with worker_init_fn on each process
        self.basis = None
        self.detectors = None
        self.data = None
        self.asds = None
        
    def _worker_init_fn(self, worker_id: int=None):
        self.detectors = {ifo: Detector(ifo) for ifo in self.ifos}

        self.data = np.load(self.data_dir / self.data_file, mmap_mode='r')

        # save asds as stacked numpy array for faster compute
        if self.psd_dir is not None:
            asds = []
            for ifo in self.ifos:
                psd = load_psd_from_file(self.psd_dir / f'{ifo}_PSD.npy')
                asds.append(psd[:self.static_args['fd_length']] ** 0.5)
            self.asds = np.stack(asds)

        if self.ref_ifo is not None:
            # psd = load_psd_from_file(self.psd_dir / f'{self.ref_ifo}_PSD.npy')
            self.asds /= self.asds[self.ifos.index(self.ref_ifo)]
        
        self.asds = self.asds.astype(self.real_dtype)
        self.parameters = self.parameters.astype(self.real_dtype)
        self.mean = self.mean.astype(self.real_dtype)
        self.std = self.std.astype(self.real_dtype)

        # reduced basis encoder - to do: put this in DataSet
        self.encoder.load(n=self.n, mmap_mode='r', verbose=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _collate_fn(self, batch: Tuple[np.ndarray]):
        # get data
        waveforms = np.stack([item[0] for item in batch]).astype(self.complex_dtype)
        intrinsics = np.stack([item[1] for item in batch])
        extrinsics = self.extrinsics.draw(waveforms.shape[0])
        
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
                time_shift=False,
            )
            
        # whiten
        projections /= self.asds[None]

        # parameter standardization
        parameters = np.concatenate([intrinsics, np.array(extrinsics.tolist())], axis=1).astype(self.real_dtype)
        parameters = (parameters - self.mean) / self.std

        # send to torch tensors
        projections = torch.from_numpy(projections)  # (batch, ifo, length)
        parameters = torch.from_numpy(parameters)  # (batch, 15)

        # generate noise for whitened waveform above lowpass filter (i.e. >20Hz)
        if self.add_noise and not self.coefficient_noise:
            size = (projections.shape[0], projections.shape[1], projections.shape[2] - lowpass)
            projections[:, :, lowpass:] += torch.normal(0, self.noise_std, size, dtype=projections.dtype, device=projections.device)

        # filter out values less than f_lower (e.g. 20Hz) - to do: check truncation vs. zeroing
        projections[:, : :lowpass] = 0.0  # may already be done in data generation.

        # project to reduced basis and 
        coefficients = self.encoder(projections, scale=not self.coefficient_noise)

        if self.add_noise and self.coefficient_noise:
            # add noise to reduced basis as per: https://github.com/stephengreen/lfi-gw/blob/f4a8aceb80965eb2ad8bf59b1499b93a3c7b9194/lfigw/waveform_generator.py#L1580
            coefficients += torch.normal(0, self.noise_std, coefficients.shape, dtype=coefficients.dtype, device=coefficients.device)
            coefficients *= self.encoder.standardization

        # flatten for 1-d residual network input
        coefficients = torch.cat([coefficients.real, coefficients.imag], dim=1)
        coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])

        return coefficients, parameters
        
    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), self.parameters.iloc[idx].values


class WaveformDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        static_args_ini: str,
        data_file: str='waveforms.npy',
        intrinsics_ini: Optional[str]=None,
        extrinsics_ini: Optional[str]=None,
        psd_dir: Optional[Union[str, os.PathLike]]=None,
        ifos: List[str]=['H1','L1'],
        ref_ifo: Optional[str]=None,
        downcast: bool=False,
        add_noise: bool=False,
        time_shift: bool=False,
    ):
        # load static argument file
        self.downcast = downcast
        self.add_noise = add_noise
        self.time_shift = time_shift

        _, self.static_args = read_ini_config(static_args_ini)

        # configure data and settings
        self.data_file = data_file
        self.data_dir = Path(data_dir)
        self.psd_dir = Path(psd_dir) if psd_dir is not None else None
        
        if ref_ifo is not None:
            assert ref_ifo in ifos, f"ref_ifo {ref_ifo} not in {ifos}."

        self.ref_ifo = ref_ifo
        self.ifos = ifos

        
        self.complex_dtype = np.complex64 if self.downcast else np.complex128
        self.real_dtype = np.float32 if self.downcast else np.float64
        
        # ground truth parameters
        self.parameters = pd.read_csv(self.data_dir / 'parameters.csv', index_col=0)
        
        # loaded on each worker
        if intrinsics_ini is not None:
            self.intrinsics = ParameterGenerator(config_files=intrinsics_ini, seed=None)
        else:
            self.intrinsics = None
        if extrinsics_ini is not None:
            self.extrinsics = ParameterGenerator(config_files=extrinsics_ini, seed=None)
        else:
            self.extrinsics = None
        
        # parameter statistics for standardization
        self.mean, self.std = np.concatenate(
            [self.intrinsics.statistics.values, self.extrinsics.statistics.values]
        ).T.astype(self.real_dtype)

        # reduced basis
        self.lowpass = int(self.static_args['f_lower'] / self.static_args['delta_f'])

        # the following are loaded with worker_init_fn on each process
        self.basis = None
        self.detectors = None
        self.data = None
        self.asds = None
        
    def _worker_init_fn(self, worker_id: int=None):
        self.detectors = {ifo: Detector(ifo) for ifo in self.ifos}

        self.data = np.load(self.data_dir / self.data_file, mmap_mode='c')

        # save asds as stacked numpy array for faster compute
        if self.psd_dir is not None:
            asds = []
            for ifo in self.ifos:
                psd = load_psd_from_file(self.psd_dir / f'{ifo}_PSD.npy')
                asds.append(psd[:self.static_args['fd_length']] ** 0.5)
                
            self.asds = np.stack(asds).astype(self.complex_dtype)

    def _collate_fn(self, batch: Tuple[np.ndarray]):
        # get data
        waveforms = np.stack([item[0] for item in batch]).astype(self.complex_dtype)
        intrinsics = np.stack([item[1] for item in batch])
        extrinsics = self.extrinsics.draw(waveforms.shape[0])  # generate in real time

        projections = np.empty((waveforms.shape[0], len(self.ifos), self.static_args['fd_length']), dtype=waveforms.dtype)
        
        for i, ifo in enumerate(self.ifos):
            # batch project for each detector -  1s per call (batch_size=2000)
            projections[:, i, :] = batch_project(
                self.detectors[ifo],
                extrinsics,  # np.recarray
                waveforms,
                self.static_args,
                time_shift=self.time_shift,
            )
            
        # filter out values less than f_lower (e.g. 20Hz) - to do: check truncation vs. zeroing
        projections[:, : : self.lowpass] = 0.0  # may already be done in data generation.
            
        # whiten - warning: division by PSD not perfectly reliable unless we are in double precision
        # 0.07s per call (batch_size=2000)
        if self.ref_ifo is not None:
            # broadcasted over batch
            projections /= (self.asds / self.asds[self.ifos.index(self.ref_ifo)])[None]
        else:
            projections /= self.asds[None]

        # parameter standardization
        parameters = np.concatenate([intrinsics, np.array(extrinsics.tolist())], axis=1).astype(self.real_dtype)
        parameters = (parameters - self.mean) / self.std  # check if this upcasts

        # send to torch tensors
        projections = torch.from_numpy(projections)  # (batch, ifo, length)
        parameters = torch.from_numpy(parameters)  # (batch, 15)

        return projections, parameters
        
    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), self.parameters.iloc[idx].values