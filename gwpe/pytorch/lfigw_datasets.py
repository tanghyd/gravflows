import os

from typing import Optional, Union, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import pycbc

# local imports# local imports
from ..utils import read_ini_config
from ..parameters import ParameterGenerator
from ..noise import get_noise_std_from_static_args, load_psd_from_file
from ..waveforms import batch_project

from .basis import BasisEncoder

class LFIGWDataset(Dataset):
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
        distance_scale: bool=True,
        time_shift: bool=False,
    ):
        # load static argument file
        self.downcast = downcast
        self.complex_dtype = np.complex64 if self.downcast else np.complex128
        self.real_dtype = np.float32 if self.downcast else np.float64
        
        self.add_noise = add_noise
        self.distance_scale = distance_scale
        self.time_shift = time_shift

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
        self.encoder = BasisEncoder(
            self.basis_dir,
            static_args_ini,
            file=None,
            ifos=ifos,
            preload=False,
            verbose=False,
        )

        # the following are loaded with worker_init_fn on each process
        self.detectors = None
        self.data = None
        self.whiten = None
        
    def _worker_init_fn(self, worker_id: int=None):
        self.detectors = {ifo: pycbc.detector.Detector(ifo) for ifo in self.ifos}

        self.data = np.load(self.data_dir / self.data_file, mmap_mode='r')

        # reduced basis encoder
        self.encoder.load(n=self.n, mmap_mode='r', verbose=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # load psd for relative whitening
        if self.psd_dir is not None:            
            psds = {}
            for ifo in self.ifos:
                psds[ifo] = load_psd_from_file(self.psd_dir / f'{ifo}_PSD.npy')
                psds[ifo] = psds[ifo][:self.static_args['fd_length']].data #** 0.5
                
            if self.ref_ifo is not None:
                self.whiten = []
                for ifo in psds:
                    if ifo == self.ref_ifo:
                        whiten = np.identity(self.n)
                    else:
                        whiten = (psds[self.ref_ifo] / psds[ifo]) ** 0.5  # relative whiten ratio
                        whiten = ((self.encoder.basis.Vh * whiten) @ self.encoder.basis.V)  # whiten RB
                    self.whiten.append(whiten.astype(self.complex_dtype))

                self.whiten = np.concatenate(self.whiten)
                self.whiten = torch.from_numpy(self.whiten)  # better to do this in collate_fn?

        self.parameters = self.parameters.astype(self.real_dtype)
        self.mean = self.mean.astype(self.real_dtype)
        self.std = self.std.astype(self.real_dtype)

    def _collate_fn(self, batch: Tuple[np.ndarray]):
        # get data
        waveforms = np.stack([item[0] for item in batch]).astype(self.complex_dtype)
        intrinsics = np.stack([item[1] for item in batch])
        extrinsics = self.extrinsics.draw(waveforms.shape[0])

        coefficients = np.empty((waveforms.shape[0], len(self.ifos), waveforms.shape[2]), dtype=waveforms.dtype)
        for i, ifo in enumerate(self.ifos):
            # batch project for each detector -  1s per call (batch_size=2000)
            coefficients[:, i, :] = batch_project(
                self.detectors[ifo],
                extrinsics,  # np.recarray
                waveforms,
                self.static_args,
                distance_scale=self.distance_scale,
                time_shift=self.time_shift,
            )
            
        # parameter standardization
        parameters = np.concatenate([intrinsics, np.array(extrinsics.tolist())], axis=1).astype(self.real_dtype)
        parameters = (parameters - self.mean) / self.std

        # send to torch tensors
        coefficients = torch.from_numpy(coefficients)  # (batch, ifo, length)
        parameters = torch.from_numpy(parameters)  # (batch, 15)

        # whiten with relative whitening in SVD basis
        coefficients = torch.einsum('bij, ijk -> bik', coefficients, self.whiten)
        
        if self.add_noise:
            # add noise to reduced basis as per: https://github.com/stephengreen/lfi-gw/blob/f4a8aceb80965eb2ad8bf59b1499b93a3c7b9194/lfigw/waveform_generator.py#L1580
            # coefficients += torch.normal(0, self.noise_std, coefficients.shape, dtype=coefficients.dtype, device=coefficients.device)
            coefficients += torch.normal(0, self.noise_std, coefficients.shape) + 1j*torch.normal(0, self.noise_std, coefficients.shape)
            coefficients *= self.encoder.standardization

        # flatten for 1-d residual network input
        coefficients = torch.cat([coefficients.real, coefficients.imag], dim=1)
        coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])

        return coefficients, parameters
        
    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), self.parameters.iloc[idx].values

class lfigwWaveformDataset(Dataset):
    def __init__(
        self,
        n: int,
        data_dir: Union[str, os.PathLike],
        basis_dir: Union[str, os.PathLike], #'lfigw/data/basis/',
        static_args_ini: str,
        data_file: str='coefficients.npy',
        intrinsics_ini: Optional[str]=None,
        extrinsics_ini: Optional[str]=None,
        psd_dir: Optional[Union[str, os.PathLike]]=None,
        ifos: List[str]=['H1','L1'],
        ref_ifo: Optional[str]=None,
        downcast: bool=True,
        add_noise: bool=False,
        distance_scale: bool=True,
        time_shift: bool=False,
        seed: Optional[int]=None,
    ):
        self.seed = seed
        self.downcast = downcast
        self.complex_dtype = np.complex64 if self.downcast else np.complex128
        self.real_dtype = np.float32 if self.downcast else np.float64
        
        self.add_noise = add_noise
        self.distance_scale = distance_scale
        self.time_shift = time_shift

        _, self.static_args = read_ini_config(static_args_ini)
        
        # configure data and settings
        self.data_file = data_file  # coefficients.npy
        self.data_dir = Path(data_dir)  # lfigw/data/train
        self.basis_dir = Path(basis_dir)  # lfigw/data/basis
        self.psd_dir = Path(psd_dir) if psd_dir is not None else None
        
        if ref_ifo is not None:
            assert ref_ifo in ifos, f"ref_ifo {ref_ifo} not in {ifos}."

        self.ref_ifo = ref_ifo
        self.ifos = ifos

        self.noise_std = get_noise_std_from_static_args(self.static_args)
        
        if intrinsics_ini is not None:
            self.intrinsics = ParameterGenerator(config_files=intrinsics_ini, seed=seed)
        else:
            self.intrinsics = None
        if extrinsics_ini is not None:
            self.extrinsics = ParameterGenerator(config_files=extrinsics_ini, seed=seed)
        else:
            self.extrinsics = None
     
        # ground truth parameters - extrinsics are generated but only polarisations used
        self.parameters = pd.read_csv(self.data_dir / 'parameters.csv', index_col=0)
        # if reorder:
        #     self.parameters.drop(columns='time', inplace=True)  # to do: remove hard-coding
        #     self.parameters = self.parameters[self.intrinsics.parameters]  # remove extrinsics
        
        # statistics estimated directly from samples
#         self.parameters['distance'] = self.extrinsics.draw(len(self)).distance
#         self.mean = self.parameters.mean(axis=0).values
#         self.std = self.parameters.std(axis=0).values
        
        # analytic computation
        self.mean, self.std = pd.concat(
            [self.intrinsics.statistics, self.extrinsics.statistics]
        ).values.T.astype(self.real_dtype)

        self.mean = torch.from_numpy(self.mean)
        self.std = torch.from_numpy(self.std)
        
        # stats = pd.read_csv(self.data_dir / 'statistics.csv', index_col=0).T
        # self.mean, self.std = stats.drop(columns='time').values

        # reduced basis encoder
        self.n = n
        self.encoder = BasisEncoder(
            data_dir=self.basis_dir,
            static_args_ini=static_args_ini,
            file=None,
            ifos=self.ifos,
            preload=False,
            verbose=False,
        )
        
        # load lfigw basis arrays
        self.encoder.basis.V = np.load(self.basis_dir / 'V.npy')
        self.encoder.basis.Vh = np.load(self.basis_dir / 'Vh.npy')
        # self.encoder.basis.standardization = np.load(self.basis_dir / 'standardization.npy')

        # restandardization of basis coefficients (for projected coeff rather than polarisation)
        data = np.load(self.data_dir / self.data_file)  # memmap's can't be passed to workers
        n_samples = 100000
        samples = self.extrinsics.draw(n_samples)
        projected_coefficients = np.empty((n_samples, len(self.ifos), self.n), dtype=data.dtype)
        for i, ifo in enumerate(self.ifos):
            projected_coefficients[:, i, :] = batch_project(
                pycbc.detector.Detector(ifo),
                samples[:n_samples],
                data[:n_samples],
                self.static_args,
                distance_scale=self.distance_scale,
                time_shift=self.time_shift,
            )

        self.encoder.basis.standardization = self.encoder.basis._fit_standardization(projected_coefficients)
        # self.encoder.load(encoder_only=True)  # in worker_init_fn or here?

        # the following are loaded with worker_init_fn on each process
        self.detectors = None
        self.data = None
        self.whiten = None
        self.psds = None
        
    def _worker_init_fn(self, worker_id: int=None, seed: Optional[int]=None):
        # dataloader reproducibility
        if seed is None:
            # if seed is an int, all workers should be identical (to do: check)
            if self.seed is not None:
                # set workers to different seeds
                seed = self.seed + worker_id
            else:
                # random seed as set when worker initialised
                seed = int(torch.initial_seed()) % (2**32-1)

        np.random.seed(seed)
        torch.random.manual_seed(seed)
        
        self.detectors = {ifo: pycbc.detector.Detector(ifo) for ifo in self.ifos}
        self.data = np.load(self.data_dir / self.data_file, mmap_mode='r')
        
        # load encoder manually
        self.encoder.load(encoder_only=True)

        if self.psd_dir is not None:
            # load psds from lfigw implementation
            self.psds = {}
            for ifo in self.ifos:
                psd = pycbc.types.load_frequencyseries(self.psd_dir / f'PSD_{ifo}.txt')
                psd = psd[:self.static_args['fd_length']].data
            
                # following Green (2020) we copy boundary values
                # rather than zero-ing them after division
                lowpass = int(self.static_args['f_lower'] / self.static_args['delta_f'])
                psd[:lowpass] = psd[lowpass]
                psd[-1:] = psd[-2]  # upper boundary is copied too

                self.psds[ifo] = psd

            if self.ref_ifo is not None:
                self.whiten = []
                for ifo in self.psds:
                    # if ifo == self.ref_ifo:
                    #     whiten = np.identity(self.n)[None]
                    # else:
                    whiten = (self.psds[self.ref_ifo] / self.psds[ifo]) ** 0.5  # relative whiten ratio
                    whiten = ((self.encoder.basis.Vh * whiten) @ self.encoder.basis.V)  # whiten RB
                    self.whiten.append(whiten.astype(self.complex_dtype))

                self.whiten = torch.from_numpy(np.concatenate(self.whiten))


    def _collate_fn(self, batch: Tuple[np.ndarray], dynamic: bool=True):
        # get data
        waveforms = np.stack([item[0] for item in batch]).astype(self.complex_dtype)
        parameters = np.stack([item[1] for item in batch])

        if dynamic:
            # randomly draw extrinsics
            extrinsics = self.extrinsics.draw(waveforms.shape[0])
            parameters = np.concatenate([parameters, np.array(extrinsics.tolist())], axis=1)
        else:
            # extrinsics are present in parameters - get recarray for batch_project
            extrinsic_cols = [self.parameters.columns.tolist().index(param) for param in self.extrinsics.parameters]
            extrinsics = pd.DataFrame(parameters[:, extrinsic_cols], columns=self.extrinsics.parameters)  # easy to convert to rec.array
            extrinsics = extrinsics.to_records(index=False)  # batch_project typically takes np.recarray

        parameters = torch.from_numpy(parameters.astype(self.real_dtype))  # (batch, 15)

        coefficients = np.empty((waveforms.shape[0], len(self.ifos), waveforms.shape[2]), dtype=waveforms.dtype)
        for i, ifo in enumerate(self.ifos):
            # batch project for each detector -  1s per call (batch_size=2000)
            coefficients[:, i, :] = batch_project(
                self.detectors[ifo],
                extrinsics,  # np.recarray
                waveforms,
                self.static_args,
                distance_scale=self.distance_scale,
                time_shift=self.time_shift,
            )

        coefficients = torch.from_numpy(coefficients)  # (batch, ifo, length)

        # whiten with relative whitening in SVD basis
        coefficients = torch.einsum('bij, ijk -> bik', coefficients, self.whiten)
        
        if self.add_noise:
            # add gaussian noise to whitened data
            coefficients += (
                torch.normal(0, self.noise_std, coefficients.shape)
                + 1j*torch.normal(0, self.noise_std, coefficients.shape)
            )

        # standardize data
        coefficients *= self.encoder.standardization
        parameters = ((parameters - self.mean) / self.std)

        # flatten for 1-d residual network input
        coefficients = torch.cat([coefficients.real, coefficients.imag], dim=2)
        coefficients = coefficients.reshape(coefficients.shape[0], coefficients.shape[1]*coefficients.shape[2])

        return coefficients, parameters
    
    def __len__(self):
        return len(self.parameters)
    
    def __getitem__(self, idx):
        return np.array(self.data[idx]), self.parameters.iloc[idx].values
