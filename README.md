# Master of Data Science Research Project

This repository is for the research work of Daniel Tang (21140852) for work on flow-based models for gravitational wave parameter estimation.

We have forked code written by Stephen R. Green and Jonathan Gair [arXiv:2008.03312](https://arxiv.org/abs/2008.03312) to leverage their previous work on training a neural spline flow model for gravitational wave parameter estimation.

We have also forked code from https://github.com/damonbeveridge/samplegen - a repostiory that provides additional features to the sample generation code by Timothy Gebhard and Niki Kilbertus for generating realistic synthetic gravitational-wave data. Our main use of this code is to improve pre-existing code handling .hdf5 data files and bit-masks for valid gravitational wave strains as well as PyCBC config file handling.
## Current Goals

We have been investigating how to refactor the waveform generation code for potential run-time speedups using multi-threading (for I/O bound fetching open source data) and multi-processing (for CPU bound simluating of waveform models given samples from our priors).

Additionally, we intend to rewrite a reproducible distributed training framework with PyTorch and DataDistributedParallel help accelerate our research.
