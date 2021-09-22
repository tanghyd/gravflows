# Master of Data Science Research Project

This repository is for the research work of Daniel Tang (21140852) for work on flow-based models for gravitational wave parameter estimation.

We have forked code written by Stephen R. Green and Jonathan Gair [arXiv:2008.03312](https://arxiv.org/abs/2008.03312) to leverage their previous work on training a neural spline flow model for gravitational wave parameter estimation.
We have also forked code from samplegen by Gabbard et al. (subsequently edited by Damon Beveridge) to utilise pre-existing code handling bit-masks for valid gravitational wave strain data and PyCBC config file handling.
## Current Goals

We have been investigating how to refactor the waveform generation code for potential run-time speedups using multi-threading (for I/O bound fetching open source data) and multi-processing (for CPU bound simluating of waveform models given samples from our priors). Additionally, we intend to rewrite a more comprehensive and reproducible training framework with PyTorch (or possibly PyTorch Lightning) to help accelerate our research.
