# Master of Data Science Research Project

This repository is for the research work of Daniel Tang (21140852) for work on flow-based models for gravitational wave parameter estimation.

We have forked the code written by Stephen R. Green and Jonathan Gair [arXiv:2008.03312](https://arxiv.org/abs/2008.03312) to leverage their previous work on gravitational wave generation and training a neural spline flow model.
We have also forked code from samplegen by Gabbard et al., subsequently edited by Damon Beveridge.

## Current Goals

We have been investigating how to refactor the waveform generation code for potential run-time speedups using multi-threading (for I/O bound fetching open source data) and multi-processing (for CPU bound simluating of waveform models given samples from our priors). Additionally, we intend to rewrite a more comprehensive and reproducible training framework with PyTorch (or possibly PyTorch Lightning) to help accelerate our research.

We also hope to include research from Noise Flows and Fourier Flows - however we may need to rewrite these models from scratch in PyTorch.
- Noise Flows: https://arxiv.org/abs/1908.08453
- Fourier Flows: https://openreview.net/forum?id=PpshD0AXfA

We propose that noise modelling may be appropriate to help with denoising strain data, and fourier flows may be more suited to the data model - especially as the work done by Green and Gair work with waveform signals in the frequency domain (although it is projected to a reduced basis fit via randomized SVD).