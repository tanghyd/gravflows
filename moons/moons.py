from typing import Union, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np

import sklearn.datasets as datasets

import torch
from torch.utils.data import Dataset

# we import ligo.skymap exclusively for cylon colourmap
# if this is not a desired colourmap, delete the import
import ligo.skymap.plot

# PyTorch DataSet
class TwoMoonsDataset(Dataset):
    def __init__(self, n, noise=0.1):
        self.n = n
        x, y = datasets.make_moons(n, noise=noise)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


def map_colours(
    scalars: Union[np.ndarray, torch.Tensor],
    cmap: str='viridis',
    normalize: bool=True,
    bytes: bool=True,
    alpha: Optional[float]=None,
):
    """Returns an RGB array given an input array of scalar values and a valid matplotlib colourmap.
    
    Arguments:
        scalars - input array of scalars. Can be either np.ndarray or torch.Tensor.
        cmap: str='default' - the name of the matplotlib colourmap to get, i.e. matplotlib.pyplot.get_cmap(cmap).
        normalize: bool=True - whether or not to normalize the input values to the range [0, 1].
        preserve: bool=True - if True, the output array will match the input array (i.e. torch vs. numpy)
        bytes: bool=True - if True, returns uint8 RGB values from 0 to 255, else values are in [0, 1].
        alpha: Optional[float]=1. - sets the degree of transparency in the output image.
        
    Returns:
        A numpy array of RGBA values.
    """
    z = scalars.detach().cpu().numpy() if isinstance(scalars, torch.Tensor) else scalars
    assert isinstance(z, np.ndarray)
    
    colourmap = plt.get_cmap(cmap)
    normalize = Normalize()
    return colourmap(normalize(z), alpha=alpha, bytes=bytes)


def cascade_log_probs(flow, inputs, context):
    "Returns a stacked tensor of log-likelihoods evaluated at intermediate transforms within a flow given a batched input."
    batch_size = inputs.shape[0]
    outputs = inputs
    total_logabsdet = inputs.new_zeros(batch_size)
    log_probs = []
    for transform in flow._transform._transforms:
        outputs, logabsdet = transform(outputs, context)
        total_logabsdet += logabsdet
        log_prob = flow._distribution.log_prob(outputs, context=flow._embedding_net(context))
        log_probs.append(log_prob + total_logabsdet)
    return torch.stack(log_probs)