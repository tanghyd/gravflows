# Code originally forked from Green and Gair (2020)
# https://github.com/stephengreen/lfi-gw/blob/master/lfigw/nde_flows.py
 
from typing import Union, Optional
import numpy as np
import torch
from torch.nn import functional as F

from nflows import distributions, flows, transforms, utils
from nflows.nn import nets

def create_linear_transform(param_dim):
    """Create the composite linear transform PLU.

    Arguments:
        input_dim {int} -- dimension of the space

    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True)
    ])


def create_base_transform(
    i,
    param_dim,
    context_dim=None,
    hidden_dim=512,
    num_transform_blocks=5,
    activation='relu',
    dropout_probability=0.0,
    batch_norm=False,
    num_bins=8,
    tail_bound=1.,
    apply_unconditional_transform=False,
    base_transform_type='rq-coupling'
):
    """Build a base NSF transform of x, conditioned on y.

    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
    in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).

    Code is adapted from the uci.py example from
    https://github.com/bayesiains/nsf.

    A coupling flow fixes half the components of x, and applies a transform
    to the remaining components, conditioned on the fixed components. This is
    a restricted form of an autoregressive transform, with a single split into
    fixed/transformed components.

    The transform here is a neural spline flow, where the flow is parametrized
    by a residual neural network that depends on x_fixed and y. The residual
    network consists of a sequence of two-layer fully-connected blocks.

    Arguments:
        i {int} -- index of transform in sequence
        param_dim {int} -- dimensionality of x

    Keyword Arguments:
        context_dim {int} -- dimensionality of y (default: {None})
        hidden_dim {int} -- number of hidden units per layer (default: {512})
        num_transform_blocks {int} -- number of transform blocks comprising the
                                      transform (default: {2})
        activation {str} -- activation function (default: {'relu'})
        dropout_probability {float} -- probability of dropping out a unit
                                       (default: {0.0})
        batch_norm {bool} -- whether to use batch normalization
                             (default: {False})
        num_bins {int} -- number of bins for the spline (default: {8})
        tail_bound {[type]} -- [description] (default: {1.})
        apply_unconditional_transform {bool} -- whether to apply an
                                                unconditional transform to
                                                fixed components
                                                (default: {False})

        base_transform_type {str} -- type of base transform
                                     ([rq-coupling], rq-autoregressive)

    Returns:
        Transform -- the NSF transform
    """

    if activation == 'elu':
        activation_fn = F.elu
    elif activation == 'relu':
        activation_fn = F.relu
    elif activation == 'leaky_relu':
        activation_fn = F.leaky_relu
    else:
        activation_fn = F.relu   # Default
        print('Invalid activation function specified. Using ReLU.')

    if base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(
                lambda in_features, out_features:
                    nets.ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        hidden_features=hidden_dim,
                        context_features=context_dim,
                        num_blocks=num_transform_blocks,
                        activation=activation_fn,
                        dropout_probability=dropout_probability,
                        use_batch_norm=batch_norm
                    )
                ),
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform
        )

    elif base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=param_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=activation_fn,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm
        )

    else:
        raise ValueError

def create_transform(
    num_flow_steps: int,
    param_dim: int,
    context_dim: int,
    base_transform_kwargs: dict,
):
    """Build a sequence of NSF transforms, which maps parameters x into the
    base distribution u (noise). Transforms are conditioned on strain data y.

    Note that the forward map is f^{-1}(x, y).

    Each step in the sequence consists of
        * A linear transform of x, which in particular permutes components
        * A NSF transform of x, conditioned on y.
        
    There is one final linear transform at the end.

    This function was adapted from the uci.py example in
    https://github.com/bayesiains/nsf

    Arguments:
        num_flow_steps {int} -- number of transforms in sequence
        param_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        base_transform_kwargs {dict} -- hyperparameters for NSF step

    Returns:
        Transform -- the constructed transform


    """
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(param_dim),
            create_base_transform(
                i,
                param_dim,
                context_dim=context_dim,
                **base_transform_kwargs
            )
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform(param_dim)
    ])

    # This architecture has been re-compartmentalized to have an initial linear
    # transform, followed by pairs of (NSF, linear) transforms. The architecture
    # should be exactly the same as in lfigw/nde_flows.py but intermediate layers
    # have been grouped differently for ease of visualising intermediate predictions.

    # transform = transforms.CompositeTransform([
    #     transforms.CompositeTransform([
    #         create_linear_transform(param_dim),
    #         create_base_transform(
    #             i,
    #             param_dim,
    #             context_dim=context_dim,
    #             **base_transform_kwargs
    #         )
    #     ]) for i in range(num_flow_steps-1)
    # ] + [transforms.CompositeTransform([
    #         create_linear_transform(param_dim),
    #         create_base_transform(
    #             num_flow_steps-1,
    #             param_dim,
    #             context_dim=context_dim,
    #             **base_transform_kwargs
    #         ),
    #         create_linear_transform(param_dim)
    #     ])]
    # )

    return transform


def create_NDE_model(input_dim, context_dim, num_flow_steps,
                     base_transform_kwargs):
    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.

    This models the posterior distribution p(x|y).

    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y

    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps

    Returns:
        Flow -- the model
    """
    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(num_flow_steps, input_dim, context_dim, base_transform_kwargs)
    flow = flows.Flow(transform, distribution)

    # Store hyperparameters - useful for loading from file.
    flow.model_hyperparams = {
        'input_dim': input_dim,
        'num_flow_steps': num_flow_steps,
        'context_dim': context_dim,
        'base_transform_kwargs': base_transform_kwargs
    }

    return flow

def sample_flow(
    flow,
    n: int=50000,
    context: Optional[Union[np.ndarray, torch.Tensor]]=None,
    batch_size: int=512,
    output_device: Union[str, torch.device]='cpu',
    dtype=torch.float64,
):
    """Draw samples from the posterior.
    
    The nsf package concatenates on the wrong dimension (dim=0 instead of dim=1).
        
    Arguments:
        flow {Flow} -- NSF model
        y {array} -- strain data
        nsamples {int} -- number of samples desired

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        batch_size {int} -- batch size for sampling (default: {512})

    Returns:
        Tensor -- samples
    """
    if not flow.training: print("WARNING: Flows not in eval mode may generate incorrect samples.")
    with torch.inference_mode():
        if context is not None:
            if not isinstance(context, torch.Tensor):
                context = torch.from_numpy(context)
            if len(context.shape) == 1:
                # if 1 context tensor provided, unsqueeze batch dim
                context = context.unsqueeze(0)

        num_batches = n // batch_size
        num_leftover = n % batch_size

        samples = [flow.sample(batch_size, context).to(output_device, dtype) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(flow.sample(num_leftover, context).to(output_device, dtype))


        samples = torch.cat(samples, dim=1)

        return samples