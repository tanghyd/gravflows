import numpy as np

def match_precision(data: np.ndarray, real: bool=True):
    """Convenience function returns matching types.
    
    Works for np.ndarrays rather than only pycbc types.

    Arguments:
        real: bool
            If true, returns np.float; else np.complex.
    """
    if data.dtype in (np.float32, np.complex64):
        if real:
            return np.float32
        else:
            return np.complex64
    elif data.dtype in (np.float64, np.complex128):
        if real: 
            return np.float64
        else:
            return np.complex128
    else:
        raise TypeError("Input data array is neither single/double precision or real/complex.")
