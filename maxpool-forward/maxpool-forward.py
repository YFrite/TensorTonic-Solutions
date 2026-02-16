import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.array(X)
    
    view = np.lib.stride_tricks.sliding_window_view(X, (pool_size, pool_size))
    
    view = view[::stride, ::stride]
    
    return view.max(axis=(2, 3)).tolist()