import numpy as np

def make_diagonal(v: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, N).
    """
    v_arr = np.array(v)
    N = v_arr.shape[0]
    
    A = np.zeros(shape=(N, N))
    A.flat[::N + 1] = v_arr

    return A