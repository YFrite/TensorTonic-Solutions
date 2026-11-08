import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """

    x_arr = np.array(x)
    n = x_arr.shape[0]
    x_mean = x_arr.mean()
    variance = float(1 / (n - 1) * ((x_arr - x_mean) ** 2).sum())
    std = variance ** 0.5
    
    return {"standard_deviation": std, "variance": variance}