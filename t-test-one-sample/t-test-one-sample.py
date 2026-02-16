import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x)
    n = x.shape[0]
    mean = np.mean(x)

    std = np.sqrt(1 / (n-1) * ((x - mean)**2).sum())
    
    return (mean - mu0) / (std / np.sqrt(n))
