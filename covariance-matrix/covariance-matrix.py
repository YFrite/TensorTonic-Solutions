import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X)
    if X.shape[0] < 2 or len(X.shape) != 2:
        return None
    
    X_c = X - np.mean(X, axis=0)

    return 1 / (X.shape[0] - 1) * (X_c.T@X_c)