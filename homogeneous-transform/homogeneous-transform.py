import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.asarray(points)
    T = np.asarray(T)
    points = np.atleast_2d(points)
    points_h = np.c_[points, np.ones(points.shape[:-1] + (1,))]
    
    res_h = points_h @ T.T
    
    return res_h[..., :3].flatten() if res_h.shape[0] == 1 else res_h[..., :3]