import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w_arr = np.array(w, dtype=np.float64)
    g_arr = np.array(g, dtype=np.float64)
    s_arr = np.array(s, dtype=np.float64)

    s_new = beta * s_arr + (1.0 - beta) * (g_arr ** 2)
    w_new = w_arr - (lr / np.sqrt(s_new + eps)) * g_arr

    return w_new.tolist(), s_new.tolist()
    