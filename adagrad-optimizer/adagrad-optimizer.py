import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    w_arr = np.array(w)
    G_arr = np.array(G)
    g_arr = np.array(g)

    G_new = G_arr + g_arr ** 2
    w_new = w_arr - lr * (g_arr / (G_new + eps)**0.5)

    return {"new_w": w_new, "new_G": G_new}