import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    m_arr = np.array(m)
    v_arr = np.array(v)
    g_arr = np.array(grad)
    w_arr = np.array(w)

    m_new = beta1 * m_arr + (1-beta1) * g_arr
    v_new = beta2 * v_arr + (1-beta2) * g_arr ** 2

    w_new = w_arr - lr * (m_new / (v_new ** 0.5 + eps)) - lr * weight_decay * w_arr

    return {"new_w": w_new, "new_m": m_new, "new_v": v_new}