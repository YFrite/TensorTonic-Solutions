import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    return sum([float(x_i * y_i) for x_i, y_i in zip(x, y)])