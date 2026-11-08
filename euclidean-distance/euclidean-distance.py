import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    return sum([(x_i - y_i)**2 for x_i, y_i in zip(x, y)]) ** 0.5