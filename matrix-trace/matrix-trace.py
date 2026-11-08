import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    return sum([A[i][i] for i in range(len(A))])