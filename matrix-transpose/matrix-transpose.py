import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    m, n = len(A[0]), len(A)
    A_T = np.zeros((m, n))

    for i in range(n):
        for j in range(m):
            A_T[j][i] = A[i][j]

    return A_T 

    
    # Lol: return np.array(A).T
    # 2: return np.array([[row[i] for row in A] for i in range(len(A[0]))])
    # 3: return np.array([list(row) for row in zip(*A)])