import numpy as np
from math import factorial

primal_simplex_div = np.vectorize(
    # Special division option used in primal algorithm.
    lambda n, d: n / d
    if d > 0
    else np.inf
)

dual_simplex_div = np.vectorize(
    # Special division option used in dual algorithm.
    lambda n, d: -1 * n / d
    if d < 0
    else np.inf
)

def get_bounds_on_bfs(A: np.array, b: np.array) -> float:
    """Get bounds on magnitude of x_i in all bfs.
    ref: lemma 2.1 combinatorial optimization - algorithms and complexity
    
    Parameters
    ----------
    A : np.array
        (m, n) matrix defining linear combinations subject to equality constraints.
    b : np.array
        (m, 1) vector defining the equality constraints.

    Returns
    -------
    float
        Scalar that bounds absolute value of each element of any bfs.
    """
    m = A.shape[0]
    alpha = np.abs(A).max()
    beta = np.abs(b).max()
    M = factorial(m) * alpha ** (m-1) * beta
    return M