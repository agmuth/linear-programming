import numpy as np 

primal_simplex_div = np.vectorize(
    lambda n, d: n / d if d > 0 else np.inf
)

dual_simplex_div = np.vectorize(
    lambda n, d: -1 * n / d if d < 0 else np.inf
)

