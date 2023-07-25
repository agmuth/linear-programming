from dataclasses import dataclass
from math import factorial

import numpy as np

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
    M = factorial(m) * alpha ** (m - 1) * beta
    return M


@dataclass
class LinProgResult:
    x: np.array
    basis: np.array
    cost: float
    iters: int
    optimum: bool


class ProblemPreprocessingUtils:
    @staticmethod
    def preprocess_problem(c, A, b):
        c = np.array(c).astype(np.float32)
        A = np.array(A).astype(np.float32)
        b = np.array(b).astype(np.float32)

        b_is_neg_index = b < 0
        A[b_is_neg_index] *= -1
        b[b_is_neg_index] *= -1

        return c, A, b

    @staticmethod
    def canonical_form_to_standard_form(c, G, h):
        # assumes problem is of form min cx sbj. Gx <= h
        c = np.array(c).astype(np.float32)
        G = np.array(G).astype(np.float32)
        h = np.array(h).astype(np.float32)

        slack_vars = np.eye(h.shape[0])
        A = np.hstack([G, slack_vars])
        b = np.array(h)
        c = np.concatenate([c, np.zeros(h.shape[0])])
        return ProblemPreprocessingUtils.preprocess_problem(c, A, b)

    @staticmethod
    def add_variables_bounds_to_coefficient_matrix(c, A, b, lb, ub):
        # assumes problem is of form min cx sbj. Ax=b lb<=x<=ub
        c = np.array(c).astype(np.float32)
        A = np.array(A).astype(np.float32)
        b = np.array(b).astype(np.float32)

        if lb is None:
            lb = np.repeat(0.0, A.shape[1])
        else:
            lb = np.array(lb).astype(np.float32)
        if ub is None:
            ub = np.repeat(np.inf, A.shape[1])
        else:
            ub = np.array(ub).astype(np.float32)

        # add lb/ub surplus/slack vars to A
        lb_surplus_index = ~np.isclose(lb, 0.0)
        ub_slack_index = ~np.isclose(ub, np.inf)
        if lb_surplus_index.any() or ub_slack_index.any():
            offset = A.shape[1]
            zeros = np.zeros(
                (
                    A.shape[0],
                    offset * (int(lb_surplus_index.any()) + int(ub_slack_index.any())),
                )
            )
            A = np.hstack([A, zeros])
            for i, bnd in enumerate(lb_surplus_index):
                if not bnd:
                    continue
                # add constraint of form `x_i - s_i = lb_i` to A
                A = np.vstack([A, np.zeros((1, A.shape[1]))])
                A[-1, i] += 1
                A[-1, offset + i] -= 1
            for i, bnd in enumerate(ub_slack_index):
                if not bnd:
                    continue
                # add constraint of form `x_i + s_i = ub_i` to A
                A = np.vstack([A, np.zeros((1, A.shape[1]))])
                A[-1, i] += 1
                A[-1, 2 * offset + i] += 1
        # vars are now all 0 <= x < inf
        # absorb lower/upper bounds into `b`
        b = np.concatenate([b, lb[lb_surplus_index], ub[ub_slack_index]])
        c = np.concatenate([c, np.zeros(lb_surplus_index.sum() + ub_slack_index.sum())])
        return ProblemPreprocessingUtils.preprocess_problem(c, A, b)
