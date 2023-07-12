from math import factorial

import numpy as np

from linprog.primal_simplex_solvers import *
from linprog.utils import primal_simplex_div, get_bounds_on_bfs


class PrimalDualAlgorithm:
    """Primal-Dual Algorithm for Linear Programs."""

    def __init__(self, c: np.array, A: np.array, b: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        """
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.counter = 0
        self.optimum=False

    def solve(self):
        """Loop implementeing primal-dual algorithm.

        Returns
        -------
        _type_
            _description_
        """
        # zero vector is dual feasible if c >= 0
        bfs_unrestricted_dual = np.zeros(self.m)
        expanded_dual_to_get_initial_bfs = False
        if self.c.min() < 0:
            expanded_dual_to_get_initial_bfs = True
            M = get_bounds_on_bfs(self.A, self.b)
            # pg. 105 combinatorial optimization - algorithms and complexity
            self.c = np.hstack([self.c, np.zeros(1)])
            self.A = np.vstack(
                [np.hstack([self.A, np.zeros((self.m, 1))]), np.ones((1, self.n + 1))]
            )
            self.b = np.hstack([self.b, self.n * M * np.ones(1)])
            self.m, self.n = self.A.shape
            bfs_unrestricted_dual = np.hstack(
                [bfs_unrestricted_dual, self.c.min() * np.ones(1)]
            )

        while True:
            # solve restricted primal
            admissiable_set = np.isclose(bfs_unrestricted_dual @ self.A, self.c)
            inadmissable_set = ~admissiable_set

            c_restricted_primal = np.hstack(
                [np.zeros(admissiable_set.sum()), np.ones(self.m)]
            )
            A_restricted_primal = np.hstack(
                [self.A[:, admissiable_set], np.eye(self.m)]
            )
            b_restricted_primal = np.array(self.b)
            basis_restricted_primal = (
                np.arange(self.m) + admissiable_set.sum()
            )  # take artifical vars as basis

            solver_restricted_primal = PrimalRevisedSimplexSolver(
                c_restricted_primal,
                A_restricted_primal,
                b_restricted_primal,
                basis_restricted_primal,
            )
            res_restricted_primal = solver_restricted_primal.solve()

            if res_restricted_primal["cost"] > 0.0:
                # complementary slackness/primal feasibility to original problem not satisfied/attained
                # modify dual soln so that more variables are able to take non zero values in the restricted primal
                basis_restricted_primal = res_restricted_primal["basis"]
                bfs_restricted_dual = c_restricted_primal[
                    basis_restricted_primal
                ] @ np.linalg.inv(A_restricted_primal[:, basis_restricted_primal])
                theta = np.min(
                    primal_simplex_div(
                        self.c - bfs_unrestricted_dual @ self.A,
                        bfs_restricted_dual @ self.A,
                    )[inadmissable_set]
                )
                bfs_unrestricted_dual += theta * bfs_restricted_dual
            else:
                break  # complementary slackness attained

        bfs_restricted_primal = np.zeros(2 * admissiable_set.sum())
        bfs_restricted_primal[res_restricted_primal["basis"]] += res_restricted_primal[
            "x"
        ]

        bfs_unrestricted_primal = np.zeros(self.n)
        bfs_unrestricted_primal[admissiable_set] += bfs_restricted_primal[
            : admissiable_set.sum()
        ]

        basis_unrestricted_primal = np.arange(self.n)[admissiable_set][
            res_restricted_primal["basis"][
                res_restricted_primal["basis"] < admissiable_set.sum()
            ]
        ]
        cost_unrestricted_primal = np.dot(self.c, bfs_unrestricted_primal)

        if expanded_dual_to_get_initial_bfs:
            basis_unrestricted_primal = basis_unrestricted_primal[
                basis_unrestricted_primal != self.n - 1
            ]
            bfs_unrestricted_primal = bfs_unrestricted_primal[:-1]

        res = {
            "bfs": bfs_unrestricted_primal,
            "basis": basis_unrestricted_primal,
            "cost": cost_unrestricted_primal,
            "iters": -1,
        }

        return res
