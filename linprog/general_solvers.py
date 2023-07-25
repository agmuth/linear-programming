import numpy as np

from linprog.primal_solvers import *


class TwoPhaseRevisedSimplexSolver:
    def __init__(self, c: np.array, A: np.array, b: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): length n vector cost vector.
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): length m vector defining the equalies constraints.
        """
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1

    def _phase_one(self, maxiters: int = 100):
        """Phase I of two phase simplex method.

        Parameters
        ----------
        maxiters : int, optional
            _description_, by default 100

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        c = np.hstack([np.zeros(self.n), np.ones(self.m)])
        A = np.hstack([self.A, np.eye(self.m)])
        b = np.array(self.b)
        basis = np.arange(self.n, self.n + self.m)
        solver = PrimalRevisedSimplexSolver(c, A, b, basis)
        res = solver.solve(maxiters)

        if res.cost > 0:
            if res.optimum:
                raise ValueError("Problem is unfeasible.")
            else:
                raise ValueError("Phase one did not converge.")

        if res.basis.max() >= self.n:
            # artificial variables still in basis at zero level
            # pivot out when/where possible
            for i in range(self.n, self.n + self.m):
                if i in res.basis:
                    non_basic_non_artifical_vars = np.array(
                        [i for i in range(self.n) if i not in res.basis]
                    )
                    potential_pivots = (solver.inv_basis_matrix @ solver.A)[
                        np.argmax(res.basis == i), non_basic_non_artifical_vars
                    ] > 0
                    if potential_pivots.any():
                        col_in_A_to_enter_basis = non_basic_non_artifical_vars[
                            np.argmax(potential_pivots)
                        ]
                        col_in_basis_to_leave_basis = np.where(res.basis == i)[0][0]
                        solver.pivot(
                            col_in_basis_to_leave_basis, col_in_A_to_enter_basis
                        )
                        res = solver._get_solver_return_object()

        if res.basis.max() >= self.n:
            # unable to pivot out one or more artifical vairables from basis
            # constraints redundent -> remove
            non_redundent_equs_idx = res.basis <= self.n
            self.A = self.A[non_redundent_equs_idx]
            self.b = self.b[non_redundent_equs_idx]
            self.m = non_redundent_equs_idx.sum()
            res.basis = res.basis[non_redundent_equs_idx]

        return res.basis

    def solve(
        self,
        maxiters1: int = 100,
        maxiters2: int = 100,
        return_phase_1_basis: bool = False,
    ):
        """Implement two phase simplex solver.

        Parameters
        ----------
        maxiters1 : int, optional
            Maximum number of iters in phase one, by default 100
        maxiters2 : int, optional
           Maximum number of iters in phase two, by default 100
        return_phase_1_basis : bool, optional
            Return basis from phase one no phase two, by default False

        Returns
        -------
        _type_
            _description_
        """
        phase_2_starting_basis = self._phase_one(maxiters1)
        if return_phase_1_basis:
            return phase_2_starting_basis
        # phase 2
        solver = PrimalRevisedSimplexSolver(
            self.c, self.A, self.b, phase_2_starting_basis
        )
        res = solver.solve(maxiters2)
        return res
