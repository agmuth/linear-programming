import numpy as np

from linprog.data_classes import LinProgResult
from linprog.exceptions import (BasisIsPrimalInfeasibleError,
                                PrimalIsUnboundedError)
from linprog.preprocessing import ProblemPreprocessingUtils
from linprog.utils import primal_simplex_div


class PrimalNaiveSimplexSolver:
    """Naive Primal Simplex algorithm that implements Bland's selection rule to avoid cycling."""

    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        basis : np.array
            array of length `m` mapping columns in `A` to their indicies in the basic feasible solution (bfs).
        """
        (
            self.c,
            self.A,
            self.b,
        ) = self._preprocess_problem(c, A, b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis).astype(int)
        # need to set these here instead of calling `_update` mthods for inheritence
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = self.inv_basis_matrix @ self.b
        self.counter = None
        self.optimum = None
        self._check_basis_feasibility()

    def _check_basis_feasibility(self):
        if not self._basis_is_primal_feasible():
            raise BasisIsPrimalInfeasibleError

    def _basis_is_primal_feasible(self):
        return np.all(np.linalg.inv(self.A[:, self.basis]) @ self.b >= 0.0)

    def _basis_is_dual_feasible(self):
        return np.all(
            self.c[self.basis] @ np.linalg.inv(self.A[:, self.basis]) @ self.A <= self.c
        )

    def _preprocess_problem(self, c: np.array, A: np.array, b: np.array):
        """Misc preprocessing."""
        return ProblemPreprocessingUtils.preprocess_problem(c, A, b)

    def _get_reduced_costs(self):
        """
        Get the reduced cost vector ie the cost of `x_i` minus the cost of representing `x_i`
        as a linear combination of the current basis.
        """
        reduced_costs = self.c - self.c[self.basis] @ self.inv_basis_matrix @ self.A
        reduced_costs[self.basis] = 0  # avoid numerical errors
        return reduced_costs

    def _update_bfs(self):
        """Update current basic feasible solution."""
        self.bfs = self.inv_basis_matrix @ self.b

    def _get_bfs_expanded(self):
        x = np.zeros(self.n)
        x[self.basis] = self.bfs
        return x

    def _calc_current_cost(self):
        """calculate the cost/onjective value of the current basic feasible solution."""
        return np.dot(self.c, self._get_bfs_expanded())

    def _get_solver_return_object(self):
        """Build return object from calling `self.solve`."""
        res = LinProgResult(
            x=self._get_bfs_expanded(),
            basis=self.basis,
            cost=self._calc_current_cost(),
            iters=self.counter,
            optimum=self.optimum,
        )
        return res

    def _update_inv_basis_matrix(self):
        """Naively update inverse basis matrix by inverting subset of columns in `A`."""
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])

    def _update_basis(
        self, col_in_basis_to_leave_basis: int, col_in_A_to_enter_basis: int
    ):
        """Update basis corresponding to current basic feasible solution.

        Parameters
        ----------
        col_in_basis_to_leave_basis : int
            Index of column in/wrt `basis` to leave `basis`.
        col_in_A_to_enter_basis : int
            Index of column in/wrt `A` to enter `basis`.
        """
        self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis

    def _primal_get_feasible_direction(self, col_in_A_to_enter_basis: int) -> np.array:
        """Get feasible primal direction wrt non basic variable entering the basis.

        Parameters
        ----------
        col_in_A_to_enter_basis : int
            col_in_A_to_enter_basis : int
                Index of column in/wrt `A` to enter `basis`.

        Returns
        -------
        np.array
            Feasible primal direction.

        Raises
        ------
        ValueError
            Value error if problem is unbounded.
        """
        feasible_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis]
        if self._primal_check_for_unbnoundedness(feasible_direction):
            # optimal cost is -inf -> problem is unbounded
            raise PrimalIsUnboundedError
        return feasible_direction

    def _primal_get_col_in_A_to_enter_basis(self, reduced_costs: np.array):
        """Returns index of nonbasic variable in A to enter basis."""
        col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
        return col_in_A_to_enter_basis

    def _primal_check_for_optimality(self, reduced_costs) -> bool:
        """Current basic feasible solution is optimal if reduced ccosts are all non negative."""
        return (reduced_costs.min() >= 0) or np.isclose(reduced_costs.min(), 0)

    def _primal_check_for_unbnoundedness(self, feasible_direction: np.array) -> bool:
        """Problem is unbounded if we can move infintely far in the feasible direction."""
        problem_is_unbounded = feasible_direction.max() <= 0
        return problem_is_unbounded

    def _primal_ratio_test(self, col_in_A_to_enter_basis: int) -> int:
        """Primal ratio test to see how far we can move in the feasible direction while maintaining primal feasibility.

        Parameters
        ----------
        col_in_A_to_enter_basis : int
            Index of column in/wrt `A` to enter `basis`.

        Returns
        -------
        int
            Index of column in/wrt `basis` to leave `basis`.
        """
        feasible_direction = self._primal_get_feasible_direction(
            col_in_A_to_enter_basis
        )
        thetas = primal_simplex_div(self.bfs, feasible_direction)
        col_in_basis_to_leave_basis = np.argmin(thetas)
        return col_in_basis_to_leave_basis

    def pivot(
        self, col_in_basis_to_leave_basis: np.array, col_in_A_to_enter_basis: np.array
    ):
        self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
        self._update_inv_basis_matrix()
        self._update_bfs()

    def solve(self, maxiters: int = 100):
        """Primal Simplex algorithm loop.

        Parameters
        ----------
        maxiters : int, optional
            maximum number of simplex steps, by default 100

        Returns
        -------
        _type_
            _description_
        """
        self.counter = 0
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                self.optimum = True
                break

            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(
                reduced_costs
            )
            col_in_basis_to_leave_basis = self._primal_ratio_test(
                col_in_A_to_enter_basis
            )

            self.pivot(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)

        return self._get_solver_return_object()


class PrimalRevisedSimplexSolver(PrimalNaiveSimplexSolver):
    """Revised Primal Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `PrimalNaiveSimplexSolver`.
    """

    def _calc_premultiplication_inv_basis_update_matrix(
        self, col_in_A_to_enter_basis, col_in_basis_to_leave_basis
    ) -> np.array:
        """Calculate matrix to premultiply `self.inv_basis_matrix` by corresponding to a basis change.

        Parameters
        ----------
        col_in_A_to_enter_basis : _type_
            Index of column in/wrt `A` to enter `basis`.
        col_in_basis_to_leave_basis : _type_
            Index of column in/wrt `basis` to leave `basis`.

        Returns
        -------
        np.array
            premultiplication matrix
        """
        feasible_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis]
        premult_inv_basis_update_matrix = np.eye(self.m)
        premult_inv_basis_update_matrix[
            :, col_in_basis_to_leave_basis
        ] = -feasible_direction
        premult_inv_basis_update_matrix[
            col_in_basis_to_leave_basis, col_in_basis_to_leave_basis
        ] = 1
        premult_inv_basis_update_matrix[
            :, col_in_basis_to_leave_basis
        ] /= feasible_direction[col_in_basis_to_leave_basis]
        return premult_inv_basis_update_matrix

    def _update_inv_basis_matrix(self, premult_inv_basis_update_matrix):
        """Override `_update_inv_basis_matrix` from `PrimalNaiveSimplexSolver`."""
        self.inv_basis_matrix = premult_inv_basis_update_matrix @ self.inv_basis_matrix

    def _update_bfs(self, premult_inv_basis_update_matrix):
        """Override `_update_bfs` from `PrimalNaiveSimplexSolver`."""
        self.bfs = premult_inv_basis_update_matrix @ self.bfs

    def pivot(self, col_in_basis_to_leave_basis, col_in_A_to_enter_basis):
        premult_inv_basis_update_matrix = (
            self._calc_premultiplication_inv_basis_update_matrix(
                col_in_A_to_enter_basis, col_in_basis_to_leave_basis
            )
        )
        self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
        self._update_inv_basis_matrix(premult_inv_basis_update_matrix)
        self._update_bfs(premult_inv_basis_update_matrix)
