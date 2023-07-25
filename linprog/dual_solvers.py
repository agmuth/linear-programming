import numpy as np

from linprog.exceptions import BasisIsDualInfeasibleError, DualIsUnboundedError
from linprog.primal_solvers import (PrimalNaiveSimplexSolver,
                                    PrimalRevisedSimplexSolver)
from linprog.utils import dual_simplex_div


class DualNaiveSimplexSolver(PrimalNaiveSimplexSolver):
    """Naive Dual Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `PrimalNaiveSimplexSolver`.
    """

    def _check_basis_feasibility(self):
        if not self._basis_is_dual_feasible():
            raise BasisIsDualInfeasibleError

    def _dual_get_feasible_direction(self, col_in_basis_to_leave_basis) -> np.array:
        """Get feasible dual direction wrt basic variable leaving the basis.

        Parameters
        ----------
        col_in_A_to_enter_basis : int
            col_in_A_to_enter_basis : int
                Index of column in/wrt `A` to leave `basis`.

        Returns
        -------
        np.array
            Feasible dual direction.

        Raises
        ------
        ValueError
            Value error if problem is unbounded.
        """
        feasible_direction = (
            self.inv_basis_matrix[col_in_basis_to_leave_basis, :] @ self.A
        )
        feasible_direction[self.basis] = 0  # avoid numerical errors
        if self._dual_check_for_unbnoundedness(
            feasible_direction
        ):  # maybe move into calc
            raise DualIsUnboundedError
        return feasible_direction

    def _dual_get_col_in_basis_to_leave_basis(self) -> bool:
        """Returns index of basic variable in basis to leave basis."""
        col_in_basis_to_leave_basis = np.argmax(self.bfs < 0)
        return col_in_basis_to_leave_basis

    def _dual_check_for_optimality(self) -> bool:
        """Solution to the dual problem is optimal if it also satisfies primal problem."""
        return (self.bfs.min() >= 0) or np.isclose(self.bfs.min(), 0)

    def _dual_check_for_unbnoundedness(self, feasible_direction: np.array) -> bool:
        """Problem is unbounded if we can move infintely far in the feasible direction."""
        problem_is_unbounded = feasible_direction.min() >= 0
        return problem_is_unbounded

    def _dual_ratio_test(self, col_in_basis_to_leave_basis: int) -> int:
        """Dual ratio test to see how far we can move in the feasible direction while maintaining dual feasibility.

        Parameters
        ----------
        col_in_basis_to_leave_basis : int
            Index of column in/wrt `basis` to leave `basis`.

        Returns
        -------
        int
            Index of column in/wrt `A` to enter `basis`.
        """
        feasible_direction = self._dual_get_feasible_direction(
            col_in_basis_to_leave_basis
        )
        reduced_costs = self._get_reduced_costs()
        thetas = dual_simplex_div(reduced_costs, feasible_direction)
        col_in_A_to_enter_basis = np.argmin(thetas)
        return col_in_A_to_enter_basis

    def solve(self, maxiters: int = 100):
        """Dual Simplex algorithm loop.

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
        while self.counter < maxiters:
            self.counter += 1

            if self._dual_check_for_optimality():
                # optimal solution found break
                self.optimum = True
                break

            col_in_basis_to_leave_basis = self._dual_get_col_in_basis_to_leave_basis()
            col_in_A_to_enter_basis = self._dual_ratio_test(col_in_basis_to_leave_basis)
            self.pivot(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)

        return self._get_solver_return_object()


class DualRevisedSimplexSolver(PrimalRevisedSimplexSolver, DualNaiveSimplexSolver):
    """Revised Dual Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `DualNaiveSimplexSolver` and `PrimalRevisedSimplexSolver`.
    """

    def pivot(self, col_in_basis_to_leave_basis, col_in_A_to_enter_basis):
        # bound pivot method to `PrimalRevisedSimplexSolver`
        PrimalRevisedSimplexSolver.pivot(
            self, col_in_basis_to_leave_basis, col_in_A_to_enter_basis
        )
