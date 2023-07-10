import numpy as np
from linprog.tableau import Tableau
from linprog.utils import *
from linprog.primal_simplex_solvers import *


class DualNaiveSimplexSolver(PrimalNaiveSimplexSolver):
    """Naive Dual Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `PrimalNaiveSimplexSolver`.
    """
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
        feasible_direction = self.inv_basis_matrix[col_in_basis_to_leave_basis, :] @ self.A
        feasible_direction[self.basis] = 0  # avoid numerical errors
        if self._dual_check_for_unbnoundedness(feasible_direction):  # maybe move into calc
                raise ValueError("Feasible direction is non negative. Problem is unbounded.")
        return feasible_direction
    
    def _dual_get_col_in_basis_to_leave_basis(self) -> bool:
        """Returns index of basic variable in basis to leave basis."""
        col_in_basis_to_leave_basis = np.argmax(self.bfs < 0)
        return col_in_basis_to_leave_basis
         
    def _dual_check_for_optimality(self) -> bool:
        """Solution to the dual problem is optimal if it also satisfies primal problem."""
        bfs_is_optimal = (self.bfs.min() >= 0)
        return bfs_is_optimal
    
    def _dual_check_for_unbnoundedness(self, feasible_direction: np.array) -> bool:
        """Problem is unbounded if we can move infintely far in the feasible direction."""
        problem_is_unbounded = (feasible_direction.min() >= 0)
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
        feasible_direction = self._dual_get_feasible_direction(col_in_basis_to_leave_basis)
        reduced_costs = self._get_reduced_costs()
        thetas = dual_simplex_div(reduced_costs, feasible_direction)
        col_in_A_to_enter_basis = np.argmin(thetas)
        return col_in_A_to_enter_basis

    def solve(self, maxiters: int=100):
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
                break

            col_in_basis_to_leave_basis = self._dual_get_col_in_basis_to_leave_basis()
            col_in_A_to_enter_basis = self._dual_ratio_test(col_in_basis_to_leave_basis)
            self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
            self._update_inv_basis_matrix()
            self._update_bfs()           

        return self._get_solver_return_values()
    
    
class DualRevisedSimplexSolver(DualNaiveSimplexSolver, PrimalRevisedSimplexSolver):
    """Revised Dual Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `DualNaiveSimplexSolver` and `PrimalRevisedSimplexSolver`.
    """
   
    def solve(self, maxiters: int=100):
        """Override `solve` from `DualNaiveSimplexSolver`."""
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            if self._dual_check_for_optimality():
                # optimal solution found break
                break

            col_in_basis_to_leave_basis = self._dual_get_col_in_basis_to_leave_basis()
            feasible_direction = self._dual_get_feasible_direction(col_in_basis_to_leave_basis)

            if self._dual_check_for_unbnoundedness(feasible_direction):
                raise ValueError("Problem is unbounded.")

            reduced_costs = self._get_reduced_costs()
            
            thetas = dual_simplex_div(reduced_costs, feasible_direction)
            col_in_A_to_enter_basis = np.argmin(thetas)
    
            premultiplication_inv_basis_update_matrix = self._calc_premultiplication_inv_basis_update_matrix(col_in_A_to_enter_basis, col_in_basis_to_leave_basis)
            self._update_basis(col_in_basis_to_leave_basis,  col_in_A_to_enter_basis)
            self._update_of_inv_basis_matrix(premultiplication_inv_basis_update_matrix)
            self._update_update_bfs(premultiplication_inv_basis_update_matrix)

        return self._get_solver_return_values()


class DualTableauSimplexSolver():
    """Tableau implementation of Dual Simplex Algorithm."""
    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array) -> None:
        """init

        Parameters
        ----------
        c : np.array
            (1, n) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m, 1) vector defining the equality constraints.
        basis : np.array
            array of length `m` mapping columns in `A` to their indicies in the basic feasible solution (bfs).
        """
        self.tableau = Tableau(c, A, b, basis)
    
    def solve(self, maxiters=100):
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
            self.tableau.tableau[0, 1:][self.tableau.basis] = 0  # avoid numerical errors
            if self.tableau.tableau[1:, 0].min() >= 0:  # check for termination condition
                break

            pivot_row = np.argmax(self.tableau.tableau[1:, 0] < 0) + 1

            if self.tableau.tableau[pivot_row, 1:].min() >= 0:
                raise ValueError("`pivot_row` entries are all non negative. problem is unbounded.")


            self.tableau.tableau[pivot_row, 1:] < 0

            pivot_col = np.argmin(
                [
                    r if v < 0 else np.inf for v, r in zip(
                        self.tableau.tableau[pivot_row, 1:],
                        primal_simplex_div(
                            self.tableau.tableau[0, 1:], 
                            np.abs(self.tableau.tableau[pivot_row, 1:])
                        )
                    )
                    
                ]
        
            ) \
            + 1  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]
        return {"x": self.bfs, "basis": self.basis, "cost": self.tableau.tableau[0, 0], "iters": self.counter}
