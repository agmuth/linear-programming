import numpy as np

from linprog.tableau import Tableau
from linprog.utils import *


class PrimalNaiveSimplexSolver():
    """Naive Primal Simplex algorithm that implements Bland's selection rule to avoid cycling."""
    
    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

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
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis)
        # need to set these here instead of calling `_update` mthods for inheritence 
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = self.inv_basis_matrix @ self.b
        self.counter = None

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
    
    def _get_solver_return_values(self):
        """Build return object from calling `self.solve`."""
        res = {"x": self.bfs, "basis": self.basis, "cost": self._calc_bfs_cost(), "iters": self.counter}
        return res

    def _calc_bfs_cost(self):
        """calculate the cost/onjective value of the current basic feasible solution."""
        return np.dot(self.c[self.basis], self.bfs)

    def _update_inv_basis_matrix(self):
        """Naively update inverse basis matrix by inverting subset of columns in `A`."""
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])

    def _update_basis(self, col_in_basis_to_leave_basis: int, col_in_A_to_enter_basis: int):
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
            raise ValueError("Reduced cost vector has all non-positive entries. Rroblem is unbounded.") 
        return feasible_direction
    
    def _primal_get_col_in_A_to_enter_basis(self, reduced_costs: np.array):
        """Returns index of nonbasic variable in A to enter basis."""
        col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
        return col_in_A_to_enter_basis
    
    def _primal_check_for_optimality(self, reduced_costs) -> bool:
        """Current basic feasible solution is optimal if reduced ccosts are all non negative."""
        bfs_is_optimal = (reduced_costs.min() >= 0)
        return bfs_is_optimal
    
    def _primal_check_for_unbnoundedness(self, feasible_direction: np.array) -> bool:
        """Problem is unbounded if we can move infintely far in the feasible direction."""
        problem_is_unbounded = (feasible_direction.max() <= 0)
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
        feasible_direction = self._primal_get_feasible_direction(col_in_A_to_enter_basis)
        thetas = primal_simplex_div(self.bfs, feasible_direction)
        col_in_basis_to_leave_basis = np.argmin(thetas)
        return col_in_basis_to_leave_basis

    def solve(self, maxiters: int=100):
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
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                break
            
            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(reduced_costs)
            col_in_basis_to_leave_basis = self._primal_ratio_test(col_in_A_to_enter_basis)

            self._update_basis(col_in_basis_to_leave_basis,  col_in_A_to_enter_basis)
            self._update_inv_basis_matrix()
            self._update_bfs()

        return self._get_solver_return_values()
    
    

class PrimalRevisedSimplexSolver(PrimalNaiveSimplexSolver):
    """Revised Primal Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `PrimalNaiveSimplexSolver`.
    """

    def _calc_premultiplication_inv_basis_update_matrix(self, col_in_A_to_enter_basis, col_in_basis_to_leave_basis) -> np.array:
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
        premult_inv_basis_update_matrix[:, col_in_basis_to_leave_basis] = -feasible_direction
        premult_inv_basis_update_matrix[col_in_basis_to_leave_basis, col_in_basis_to_leave_basis] = 1
        premult_inv_basis_update_matrix[:, col_in_basis_to_leave_basis] /= feasible_direction[col_in_basis_to_leave_basis]
        return premult_inv_basis_update_matrix

    def _update_of_inv_basis_matrix(self, premult_inv_basis_update_matrix):      
        """Override `_update_of_inv_basis_matrix` from `PrimalNaiveSimplexSolver`."""
        self.inv_basis_matrix = premult_inv_basis_update_matrix @ self.inv_basis_matrix

    def _update_update_bfs(self, premult_inv_basis_update_matrix):
        """Override `_update_update_bfs` from `PrimalNaiveSimplexSolver`."""
        self.bfs = premult_inv_basis_update_matrix @ self.bfs

    def solve(self, maxiters: int=100):
        """Override `solve` from `PrimalNaiveSimplexSolver`."""
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                break
            
            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(reduced_costs)
            col_in_basis_to_leave_basis = self._primal_ratio_test(col_in_A_to_enter_basis)

            premult_inv_basis_update_matrix = self._calc_premultiplication_inv_basis_update_matrix(col_in_A_to_enter_basis, col_in_basis_to_leave_basis)

            self._update_basis(col_in_basis_to_leave_basis,  col_in_A_to_enter_basis)
            self._update_of_inv_basis_matrix(premult_inv_basis_update_matrix)
            self._update_update_bfs(premult_inv_basis_update_matrix)

        return self._get_solver_return_values()
    

class PrimalTableauSimplexSolver():
    """Tableau implementation of Primal Simplex Algorithm."""
    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

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
        while self.counter < maxiters:
            self.counter += 1
            self.tableau.tableau[0, 1:][self.tableau.basis] = 0  # avoid numerical errors
            if self.tableau.tableau[0, 1:].min() >= 0:  # 0^th row is reduced costs
                # optimal solution found break
                break

            pivot_col = np.argmax(self.tableau.tableau[0, 1:] < 0) + 1

            if self.tableau.tableau[1:, pivot_col].max() <= 0:  # check to make sure search direction has at least one positive entry
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("Reduced cost vector has all non-positive entries. Rroblem is unbounded.")

            pivot_row = np.argmin(
                primal_simplex_div(
                        self.tableau.tableau[1:, 0], 
                        self.tableau.tableau[1:, pivot_col]  # feasible direction
                )
        
            ) \
            + 1  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]
        return {"x": self.bfs, "basis": self.basis, "cost": self.tableau.tableau[0, 0], "iters": self.counter}

