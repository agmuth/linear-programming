import numpy as np
from linearprogramming.tableau import Tableau
from linearprogramming.utils import *
from linearprogramming.primal_simplex_solvers import *


class DualNaiveSimplexSolver(PrimalNaiveSimplexSolver):
    """
        Naive  Dual Simplex algorithm that implements Bland's selection rule to avoid cycling. 
    
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)
        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
    """

    def _dual_get_search_direction(self, col_in_basis_to_leave_basis):
        search_direction = self.inv_basis_matrix[col_in_basis_to_leave_basis, :] @ self.A
        search_direction[self.basis] = 0  # avoid numerical errors
        return search_direction
    
    def _dual_get_col_in_basis_to_leave_basis(self):
        col_in_basis_to_leave_basis = np.argmax(self.bfs < 0)
        return col_in_basis_to_leave_basis
         
    def _dual_check_for_optimality(self):
        bfs_is_optimal = (self.bfs.min() >= 0)
        return bfs_is_optimal
    
    def _dual_check_for_unbnoundedness(self, search_direction):
        problem_is_unbounded = (search_direction.min() >= 0)
        return problem_is_unbounded

    def solve(self, maxiters: int=100):
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            if self._dual_check_for_optimality():
                # optimal solution found break
                break

            col_in_basis_to_leave_basis = self._dual_get_col_in_basis_to_leave_basis()
            search_direction = self._dual_get_search_direction(col_in_basis_to_leave_basis)

            if self._dual_check_for_unbnoundedness(search_direction):
                raise ValueError("Problem is unbounded.")

            reduced_costs = self._get_reduced_costs()
            
            thetas = dual_simplex_div(reduced_costs, search_direction)
            col_in_A_to_enter_basis = np.argmin(thetas)
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            self._update_inv_basis_matrix()
            self.bfs = self.inv_basis_matrix @ self.b            

        return self._get_solver_return_values()
    

class DualRevisedSimplexSolver(DualNaiveSimplexSolver, PrimalRevisedSimplexSolver):
    """
        Revised  Dual Simplex algorithm that implements Bland's selection rule to avoid cycling. 

        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)
        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
    """
   
    def solve(self, maxiters: int=100):
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            if self._dual_check_for_optimality():
                # optimal solution found break
                break

            col_in_basis_to_leave_basis = self._dual_get_col_in_basis_to_leave_basis()
            search_direction = self._dual_get_search_direction(col_in_basis_to_leave_basis)

            if self._dual_check_for_unbnoundedness(search_direction):
                raise ValueError("Problem is unbounded.")

            reduced_costs = self._get_reduced_costs()
            
            thetas = dual_simplex_div(reduced_costs, search_direction)
            col_in_A_to_enter_basis = np.argmin(thetas)
    
            premultiplication_inv_basis_update_matrix = self._calc_premultiplication_inv_basis_update_matrix(col_in_A_to_enter_basis, col_in_basis_to_leave_basis)
            self._update_basis(col_in_basis_to_leave_basis,  col_in_A_to_enter_basis)
            self._update_of_inv_basis_matrix(premultiplication_inv_basis_update_matrix)
            self._update_update_bfs(premultiplication_inv_basis_update_matrix)


        return self._get_solver_return_values()


class DualTableauSimplexSolver():
    """
        Tableau based Simplex algorithm.  
    """
    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
        """
        self.tableau = Tableau(c, A, b, basis)
    
    def solve(self, maxiters=100):
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
