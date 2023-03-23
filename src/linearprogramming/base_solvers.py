import numpy as np

from linearprogramming.tableau import Tableau
from linearprogramming.utils import *


class PrimalNaiveSimplexSolver():
    """
        Naive Simplex algorithm that implements Bland's selection rule to avoid cycling. 
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
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis)
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = self.inv_basis_matrix @ self.b
        self.counter = 0

    def _get_reduced_costs(self):
        reduced_costs = self.c - self.c[self.basis] @ self.inv_basis_matrix @ self.A
        reduced_costs[self.basis] = 0  # avoid numerical errors
        return reduced_costs
    
    def _get_solver_return_values(self):
        res = {"x": self.bfs, "basis": self.basis, "cost": np.dot(self.c[self.basis], self.bfs), "iters": self.counter}
        return res

    def _update_inv_basis_matrix(self):
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
    
    def _primal_get_search_direction(self, col_in_A_to_enter_basis):
        search_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] 
        return search_direction
    
    def _primal_get_col_in_A_to_enter_basis(self, reduced_costs):
        col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
        return col_in_A_to_enter_basis
    
    def _primal_check_for_optimality(self, reduced_costs):
        bfs_is_optimal = (reduced_costs.min() >= 0)
        return bfs_is_optimal
    
    def _primal_check_for_unbnoundedness(self, search_direction):
        problem_is_unbounded = (search_direction.max() <= 0)
        return problem_is_unbounded

    def solve(self, maxiters: int=100):
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                break
            
            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(reduced_costs)
            search_direction = self._primal_get_search_direction(col_in_A_to_enter_basis)

            if self._primal_check_for_unbnoundedness(search_direction):
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("Problem is unbounded.")

            thetas = primal_simplex_div(self.bfs, search_direction)
            col_in_basis_to_leave_basis = np.argmin(thetas)
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            self._update_inv_basis_matrix()
            self.bfs = self.inv_basis_matrix @ self.b

        return self._get_solver_return_values()


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


class PrimalRevisedSimplexSolver(PrimalNaiveSimplexSolver):
    """
        Revised Simplex algorithm that implements Bland's selection rule to avoid cycling. 

        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)
        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
    """ 

    def _revised_update_of_inv_basis_matrix(self, col_in_A_to_enter_basis, col_in_basis_to_leave_basis):
        search_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis]
        self.inv_basis_matrix = np.hstack([self.inv_basis_matrix, np.expand_dims(search_direction, 1)])
        self.inv_basis_matrix[col_in_basis_to_leave_basis, :] /= search_direction[col_in_basis_to_leave_basis]
        for i in range(self.m):
            if i == col_in_basis_to_leave_basis:
                continue
            self.inv_basis_matrix[i, :] -= search_direction[i] * self.inv_basis_matrix[col_in_basis_to_leave_basis, :] 
        self.inv_basis_matrix = self.inv_basis_matrix[:, :-1]
 
    def solve(self, maxiters: int=100):
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                break
            
            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(reduced_costs)
            search_direction = self._primal_get_search_direction(col_in_A_to_enter_basis)

            if self._primal_check_for_unbnoundedness(search_direction):
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("Problem is unbounded.")

            thetas = primal_simplex_div(self.bfs, search_direction)
            
            # update bfs w\o matrix mult - again probably not any better as oppsed to numpy
            col_in_basis_to_leave_basis = np.argmin(thetas)
            theta_star = thetas[col_in_basis_to_leave_basis]
            self.bfs -= theta_star * search_direction
            self.bfs[col_in_basis_to_leave_basis] = theta_star
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            
            # update basis and `self.inv_basis_matrix` w\o having to invert - again probably not any better as opposed to numpy
            self._revised_update_of_inv_basis_matrix(col_in_A_to_enter_basis, col_in_basis_to_leave_basis)

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
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            self._revised_update_of_inv_basis_matrix(col_in_A_to_enter_basis, col_in_basis_to_leave_basis)
            self.bfs = self.inv_basis_matrix @ self.b  


        return self._get_solver_return_values()


class PrimalTableauSimplexSolver():
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
            if self.tableau.tableau[0, 1:].min() >= 0:  # 0^th row is reduced costs
                # optimal solution found break
                break

            pivot_col = np.argmax(self.tableau.tableau[0, 1:] < 0) + 1

            if self.tableau.tableau[1:, pivot_col].max() <= 0:  # check to make sure search direction has at least one positive entry
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("reduced cost vector has all non-positive entries. problem is unbounded.")

            pivot_row = np.argmin(
                primal_simplex_div(
                        self.tableau.tableau[1:, 0], 
                        self.tableau.tableau[1:, pivot_col]  # search direction
                )
        
            ) \
            + 1  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]
        return {"x": self.bfs, "basis": self.basis, "cost": self.tableau.tableau[0, 0], "iters": self.counter}


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

