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

    def _update_basis(self, col_in_basis_to_leave_basis, col_in_A_to_enter_basis):
        self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
    
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
            self._update_basis(col_in_basis_to_leave_basis,  col_in_A_to_enter_basis)
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

    def _calc_premultiplication_inv_basis_update_matrix(self, col_in_A_to_enter_basis, col_in_basis_to_leave_basis):
        search_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis]
        premultiplication_inv_basis_update_matrix = np.eye(self.m)
        premultiplication_inv_basis_update_matrix[:, col_in_basis_to_leave_basis] = -search_direction
        premultiplication_inv_basis_update_matrix[col_in_basis_to_leave_basis, col_in_basis_to_leave_basis] = 1
        premultiplication_inv_basis_update_matrix[:, col_in_basis_to_leave_basis] /= search_direction[col_in_basis_to_leave_basis]
        return premultiplication_inv_basis_update_matrix

    def _update_of_inv_basis_matrix(self, premultiplication_inv_basis_update_matrix):      
        # override `_update_of_inv_basis_matrix` from PrimalNaiveSimplexSolver
        self.inv_basis_matrix = premultiplication_inv_basis_update_matrix @ self.inv_basis_matrix

    def _update_update_bfs(self, premultiplication_inv_basis_update_matrix):
        # override `_update_update_bfs` from PrimalNaiveSimplexSolver
        self.bfs = premultiplication_inv_basis_update_matrix @ self.bfs

 
    def solve(self, maxiters: int=100):
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            # # generate reduced costs one at a time taking first improvement as entering col
            # for j in range(self.n):
            #     if j in self.basis:
            #         continue
            #     reduced_cost = self.c[j] - self.c[self.basis] @ self.inv_basis_matrix @ self.A[:, j]
            #     if reduced_cost < 0:
            #        col_in_A_to_enter_basis = j
            #        break
            # else:
            #     # optimal solution found -> break
            #     break

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
            premultiplication_inv_basis_update_matrix = self._calc_premultiplication_inv_basis_update_matrix(col_in_A_to_enter_basis, col_in_basis_to_leave_basis)
            self._update_basis(col_in_basis_to_leave_basis,  col_in_A_to_enter_basis)
            self._update_of_inv_basis_matrix(premultiplication_inv_basis_update_matrix)
            self._update_update_bfs(premultiplication_inv_basis_update_matrix)

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

