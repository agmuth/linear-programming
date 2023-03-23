import numpy as np 
from linearprogramming.primal_simplex_solvers import *

class TwoPhaseSimplexSolver():
    """
        Two Phase Simplex algorithm that implements Bland's selection rule to avoid cycling. 
    """
    def __init__(self, c: np.array, A: np.array, b: np.array) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
        """
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.artificial_tableau = None
        self.tableau = None


    def drive_artificial_variables_out_of_basis(self, maxiters=100):
        # PHASE I --------------------------------------
        self.artificial_tableau = PrimalTableauSimplexSolver(
            c = np.hstack([np.zeros(self.n), np.ones(self.m)]),
            A = np.hstack([self.A, np.eye(self.m)]),
            b = self.b, 
            basis = np.arange(self.n, self.n+self.m)
        )

        self.artificial_tableau.solve(maxiters)
        if self.artificial_tableau.tableau.tableau[0, 0] < 0:
            raise ValueError("Problem does not have any feasible solutions.")

        # drive any remaining aritificial variables out of basis
        for i in range(self.m+self.n-1, self.n, -1):
            if i in self.artificial_tableau.basis:
                index_in_basis = np.argmax(self.artificial_tableau.basis == i)
                if (self.artificial_tableau.tableau.tableau[index_in_basis+1, 1:self.n] == 0).all():
                    # `index_in_basis`^th constraint is redundant -> drop
                    self.artificial_tableau.tableau.tableau = np.delete(self.artificial_tableau.tableau.tableau, index_in_basis+1, 0)
                    self.artificial_tableau.basis = np.delete(self.artificial_tableau.basis, index_in_basis, 0)
                    self.artificial_tableau.bfs = np.delete(self.artificial_tableau.bfs, index_in_basis, 0)

                else:
                    # need to pivot element out of basis
                    pivot_col = np.argmax(self.artificial_tableau.tableau.tableau[index_in_basis+1, 1:self.n] > 0) + 1
                    self.artificial_tableau.tableau.pivot(i, pivot_col)

    
    
    def solve(self, maxiters=100):
        self.drive_artificial_variables_out_of_basis(maxiters)
        # PHASE II --------------------------------------
        self.tableau = PrimalTableauSimplexSolver(
            c = self.c,
            A = self.artificial_tableau.tableau.tableau[1:, 1:(self.n+1)],
            b = self.artificial_tableau.tableau.tableau[1:, 0], 
            basis = self.artificial_tableau.basis
        )

        res = self.tableau.solve(maxiters)
        self.basis = self.tableau.basis
        self.bfs = self.tableau.bfs
        return res

if __name__ == "__main__":
    pass