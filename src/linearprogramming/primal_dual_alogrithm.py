import numpy as np 
from linearprogramming.primal_simplex_solvers import *
from linearprogramming.utils import primal_simplex_div

class PrimalDualAlgorithm():
    """
        Primal Dual algorithm 
    """
    def __init__(self, c: np.array, A: np.array, b: np.array) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)
        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
        """
        # assume c is nongeative
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.counter = 0

    def solve(self):

        bfs_unrestricted_dual = np.zeros(A.shape[0])  # can always do this if c >= 0

        while True:
            admissiable_set = np.isclose(bfs_unrestricted_dual @ self.A,  self.c)
            inadmissable_set = ~admissiable_set

            c_restricted_primal = np.hstack([np.zeros(admissiable_set.sum()), np.ones(self.m)])
            A_restricted_primal = np.hstack([self.A[:, admissiable_set==1], np.eye(self.m)])
            b_restricted_primal = np.array(self.b)
            basis_restricted_primal = np.arange(self.m) + admissiable_set.sum()  # take artifical vars as basis

            solver_restricted_primal = PrimalRevisedSimplexSolver(c_restricted_primal, A_restricted_primal, b_restricted_primal, basis_restricted_primal)
            res_restricted_primal = solver_restricted_primal.solve()

            if res_restricted_primal["cost"] > 0.0:
                basis_restricted_primal = res_restricted_primal["basis"]
                bfs_restricted_dual = c_restricted_primal[basis_restricted_primal] @ np.linalg.inv(A_restricted_primal[:, basis_restricted_primal])

                theta = np.min(primal_simplex_div(c - bfs_unrestricted_dual @ A, bfs_restricted_dual @ A)[inadmissable_set])
                bfs_unrestricted_dual += theta * bfs_restricted_dual

            else:
                break

        bfs_unrestricted_primal = np.zeros(self.n)
        bfs_unrestricted_primal[admissiable_set] += res_restricted_primal['x'][res_restricted_primal['basis']]


        return bfs_unrestricted_primal



if __name__ == "__main__":
    # example 6.8 pg 272 linear programming and network flows
    c = np.array([3, 4, 6, 7, 5, 0, 0])
    A = np.array(
        [
            [2, -1, 1, 6, -5, -1, 0],
            [1, 1, 2, 1, 2, 0, -1],
        ]
    )
    b = np.array([6, 3])

    # pg 96 linear and nonlinear programming
    c = np.array([2, 1, 4])
    A = np.array(
        [
            [1, 1, 2],
            [2, 1, 3],
        ]
    )
    b = np.array([3, 5])

    solver = PrimalDualAlgorithm(c, A, b)
    bfs_unrestricted_primal = solver.solve()
    print(bfs_unrestricted_primal)