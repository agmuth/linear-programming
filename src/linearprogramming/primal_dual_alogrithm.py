import numpy as np 
from linearprogramming.primal_simplex_solvers import *
from linearprogramming.utils import primal_simplex_div
from math import factorial

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
        self.counter = 0

    def solve(self):
        bfs_unrestricted_dual = np.zeros(self.m) # can always do this if c >= 0
        expanded_dual_to_get_initial_bfs = False
        if self.c.min() < 0:
            expanded_dual_to_get_initial_bfs = True
            # lemma 2.1 combinatorial optimization - algorithms and complexity
            alpha = np.abs(self.A).max()
            beta = np.abs(self.b).max()
            M = factorial(self.m) * alpha**(self.m-1) * beta
            
            # pg. 105 combinatorial optimization - algorithms and complexity 
            self.c = np.hstack([self.c, np.zeros(1)])
            self.A = np.vstack([np.hstack([self.A, np.zeros((self.m, 1))]), np.ones((1, self.n+1))])
            self.b = np.hstack([self.b, self.n*M*np.ones(1)])
            self.m, self.n = self.A.shape
            bfs_unrestricted_dual = np.hstack([bfs_unrestricted_dual, self.c.min()*np.ones(1)])

        while True:
            admissiable_set = np.isclose(bfs_unrestricted_dual @ self.A,  self.c)
            inadmissable_set = ~admissiable_set

            c_restricted_primal = np.hstack([np.zeros(admissiable_set.sum()), np.ones(self.m)])
            A_restricted_primal = np.hstack([self.A[:, admissiable_set], np.eye(self.m)])
            b_restricted_primal = np.array(self.b)
            basis_restricted_primal = np.arange(self.m) + admissiable_set.sum()  # take artifical vars as basis

            solver_restricted_primal = PrimalRevisedSimplexSolver(c_restricted_primal, A_restricted_primal, b_restricted_primal, basis_restricted_primal)
            res_restricted_primal = solver_restricted_primal.solve()

            if res_restricted_primal["cost"] > 0.0:
                basis_restricted_primal = res_restricted_primal["basis"]
                bfs_restricted_dual = c_restricted_primal[basis_restricted_primal] @ np.linalg.inv(A_restricted_primal[:, basis_restricted_primal])

                theta = np.min(primal_simplex_div(self.c - bfs_unrestricted_dual @ self.A, bfs_restricted_dual @ self.A)[inadmissable_set])
                bfs_unrestricted_dual += theta * bfs_restricted_dual

            else:
                break

        
        bfs_restricted_primal = np.zeros(2*admissiable_set.sum())
        bfs_restricted_primal[res_restricted_primal['basis']] += res_restricted_primal['x']

        bfs_unrestricted_primal = np.zeros(self.n)
        bfs_unrestricted_primal[admissiable_set] += bfs_restricted_primal[:admissiable_set.sum()]

        basis_unrestricted_primal = np.arange(self.n)[admissiable_set][res_restricted_primal['basis'][res_restricted_primal['basis'] < admissiable_set.sum()]]
        cost_unrestricted_primal = np.dot(self.c, bfs_unrestricted_primal)

        if expanded_dual_to_get_initial_bfs:
            basis_unrestricted_primal = basis_unrestricted_primal[basis_unrestricted_primal != self.n-1]
            bfs_unrestricted_primal = bfs_unrestricted_primal[:-1]

        res = {"bfs": bfs_unrestricted_primal, "basis": basis_unrestricted_primal, "cost": cost_unrestricted_primal, "iters": -1}


        return res



if __name__ == "__main__":
    # # example 6.8 pg 272 linear programming and network flows
    # c = np.array([3, 4, 6, 7, 5, 0, 0])
    # A = np.array(
    #     [
    #         [2, -1, 1, 6, -5, -1, 0],
    #         [1, 1, 2, 1, 2, 0, -1],
    #     ]
    # )
    # b = np.array([6, 3])

    # # pg 96 linear and nonlinear programming
    # c = np.array([2, 1, 4])
    # A = np.array(
    #     [
    #         [1, 1, 2],
    #         [2, 1, 3],
    #     ]
    # )
    # b = np.array([3, 5])

    # ex 6.10 pg. 279 linear programming and network flows soln = (6, 0, 10)
    c = np.array([-2, 1, -1, 0, 0])
    A = np.array(
        [
            [1, 1, 1, 1, 0], 
            [-1, 2, 0, 0, 1]
        ]
    )
    b = np.array([6, 4])

    solver = PrimalDualAlgorithm(c, A, b)
    bfs_unrestricted_primal = solver.solve()
    print(bfs_unrestricted_primal)