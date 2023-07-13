import numpy as np

from linprog.primal_simplex_solvers import *


class TwoPhaseTableauSimplexSolver:
    """
    Two Phase Simplex algorithm that implements Bland's selection rule to avoid cycling.
    """

    def __init__(self, c: np.array, A: np.array, b: np.array) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): length n vector cost vector.
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): length m vector defining the equalies constraints.
        """
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.artificial_tableau = None
        self.tableau = None

    def _drive_artificial_variables_out_of_basis(self, maxiters=100):
        # PHASE I --------------------------------------
        self.artificial_tableau = PrimalTableauSimplexSolver(
            c=np.hstack([np.zeros(self.n), np.ones(self.m)]),
            A=np.hstack([self.A, np.eye(self.m)]),
            b=self.b,
            basis=np.arange(self.n, self.n + self.m),
        )

        self.artificial_tableau.solve(maxiters)
        if self.artificial_tableau.tableau.tableau[0, 0] < 0:
            raise ValueError("Problem does not have any feasible solutions.")

        # drive any remaining aritificial variables out of basis
        for i in range(self.m + self.n - 1, self.n, -1):
            if i in self.artificial_tableau.basis:
                index_in_basis = np.argmax(self.artificial_tableau.basis == i)
                if (
                    self.artificial_tableau.tableau.tableau[
                        index_in_basis + 1, 1 : self.n
                    ]
                    == 0
                ).all():
                    # `index_in_basis`^th constraint is redundant -> drop
                    self.artificial_tableau.tableau.tableau = np.delete(
                        self.artificial_tableau.tableau.tableau, index_in_basis + 1, 0
                    )
                    self.artificial_tableau.basis = np.delete(
                        self.artificial_tableau.basis, index_in_basis, 0
                    )
                    self.artificial_tableau.bfs = np.delete(
                        self.artificial_tableau.bfs, index_in_basis, 0
                    )

                else:
                    # need to pivot element out of basis
                    pivot_col = (
                        np.argmax(
                            self.artificial_tableau.tableau.tableau[
                                index_in_basis + 1, 1 : self.n
                            ]
                            > 0
                        )
                        + 1
                    )
                    self.artificial_tableau.tableau.pivot(i, pivot_col)

    def solve(self, maxiters=100, return_phase_1_basis: bool=False):
        self._drive_artificial_variables_out_of_basis(maxiters)
        if return_phase_1_basis:
            return self.artificial_tableau.basis
        # PHASE II --------------------------------------
        self.tableau = PrimalTableauSimplexSolver(
            c=self.c,
            A=self.artificial_tableau.tableau.tableau[1:, 1 : (self.n + 1)],
            b=self.artificial_tableau.tableau.tableau[1:, 0],
            basis=self.artificial_tableau.basis,
        )

        return self.tableau.solve(maxiters)
    
    
class TwoPhaseRevisedSimplexSolver():
    def __init__(self, c: np.array, A: np.array, b: np.array) -> None:
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1        

    def _phase_one(self, maxiters: int=100):
        c = np.hstack([np.zeros(self.n), np.ones(self.m)])
        A = np.hstack([self.A, np.eye(self.m)])
        b = np.array(self.b)
        basis = np.arange(self.n, self.n+self.m)
        solver = PrimalRevisedSimplexSolver(c,A, b, basis)
        res = solver.solve(maxiters)
        
        if res.cost > 0:
            raise ValueError("Problem is unfeasible.")
        
        if res.basis.max() >= self.n:
            #a rtificial vars in basis
            # remove 0 level artifical variables by pivoting
            for i in range(self.n, self.n+self.m):
                if i in res.basis:
                    non_basic_non_artifical_vars = np.array([i for i in range(self.n) if i not in res.basis])
                    potential_pivots = (solver.inv_basis_matrix @ solver.A)[np.argmax(res.basis == i), non_basic_non_artifical_vars] > 0
                    if potential_pivots.any():
                        col_in_A_to_enter_basis = non_basic_non_artifical_vars[np.argmax(potential_pivots)]
                        col_in_basis_to_leave_basis = np.where(res.basis == i)[0][0]
                        solver.pivot(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
                        res = solver._get_solver_return_object()
            
        if res.basis.max() >= self.n:
            non_redundent_equs_idx = (res.basis <= self.n)
            self.A = self.A[non_redundent_equs_idx]
            self.b = self.b[non_redundent_equs_idx]
            self.m = non_redundent_equs_idx.sum()
            res.basis = res.basis[non_redundent_equs_idx]
                        
        return res.basis
    
    def solve(self, maxiters1: int=100, maxiters2: int=100, return_phase_1_basis:bool=False):
        phase_2_starting_basis = self._phase_one(maxiters1)
        if return_phase_1_basis:
            return phase_2_starting_basis
        # phase 2
        solver = PrimalRevisedSimplexSolver(self.c, self.A, self.b, phase_2_starting_basis)
        res = solver.solve(maxiters2)
        return res
    

