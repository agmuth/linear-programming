import numpy as np

from linprog.primal_solvers import *
from linprog.utils import get_bounds_on_bfs
from typing import Optional
from scipy.linalg import block_diag



class TwoPhaseRevisedSimplexSolver(PrimalRevisedSimplexSolver):
    def __init__(self, c: np.array, A: np.array, b: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): length n vector cost vector.
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): length m vector defining the equalies constraints.
        """
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self._preprocess()
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1

    def _phase_one(self, maxiters: int = 100):
        """Phase I of two phase simplex method.

        Parameters
        ----------
        maxiters : int, optional
            _description_, by default 100

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        c = np.hstack([np.zeros(self.n), np.ones(self.m)])
        A = np.hstack([self.A, np.eye(self.m)])
        b = np.array(self.b)
        basis = np.arange(self.n, self.n + self.m)
        solver = PrimalRevisedSimplexSolver(c, A, b, basis)
        res = solver.solve(maxiters)

        if res.cost > 0:
            if res.optimum:
                raise ValueError("Problem is unfeasible.")
            else:
                raise ValueError("Phase one did not converge.")

        if res.basis.max() >= self.n:
            # artificial variables still in basis at zero level
            # pivot out when/where possible
            for i in range(self.n, self.n + self.m):
                if i in res.basis:
                    non_basic_non_artifical_vars = np.array(
                        [i for i in range(self.n) if i not in res.basis]
                    )
                    potential_pivots = (solver.inv_basis_matrix @ solver.A)[
                        np.argmax(res.basis == i), non_basic_non_artifical_vars
                    ] > 0
                    if potential_pivots.any():
                        col_in_A_to_enter_basis = non_basic_non_artifical_vars[
                            np.argmax(potential_pivots)
                        ]
                        col_in_basis_to_leave_basis = np.where(res.basis == i)[0][0]
                        solver.pivot(
                            col_in_basis_to_leave_basis, col_in_A_to_enter_basis
                        )
                        res = solver._get_solver_return_object()

        if res.basis.max() >= self.n:
            # unable to pivot out one or more artifical vairables from basis
            # constraints redundent -> remove
            non_redundent_equs_idx = res.basis <= self.n
            self.A = self.A[non_redundent_equs_idx]
            self.b = self.b[non_redundent_equs_idx]
            self.m = non_redundent_equs_idx.sum()
            res.basis = res.basis[non_redundent_equs_idx]

        return res.basis

    def solve(
        self,
        maxiters1: int = 100,
        maxiters2: int = 100,
        return_phase_1_basis: bool = False,
    ):
        """Implement two phase simplex solver.

        Parameters
        ----------
        maxiters1 : int, optional
            Maximum number of iters in phase one, by default 100
        maxiters2 : int, optional
           Maximum number of iters in phase two, by default 100
        return_phase_1_basis : bool, optional
            Return basis from phase one no phase two, by default False

        Returns
        -------
        _type_
            _description_
        """
        phase_2_starting_basis = self._phase_one(maxiters1)
        if return_phase_1_basis:
            return phase_2_starting_basis
        # phase 2
        solver = PrimalRevisedSimplexSolver(
            self.c, self.A, self.b, phase_2_starting_basis
        )
        res = solver.solve(maxiters2)
        return res



class SimplexSolver(PrimalRevisedSimplexSolver):
    def __init__(
        self,
        c: np.array,
        A: Optional[np.array]=None,
        b: Optional[np.array]=None,
        G: Optional[np.array]=None,
        h: Optional[np.array]=None,
        lb: Optional[np.array]=None,
        ub: Optional[np.array]=None,
    ):
        A_and_b = A is not None and b is not None
        G_and_h = G is not None and h is not None
        self.num_slack_vars = 0 if G is None else G.shape[0]
        self.num_vars = 0 if A is None else A.shape[1]
        self.num_vars = 0 if G is None else G.shape[1]
        self.num_vars += self.num_slack_vars
        
        if A_and_b and not G_and_h:
            self.A = np.array(A)
            self.b = np.array(b)
        elif not A_and_b and G_and_h:
            self.A = np.hstack([np.array(G), np.eye(self.num_slack_vars)])
            self.b = np.array(h)
        elif A_and_b and G_and_h:
            self.A = np.vstack(
                [
                    np.hstack([np.array(A), np.zeros(A.shape[0], self.num_slack_vars)]),
                    np.hstack([np.array(G), np.eye(self.num_slack_vars)]),
                    
                ]
            )
            self.b = np.concatenate([np.array(b), np.array(h)])
        else:
            raise ValueError("Polyhedral misspcified.")
        self.c = np.concatenate([np.array(c), np.zeros(self.num_slack_vars)])
        
        self._preprocess()
        
        if lb is None:
            lb = np.repeat(-np.inf, self.A.shape[1]-self.num_slack_vars)
        self.lb = np.concatenate([np.array(lb), np.repeat(0, self.num_slack_vars)])
        
        if ub is None:
            ub = np.repeat(np.inf, self.A.shape[1]-self.num_slack_vars)
        self.ub = np.concatenate([np.array(ub), np.repeat(np.inf, self.num_slack_vars)])
        
        # M_bounds = get_bounds_on_bfs(self.A, self.b)
        # self.lb[self.lb == -np.inf] = -M_bounds
        # self.ub[self.ub == np.inf] = M_bounds
        
        
    def solve(self, maxiters1: int=100, maxiters2: int=100):
        
        
        # phase I to get bfs to general problem
        
        # add lb/ub surplus/slack vars to A
        lb_surplus_index = ~np.isclose(self.lb, 0.0)
        ub_slack_index = ~np.isclose(self.ub, np.inf)
        if lb_surplus_index.any() or ub_slack_index.any():
            offset = self.A.shape[1]
            zeros = np.zeros((self.A.shape[0], offset*(int(lb_surplus_index.any()) + int(ub_slack_index.any()))))
            A = np.hstack([self.A, zeros])
            for i, b in enumerate(lb_surplus_index):
                if not b: continue
                A = np.vstack([A, np.zeros((1, A.shape[1]))])
                A[-1, i] += 1
                A[-1, offset+i] -= 1
            for i, b in enumerate(ub_slack_index):
                if not b: continue
                A = np.vstack([A, np.zeros((1, A.shape[1]))])
                A[-1, i] += 1
                A[-1, 2*offset+i] += 1
        # vars are now all 0 <= x < inf
                
        b = np.concatenate([self.b, self.lb[lb_surplus_index], self.ub[ub_slack_index]])
        c = np.zeros(A.shape[1])
        basis = TwoPhaseRevisedSimplexSolver(c, A, b).solve(maxiters1=maxiters1, return_phase_1_basis=True)
        bfs = np.zeros(A.shape[1])
        bfs[basis] = np.linalg.inv(A[:, basis]) @ b
        bfs = bfs[:self.num_vars]
            
        vars = np.arange(self.num_vars)
        lb_nonbasic_vars = vars[np.isclose(bfs, self.lb)]
        ub_nonbasic_vars = vars[np.isclose(bfs, self.ub)]
        basic_vars = vars[~np.isclose(bfs, self.lb) * ~np.isclose(bfs, self.ub)]
    
        while (len(basic_vars) < self.A.shape[0]) and (len(ub_nonbasic_vars) > 0):
            np.append(basic_vars, ub_nonbasic_vars[-1])
            ub_nonbasic_vars = ub_nonbasic_vars[:-1]
            
        while (len(basic_vars) < self.A.shape[0]) and (len(lb_nonbasic_vars) > 0):
            np.append(basic_vars, lb_nonbasic_vars[-1])
            lb_nonbasic_vars = lb_nonbasic_vars[:-1]
                
        
        solver = BoundedVariablePrimalSimplexSolver(
            c=self.c,
            A=self.A,
            b=self.b,
            lb=self.lb,
            ub=self.ub,
            basis=basic_vars,
            lb_nonbasic_vars=lb_nonbasic_vars,
            ub_nonbasic_vars=ub_nonbasic_vars,
        )
        
        res = solver.solve(maxiters=maxiters2)
        res.x = res.x[:self.num_vars-self.num_slack_vars]
        res.basis = None
        return res
 

if __name__ == "__main__":
    c = np.array([-2, -4, -1])
    G = np.array([[2, 1, 1], [1, 1, -1]])
    h = np.array([10, 4])
    lb = np.array([0, 0, 1])
    ub = np.array([4, 6, 4])
    
    solver = SimplexSolver(c=c, G=G, h=h, lb=lb, ub=ub,)
    
    res = solver.solve()
    
    print(res)
    print(G @ res.x <= h)
    print(res.x >= lb)
    print(res.x <= ub)
    print(np.dot(res.x, c))
    
    pass