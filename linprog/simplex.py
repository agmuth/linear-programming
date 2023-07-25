from typing import Optional

import numpy as np

from linprog.primal_solvers import *


class SimplexSolver(PrimalRevisedSimplexSolver):
    def __init__(
        self,
        c: np.array,
        A: Optional[np.array] = None,
        b: Optional[np.array] = None,
        G: Optional[np.array] = None,
        h: Optional[np.array] = None,
        lb: Optional[np.array] = None,
        ub: Optional[np.array] = None,
    ):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, Gx <= h, lb <= x <= ub)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m1, n) matrix defining linear combinations subject to equality constraints.
        b : np.array
            (m1,) vector defining the equality constraints.
        G : np.array
            (m2, n) matrix defining linear combinations subject to less than or equal constraints.
        h : np.array
            (m2,) vector defining the less than or equal constraints.
        lb : np.array
            (n,) vector specifying lower bounds on x. -np.inf indicates variable is unbounded below.
        ub : np.array
            (n,) vector specifying lower bounds on x. +np.inf indicates variable is unbounded above.
        """
        A_and_b = A is not None and b is not None
        G_and_h = G is not None and h is not None
        self.num_slack_vars = 0 if G is None else G.shape[0]
        self.num_vars = 0 if A is None else A.shape[1]
        self.num_vars = 0 if G is None else G.shape[1]
        self.num_vars += self.num_slack_vars

        if A_and_b and not G_and_h:
            _A = np.array(A)
            _b = np.array(b)
        elif not A_and_b and G_and_h:
            _A = np.hstack([np.array(G), np.eye(self.num_slack_vars)])
            _b = np.array(h)
        elif A_and_b and G_and_h:
            _A = np.vstack(
                [
                    np.hstack([np.array(A), np.zeros(A.shape[0], self.num_slack_vars)]),
                    np.hstack([np.array(G), np.eye(self.num_slack_vars)]),
                ]
            )
            _b = np.concatenate([np.array(b), np.array(h)])
        else:
            raise ValueError("Polyhedral misspcified.")
        _c = np.concatenate([np.array(c), np.zeros(self.num_slack_vars)])

        (self.c, self.A, self.b) = self._preprocess_problem(_c, _A, _b)

        if lb is None:
            lb = np.repeat(-np.inf, self.A.shape[1] - self.num_slack_vars)
        self.lb = np.concatenate([np.array(lb), np.repeat(0, self.num_slack_vars)])

        if ub is None:
            ub = np.repeat(np.inf, self.A.shape[1] - self.num_slack_vars)
        self.ub = np.concatenate([np.array(ub), np.repeat(np.inf, self.num_slack_vars)])

    def solve(self, maxiters1: int = 100, maxiters2: int = 100):
        # phase I to get bfs to general problem

        # add lb/ub surplus/slack vars to A
        lb_surplus_index = ~np.isclose(self.lb, 0.0)
        ub_slack_index = ~np.isclose(self.ub, np.inf)
        if lb_surplus_index.any() or ub_slack_index.any():
            offset = self.A.shape[1]
            zeros = np.zeros(
                (
                    self.A.shape[0],
                    offset * (int(lb_surplus_index.any()) + int(ub_slack_index.any())),
                )
            )
            A = np.hstack([self.A, zeros])
            for i, b in enumerate(lb_surplus_index):
                if not b:
                    continue
                # add constraint of form `x_i - s_i = lb_i` to A
                A = np.vstack([A, np.zeros((1, A.shape[1]))])
                A[-1, i] += 1
                A[-1, offset + i] -= 1
            for i, b in enumerate(ub_slack_index):
                if not b:
                    continue
                # add constraint of form `x_i + s_i = ub_i` to A
                A = np.vstack([A, np.zeros((1, A.shape[1]))])
                A[-1, i] += 1
                A[-1, 2 * offset + i] += 1
        # vars are now all 0 <= x < inf
        # absorb lower/upper bounds into `b`
        b = np.concatenate([self.b, self.lb[lb_surplus_index], self.ub[ub_slack_index]])
        # `c` doesn't matter - were passing to phase I algo which will zero out `c` and add artificial vars anyways
        c = np.zeros(A.shape[1])
        basis = TwoPhaseRevisedSimplexSolver(c, A, b).solve(
            maxiters1=maxiters1, return_phase_1_basis=True
        )
        bfs = np.zeros(A.shape[1])
        bfs[basis] = np.linalg.inv(A[:, basis]) @ b
        bfs = bfs[: self.num_vars]

        # `bfs` satifies Ax=b defined above and is a bfs for original problem defined with variable bounds outside of `A`
        # need to seperate out basic vars and non basic vars for bounded simplex algorithm

        vars = np.arange(self.num_vars)
        lb_nonbasic_vars = vars[np.isclose(bfs - self.lb, 0.0)]
        ub_nonbasic_vars = vars[np.isclose(bfs - self.ub, 0.0)]
        basic_vars = vars[
            ~np.isclose(bfs - self.lb, 0.0) * ~np.isclose(bfs - self.ub, 0.0)
        ]

        # chck to make sure we have enough basic vars
        while (len(basic_vars) < self.A.shape[0]) and (len(ub_nonbasic_vars) > 0):
            basic_vars = np.append(basic_vars, ub_nonbasic_vars[-1])
            ub_nonbasic_vars = ub_nonbasic_vars[:-1]

        while (len(basic_vars) < self.A.shape[0]) and (len(lb_nonbasic_vars) > 0):
            basic_vars = np.append(basic_vars, lb_nonbasic_vars[-1])
            lb_nonbasic_vars = lb_nonbasic_vars[:-1]

        res = BoundedVariablePrimalSimplexSolver(
            c=self.c,
            A=self.A,
            b=self.b,
            lb=self.lb,
            ub=self.ub,
            basis=basic_vars,
            lb_nonbasic_vars=lb_nonbasic_vars,
            ub_nonbasic_vars=ub_nonbasic_vars,
        ).solve(maxiters=maxiters2)
        res.x = res.x[: self.num_vars - self.num_slack_vars]  # remove slack vars
        res.basis = None  # basis is uninterpretable without slack vars
        return res
