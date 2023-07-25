import numpy as np

from linprog.data_classes import LinProgResult
from linprog.dual_solvers import DualRevisedSimplexSolver
from linprog.exceptions import (DualIsUnboundedError, PrimalIsInfeasibleError,
                                PrimalIsUnboundedError)
from linprog.primal_solvers import PrimalRevisedSimplexSolver
from linprog.utils import get_bounds_on_bfs, primal_simplex_div


class PhaseOneSimplexSolver(PrimalRevisedSimplexSolver):
    def __init__(self, c: np.array, A: np.array, b: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): length n vector cost vector.
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): length m vector defining the equalies constraints.
        """
        (
            self.c,
            self.A,
            self.b,
        ) = self._preprocess_problem(c, A, b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = None

    def solve(self, maxiters: int = 100):
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
                raise PrimalIsInfeasibleError
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

        self.basis = res.basis


class BoundedVariablePrimalSimplexSolver(PrimalRevisedSimplexSolver):
    def __init__(
        self,
        c: np.array,
        A: np.array,
        b: np.array,
        lb: np.array,
        ub: np.array,
        basis: np.array,
        lb_nonbasic_vars: np.array,
        ub_nonbasic_vars: np.array,
    ):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, lb <= x <= ub)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m, n) matrix defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        lb : np.array
            (n,) vector specifying lower bounds on x. -np.inf indicates variable is unbounded below.
        ub : np.array
            (n,) vector specifying lower bounds on x. +np.inf indicates variable is unbounded above.
        basis : np.array
            array of length `m` mapping columns in `A` to their indicies in the basic feasible solution (bfs).
        lb_nonbasic_vars : np.array
            Non basic vairables at their lower bound
        ub_nonbasic_vars : np.array
            Non basic vairables at their upper bound
        """
        (
            self.c,
            self.A,
            self.b,
        ) = self._preprocess_problem(c, A, b)
        self.lb, self.ub = np.array(lb), np.array(ub)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis).astype(int)
        self.lb_nonbasic_vars = np.array(lb_nonbasic_vars).astype(int)
        self.ub_nonbasic_vars = np.array(ub_nonbasic_vars).astype(int)
        # bound unconstrained variables by their largest value (in magnitude) possible
        M_bounds = get_bounds_on_bfs(self.A, self.b)
        self.lb[self.lb == -np.inf] = -M_bounds
        self.ub[self.ub == np.inf] = M_bounds
        # need to set these here instead of calling `_update` mthods for inheritence
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self._update_bfs()
        self.counter = None
        self.optimum = None

    def _calc_current_cost(self):
        """calculate the cost/onjective value of the current basic feasible solution."""
        return (
            self.c[self.basis] @ self.inv_basis_matrix @ self.b
            + (
                self.c[self.ub_nonbasic_vars]
                - self.c[self.basis]
                @ self.inv_basis_matrix
                @ self.A[:, self.ub_nonbasic_vars]
            )
            @ self.ub[self.ub_nonbasic_vars]
            + (
                self.c[self.lb_nonbasic_vars]
                - self.c[self.basis]
                @ self.inv_basis_matrix
                @ self.A[:, self.lb_nonbasic_vars]
            )
            @ self.lb[self.lb_nonbasic_vars]
        )

    def _update_bfs(self):
        """Update current basic feasible solution."""
        self.bfs = self.inv_basis_matrix @ (
            self.b
            - self.A[:, self.lb_nonbasic_vars] @ self.lb[self.lb_nonbasic_vars]
            - self.A[:, self.ub_nonbasic_vars] @ self.ub[self.ub_nonbasic_vars]
        )

    def _get_reduced_costs(self):
        """
        Get the reduced cost vector ie the cost of `x_i` minus the cost of representing `x_i`
        as a linear combination of the current basis.
        """
        reduced_costs = -1 * np.array(self.c)
        # reduced_costs[self.basis] += self.c[self.basis] * self.bfs
        reduced_costs[self.ub_nonbasic_vars] += (
            self.c[self.basis]
            @ self.inv_basis_matrix
            @ self.A[:, self.ub_nonbasic_vars]
        )
        reduced_costs[
            self.ub_nonbasic_vars
        ] *= -1  # we only ever use this multiplied by -1
        reduced_costs[self.lb_nonbasic_vars] += (
            self.c[self.basis]
            @ self.inv_basis_matrix
            @ self.A[:, self.lb_nonbasic_vars]
        )

        reduced_costs[self.basis] = 0  # avoid numerical errors
        return reduced_costs

    def _get_bfs_expanded(self):
        x = np.zeros(self.c.shape)
        x[self.basis] += self.bfs
        x[self.lb_nonbasic_vars] += self.lb[self.lb_nonbasic_vars].T
        x[self.ub_nonbasic_vars] += self.ub[self.ub_nonbasic_vars].T
        return x

    def _primal_get_col_in_A_to_enter_basis(self, reduced_costs: np.array):
        """Returns index of nonbasic variable in A to enter basis."""
        col_in_A_to_enter_basis = np.argmax(reduced_costs)  # just for testing
        return col_in_A_to_enter_basis

    def _primal_check_for_optimality(self, reduced_costs: np.array):
        return (reduced_costs.max() <= 0) or np.isclose(reduced_costs.max(), 0)

    def pivot(self, *args, **kwargs):
        raise NotImplementedError(
            "`pivot` operation/method not decoupled from `solve` method for this class."
        )

    def solve(self, maxiters: int = 100):
        """Override `solve` from `PrimalRevisedSimplexSolver`."""
        self.counter = 0
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1
            col_in_basis_to_leave_basis = None

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                self.optimum = True
                break

            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(
                reduced_costs
            )

            if col_in_A_to_enter_basis in self.lb_nonbasic_vars:
                # case 1: col_in_A_to_enter_basis is in self.lb_nonbasic_vars -> must increase col_in_A_to_enter_basis
                # case 1a: some basic variable drops to its lower bound
                gammas_1 = primal_simplex_div(
                    (self.bfs - self.lb[self.basis]),
                    self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis],
                )
                # case 1b: some basic variable reaches its upper bound
                gammas_2 = primal_simplex_div(
                    (self.ub[self.basis] - self.bfs),
                    -1 * self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis],
                )
                # case 1c. col_in_A_to_enter_basis reaches its upper bound
                gamma_3 = (
                    self.ub[col_in_A_to_enter_basis] - self.lb[col_in_A_to_enter_basis]
                )
                delta = min(gammas_1.min(), gammas_2.min())
                if delta == np.inf:
                    raise PrimalIsUnboundedError
                else:
                    # remove col_in_A_to_enter_basis from nonbasic vars set at lb
                    self.lb_nonbasic_vars = np.delete(
                        self.lb_nonbasic_vars,
                        np.argwhere(self.lb_nonbasic_vars == col_in_A_to_enter_basis),
                    )
                    if gamma_3 <= delta:
                        # col_in_A_to_enter_basis remains nonbasic but at ub
                        self.ub_nonbasic_vars = np.append(
                            self.ub_nonbasic_vars, col_in_A_to_enter_basis
                        )
                    else:
                        # col_in_A_to_enter_basis enters basis
                        if gammas_1.min() < gammas_2.min():
                            # basic variable leaves basis and is set to its lb
                            col_in_basis_to_leave_basis = np.argmin(gammas_1)
                            self.lb_nonbasic_vars = np.append(
                                self.lb_nonbasic_vars,
                                self.basis[col_in_basis_to_leave_basis],
                            )
                        else:
                            # basic variable leaves basis and is set to its ub
                            col_in_basis_to_leave_basis = np.argmin(gammas_2)
                            self.ub_nonbasic_vars = np.append(
                                self.ub_nonbasic_vars,
                                self.basis[col_in_basis_to_leave_basis],
                            )
            elif col_in_A_to_enter_basis in self.ub_nonbasic_vars:
                # case 2: col_in_A_to_enter_basis is in self.ub_nonbasic_vars -> must decrease col_in_A_to_enter_basis
                # case 2a: some basic variable drops to its lower bound
                gammas_1 = primal_simplex_div(
                    (self.bfs - self.lb[self.basis]),
                    -1 * self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis],
                )
                # case 2b: some basic variable reaches its upper bound
                gammas_2 = primal_simplex_div(
                    (self.ub[self.basis] - self.bfs),
                    self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis],
                )
                # case 2c. col_in_A_to_enter_basis drops to its lower bound
                gamma_3 = (
                    self.ub[col_in_A_to_enter_basis] - self.lb[col_in_A_to_enter_basis]
                )
                delta = min(gammas_1.min(), gammas_2.min())
                if delta == np.inf:
                    raise PrimalIsUnboundedError
                else:
                    self.ub_nonbasic_vars = np.delete(
                        self.ub_nonbasic_vars,
                        np.argwhere(self.ub_nonbasic_vars == col_in_A_to_enter_basis),
                    )
                    if gamma_3 <= delta:
                        # col_in_A_to_enter_basis remains nonbasic but at lb
                        self.lb_nonbasic_vars = np.append(
                            self.lb_nonbasic_vars, col_in_A_to_enter_basis
                        )
                    else:
                        # col_in_A_to_enter_basis enters basis
                        if gammas_1.min() < gammas_2.min():
                            # basic variable leaves basis and is set to its lb
                            col_in_basis_to_leave_basis = np.argmin(gammas_1)
                            self.lb_nonbasic_vars = np.append(
                                self.lb_nonbasic_vars,
                                self.basis[col_in_basis_to_leave_basis],
                            )
                        else:
                            # basic variable leaves basis and is set to its ub
                            col_in_basis_to_leave_basis = np.argmin(gammas_2)
                            self.ub_nonbasic_vars = np.append(
                                self.ub_nonbasic_vars,
                                self.basis[col_in_basis_to_leave_basis],
                            )
            else:
                raise ValueError("Column to enter basis is not nonbasic.")

            if col_in_basis_to_leave_basis is not None:
                self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
                premult_inv_basis_update_matrix = (
                    self._calc_premultiplication_inv_basis_update_matrix(
                        col_in_A_to_enter_basis, col_in_basis_to_leave_basis
                    )
                )
                self._update_inv_basis_matrix(premult_inv_basis_update_matrix)
            self._update_bfs()

        return self._get_solver_return_object()


class PrimalDualAlgorithm(DualRevisedSimplexSolver):
    """Primal-Dual Algorithm for Linear Programs."""

    def __init__(self, c: np.array, A: np.array, b: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        """
        (
            self.c,
            self.A,
            self.b,
        ) = self._preprocess_problem(c, A, b)
        self.m, self.n = A.shape
        self.counter = 0
        self.optimum = False

    def solve(self, maxiters1: int = 100, maxiters2: int = 100):
        """Loop implementeing primal-dual algorithm.

        Parameters
        ----------
        maxiters1 : int, optional
            maximum number of times restricted primal is solved, by default 100
        maxiters2 : int, optional
            maximum number of simplex steps in each  instance of the restricted primal, by default 100

        Returns
        -------
        _type_
            _description_
        """
        # zero vector is dual feasible if c >= 0
        bfs_unrestricted_dual = np.zeros(self.m)
        expanded_dual_to_get_initial_bfs = False
        if self.c.min() < 0:
            expanded_dual_to_get_initial_bfs = True
            bounding_M = get_bounds_on_bfs(self.A, self.b)
            # pg. 105 combinatorial optimization - algorithms and complexity
            self.c = np.hstack([self.c, np.zeros(1)])
            self.A = np.vstack(
                [np.hstack([self.A, np.zeros((self.m, 1))]), np.ones((1, self.n + 1))]
            )
            self.b = np.hstack([self.b, self.n * bounding_M * np.ones(1)])
            self.m, self.n = self.A.shape
            bfs_unrestricted_dual = np.hstack(
                [bfs_unrestricted_dual, self.c.min() * np.ones(1)]
            )

        while self.counter < maxiters1:
            self.counter += 1
            # solve restricted primal
            admissiable_set = np.isclose(bfs_unrestricted_dual @ self.A, self.c)
            inadmissable_set = ~admissiable_set

            c_restricted_primal = np.hstack(
                [np.zeros(admissiable_set.sum()), np.ones(self.m)]
            )
            A_restricted_primal = np.hstack(
                [self.A[:, admissiable_set], np.eye(self.m)]
            )
            b_restricted_primal = np.array(self.b)
            basis_restricted_primal = (
                np.arange(self.m) + admissiable_set.sum()
            )  # take artifical vars as basis

            solver_restricted_primal = PrimalRevisedSimplexSolver(
                c_restricted_primal,
                A_restricted_primal,
                b_restricted_primal,
                basis_restricted_primal,
            )
            res_restricted_primal = solver_restricted_primal.solve(maxiters2)

            if res_restricted_primal.cost > 0.0:
                # complementary slackness/primal feasibility to original problem not satisfied/attained
                # modify dual soln so that more variables are able to take non zero values in the restricted primal
                basis_restricted_primal = res_restricted_primal.basis
                bfs_restricted_dual = c_restricted_primal[
                    basis_restricted_primal
                ] @ np.linalg.inv(A_restricted_primal[:, basis_restricted_primal])

                if np.all(bfs_restricted_dual @ A_restricted_primal <= 0):
                    raise DualIsUnboundedError

                theta = np.min(
                    primal_simplex_div(
                        self.c - bfs_unrestricted_dual @ self.A,
                        bfs_restricted_dual @ self.A,
                    )[inadmissable_set]
                )
                bfs_unrestricted_dual += theta * bfs_restricted_dual
            else:
                self.optimum = True
                break  # complementary slackness attained

        bfs_restricted_primal = np.zeros(2 * admissiable_set.sum())
        bfs_restricted_primal[res_restricted_primal.basis] += res_restricted_primal.x[
            res_restricted_primal.basis
        ]

        bfs_unrestricted_primal = np.zeros(self.n)
        bfs_unrestricted_primal[admissiable_set] += bfs_restricted_primal[
            : admissiable_set.sum()
        ]

        basis_unrestricted_primal = np.arange(self.n)[admissiable_set][
            res_restricted_primal.basis[
                res_restricted_primal.basis < admissiable_set.sum()
            ]
        ]
        cost_unrestricted_primal = np.dot(self.c, bfs_unrestricted_primal)

        if expanded_dual_to_get_initial_bfs:
            basis_unrestricted_primal = basis_unrestricted_primal[
                basis_unrestricted_primal != self.n - 1
            ]
            bfs_unrestricted_primal = bfs_unrestricted_primal[:-1]

        res = LinProgResult(
            x=bfs_unrestricted_primal,
            basis=basis_unrestricted_primal,
            cost=cost_unrestricted_primal,
            iters=self.counter,
            optimum=self.optimum,
        )

        return res
