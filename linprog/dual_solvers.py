import numpy as np

from linprog.primal_solvers import *
from linprog.utils import *


class DualNaiveSimplexSolver(PrimalNaiveSimplexSolver):
    """Naive Dual Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `PrimalNaiveSimplexSolver`.
    """

    def _dual_get_feasible_direction(self, col_in_basis_to_leave_basis) -> np.array:
        """Get feasible dual direction wrt basic variable leaving the basis.

        Parameters
        ----------
        col_in_A_to_enter_basis : int
            col_in_A_to_enter_basis : int
                Index of column in/wrt `A` to leave `basis`.

        Returns
        -------
        np.array
            Feasible dual direction.

        Raises
        ------
        ValueError
            Value error if problem is unbounded.
        """
        feasible_direction = (
            self.inv_basis_matrix[col_in_basis_to_leave_basis, :] @ self.A
        )
        feasible_direction[self.basis] = 0  # avoid numerical errors
        if self._dual_check_for_unbnoundedness(
            feasible_direction
        ):  # maybe move into calc
            raise ValueError(
                "Feasible direction is non negative. Problem is unbounded."
            )
        return feasible_direction

    def _dual_get_col_in_basis_to_leave_basis(self) -> bool:
        """Returns index of basic variable in basis to leave basis."""
        col_in_basis_to_leave_basis = np.argmax(self.bfs < 0)
        return col_in_basis_to_leave_basis

    def _dual_check_for_optimality(self) -> bool:
        """Solution to the dual problem is optimal if it also satisfies primal problem."""
        return (self.bfs.min() >= 0) or np.isclose(self.bfs.min(), 0)

    def _dual_check_for_unbnoundedness(self, feasible_direction: np.array) -> bool:
        """Problem is unbounded if we can move infintely far in the feasible direction."""
        problem_is_unbounded = feasible_direction.min() >= 0
        return problem_is_unbounded

    def _dual_ratio_test(self, col_in_basis_to_leave_basis: int) -> int:
        """Dual ratio test to see how far we can move in the feasible direction while maintaining dual feasibility.

        Parameters
        ----------
        col_in_basis_to_leave_basis : int
            Index of column in/wrt `basis` to leave `basis`.

        Returns
        -------
        int
            Index of column in/wrt `A` to enter `basis`.
        """
        feasible_direction = self._dual_get_feasible_direction(
            col_in_basis_to_leave_basis
        )
        reduced_costs = self._get_reduced_costs()
        thetas = dual_simplex_div(reduced_costs, feasible_direction)
        col_in_A_to_enter_basis = np.argmin(thetas)
        return col_in_A_to_enter_basis

    def solve(self, maxiters: int = 100):
        """Dual Simplex algorithm loop.

        Parameters
        ----------
        maxiters : int, optional
            maximum number of simplex steps, by default 100

        Returns
        -------
        _type_
            _description_
        """
        self.counter = 0
        while self.counter < maxiters:
            self.counter += 1

            if self._dual_check_for_optimality():
                # optimal solution found break
                self.optimum = True
                break

            col_in_basis_to_leave_basis = self._dual_get_col_in_basis_to_leave_basis()
            col_in_A_to_enter_basis = self._dual_ratio_test(col_in_basis_to_leave_basis)
            self.pivot(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)

        return self._get_solver_return_object()


class DualRevisedSimplexSolver(PrimalRevisedSimplexSolver, DualNaiveSimplexSolver):
    """Revised Dual Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `DualNaiveSimplexSolver` and `PrimalRevisedSimplexSolver`.
    """

    def pivot(self, col_in_basis_to_leave_basis, col_in_A_to_enter_basis):
        # bound pivot method to `PrimalRevisedSimplexSolver`
        PrimalRevisedSimplexSolver.pivot(
            self, col_in_basis_to_leave_basis, col_in_A_to_enter_basis
        )


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
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self._preprocess()
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
