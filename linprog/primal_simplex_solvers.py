import numpy as np

from linprog.tableau import Tableau
from linprog.utils import *


class PrimalNaiveSimplexSolver:
    """Naive Primal Simplex algorithm that implements Bland's selection rule to avoid cycling."""

    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        basis : np.array
            array of length `m` mapping columns in `A` to their indicies in the basic feasible solution (bfs).
        """
        self.c, self.A, self.b = (
            np.array(c.flatten()),
            np.array(A),
            np.array(b.flatten()),
        )
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis).astype(int)
        # need to set these here instead of calling `_update` mthods for inheritence
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = self.inv_basis_matrix @ self.b
        self.counter = None
        self.optimum = None

    def _get_reduced_costs(self):
        """
        Get the reduced cost vector ie the cost of `x_i` minus the cost of representing `x_i`
        as a linear combination of the current basis.
        """
        reduced_costs = self.c - self.c[self.basis] @ self.inv_basis_matrix @ self.A
        reduced_costs[self.basis] = 0  # avoid numerical errors
        return reduced_costs

    def _update_bfs(self):
        """Update current basic feasible solution."""
        self.bfs = self.inv_basis_matrix @ self.b

    def _get_bfs_expanded(self):
        x = np.zeros(self.n)
        x[self.basis] = self.bfs
        return x

    def _calc_current_cost(self):
        """calculate the cost/onjective value of the current basic feasible solution."""
        return np.dot(self.c, self._get_bfs_expanded())

    def _get_solver_return_object(self):
        """Build return object from calling `self.solve`."""
        res = LinProgResult(
            x=self._get_bfs_expanded(),
            basis=self.basis,
            cost=self._calc_current_cost(),
            iters=self.counter,
            optimum=self.optimum,
        )
        return res

    def _update_inv_basis_matrix(self):
        """Naively update inverse basis matrix by inverting subset of columns in `A`."""
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])

    def _update_basis(
        self, col_in_basis_to_leave_basis: int, col_in_A_to_enter_basis: int
    ):
        """Update basis corresponding to current basic feasible solution.

        Parameters
        ----------
        col_in_basis_to_leave_basis : int
            Index of column in/wrt `basis` to leave `basis`.
        col_in_A_to_enter_basis : int
            Index of column in/wrt `A` to enter `basis`.
        """
        self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis

    def _primal_get_feasible_direction(self, col_in_A_to_enter_basis: int) -> np.array:
        """Get feasible primal direction wrt non basic variable entering the basis.

        Parameters
        ----------
        col_in_A_to_enter_basis : int
            col_in_A_to_enter_basis : int
                Index of column in/wrt `A` to enter `basis`.

        Returns
        -------
        np.array
            Feasible primal direction.

        Raises
        ------
        ValueError
            Value error if problem is unbounded.
        """
        feasible_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis]
        if self._primal_check_for_unbnoundedness(feasible_direction):
            # optimal cost is -inf -> problem is unbounded
            raise ValueError(
                "Reduced cost vector has all non-positive entries. Rroblem is unbounded."
            )
        return feasible_direction

    def _primal_get_col_in_A_to_enter_basis(self, reduced_costs: np.array):
        """Returns index of nonbasic variable in A to enter basis."""
        col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
        return col_in_A_to_enter_basis

    def _primal_check_for_optimality(self, reduced_costs) -> bool:
        """Current basic feasible solution is optimal if reduced ccosts are all non negative."""
        return (reduced_costs.min() >= 0) or np.isclose(reduced_costs.min(), 0)

    def _primal_check_for_unbnoundedness(self, feasible_direction: np.array) -> bool:
        """Problem is unbounded if we can move infintely far in the feasible direction."""
        problem_is_unbounded = feasible_direction.max() <= 0
        return problem_is_unbounded

    def _primal_ratio_test(self, col_in_A_to_enter_basis: int) -> int:
        """Primal ratio test to see how far we can move in the feasible direction while maintaining primal feasibility.

        Parameters
        ----------
        col_in_A_to_enter_basis : int
            Index of column in/wrt `A` to enter `basis`.

        Returns
        -------
        int
            Index of column in/wrt `basis` to leave `basis`.
        """
        feasible_direction = self._primal_get_feasible_direction(
            col_in_A_to_enter_basis
        )
        thetas = primal_simplex_div(self.bfs, feasible_direction)
        col_in_basis_to_leave_basis = np.argmin(thetas)
        return col_in_basis_to_leave_basis

    def solve(self, maxiters: int = 100):
        """Primal Simplex algorithm loop.

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
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                self.optimum = True
                break

            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(
                reduced_costs
            )
            col_in_basis_to_leave_basis = self._primal_ratio_test(
                col_in_A_to_enter_basis
            )

            self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
            self._update_inv_basis_matrix()
            self._update_bfs()

        return self._get_solver_return_object()


class PrimalRevisedSimplexSolver(PrimalNaiveSimplexSolver):
    """Revised Primal Simplex algorithm that implements Bland's selection rule to avoid cycling.
    Inherits from `PrimalNaiveSimplexSolver`.
    """

    def _calc_premultiplication_inv_basis_update_matrix(
        self, col_in_A_to_enter_basis, col_in_basis_to_leave_basis
    ) -> np.array:
        """Calculate matrix to premultiply `self.inv_basis_matrix` by corresponding to a basis change.

        Parameters
        ----------
        col_in_A_to_enter_basis : _type_
            Index of column in/wrt `A` to enter `basis`.
        col_in_basis_to_leave_basis : _type_
            Index of column in/wrt `basis` to leave `basis`.

        Returns
        -------
        np.array
            premultiplication matrix
        """
        feasible_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis]
        premult_inv_basis_update_matrix = np.eye(self.m)
        premult_inv_basis_update_matrix[
            :, col_in_basis_to_leave_basis
        ] = -feasible_direction
        premult_inv_basis_update_matrix[
            col_in_basis_to_leave_basis, col_in_basis_to_leave_basis
        ] = 1
        premult_inv_basis_update_matrix[
            :, col_in_basis_to_leave_basis
        ] /= feasible_direction[col_in_basis_to_leave_basis]
        return premult_inv_basis_update_matrix

    def _update_inv_basis_matrix(self, premult_inv_basis_update_matrix):
        """Override `_update_inv_basis_matrix` from `PrimalNaiveSimplexSolver`."""
        self.inv_basis_matrix = premult_inv_basis_update_matrix @ self.inv_basis_matrix

    def _update_bfs(self, premult_inv_basis_update_matrix):
        """Override `_update_bfs` from `PrimalNaiveSimplexSolver`."""
        self.bfs = premult_inv_basis_update_matrix @ self.bfs

    def solve(self, maxiters: int = 100):
        """Override `solve` from `PrimalNaiveSimplexSolver`."""
        self.counter = 0
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1

            reduced_costs = self._get_reduced_costs()
            if self._primal_check_for_optimality(reduced_costs):
                # optimal solution found break
                self.optimum = True
                break

            col_in_A_to_enter_basis = self._primal_get_col_in_A_to_enter_basis(
                reduced_costs
            )
            col_in_basis_to_leave_basis = self._primal_ratio_test(
                col_in_A_to_enter_basis
            )

            premult_inv_basis_update_matrix = (
                self._calc_premultiplication_inv_basis_update_matrix(
                    col_in_A_to_enter_basis, col_in_basis_to_leave_basis
                )
            )

            self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
            self._update_inv_basis_matrix(premult_inv_basis_update_matrix)
            self._update_bfs(premult_inv_basis_update_matrix)

        return self._get_solver_return_object()


class PrimalTableauSimplexSolver:
    """Tableau implementation of Primal Simplex Algorithm."""

    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Parameters
        ----------
        c : np.array
            (n, ) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        basis : np.array
            array of length `m` mapping columns in `A` to their indicies in the basic feasible solution (bfs).
        """

        self.tableau = Tableau(c, A, b, basis)

    def solve(self, maxiters=100):
        """Primal Simplex algorithm loop.

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
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1
            self.tableau.tableau[0, 1:][
                self.tableau.basis
            ] = 0  # avoid numerical errors
            if (self.tableau.tableau[0, 1:].min() >= 0) or np.isclose(
                self.tableau.tableau[0, 1:].min(), 0
            ):  # 0^th row is reduced costs
                # optimal solution found break
                self.optimum = True
                break

            pivot_col = np.argmax(self.tableau.tableau[0, 1:] < 0) + 1

            if (
                self.tableau.tableau[1:, pivot_col].max() <= 0
            ):  # check to make sure search direction has at least one positive entry
                # optimal cost is -inf -> problem is unbounded
                raise ValueError(
                    "Reduced cost vector has all non-positive entries. Rroblem is unbounded."
                )

            pivot_row = (
                np.argmin(
                    primal_simplex_div(
                        self.tableau.tableau[1:, 0],
                        self.tableau.tableau[1:, pivot_col],  # feasible direction
                    )
                )
                + 1
            )  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]
        x_soln = np.zeros(self.tableau.n)
        x_soln[self.basis] = self.bfs
        return LinProgResult(
            x=x_soln,
            basis=self.basis,
            cost=self.tableau.tableau[0, 0],
            iters=self.counter,
            optimum=self.optimum,
        )


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
        self.c, self.A, self.b = (
            np.array(c).astype(np.float32),
            np.array(A).astype(np.float32),
            np.array(b).astype(np.float32),
        )
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

    def solve(self, maxiters: int = 100):
        """Override `solve` from `PrimalRevisedSimplexSolver`."""
        self.counter = 0
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1

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
                    raise ValueError("Problem is unbounded.")
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
                    raise ValueError("Problem is unbounded.")
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

            self._update_basis(col_in_basis_to_leave_basis, col_in_A_to_enter_basis)
            premult_inv_basis_update_matrix = (
                self._calc_premultiplication_inv_basis_update_matrix(
                    col_in_A_to_enter_basis, col_in_basis_to_leave_basis
                )
            )
            self._update_inv_basis_matrix(premult_inv_basis_update_matrix)
            self._update_bfs()

        return self._get_solver_return_object()


if __name__ == "__main__":
    c = np.array([-2, -4, -1, 0, 0])
    b = np.array([10, 4])
    A = np.array([[2, 1, 1, 1, 0], [1, 1, -1, -0, 1]])
    lb = np.array([0, 0, 1, 0, 0])
    ub = np.array([4, 6, 4, np.inf, np.inf])
    basis = np.array([3, 4])
    lb_nonbasic_vars = np.array([0, 1, 2])
    ub_nonbasic_vars = np.array([])

    solver = BoundedVariablePrimalSimplexSolver(
        c,
        A,
        b,
        lb,
        ub,
        basis,
        lb_nonbasic_vars,
        ub_nonbasic_vars,
    )

    res = solver.solve()
    res
