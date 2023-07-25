import numpy as np

from linprog.utils import *


class Tableau:
    """Tableau class for carrying out tableau operations of simplex algorithm that implements Bland's selection rule to avoid cycling."""

    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array) -> None:
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
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.copy(basis)

        inv_basis_matrix = np.linalg.inv(A[:, self.basis])

        self.tableau = np.zeros((self.m + 1, self.n + 1))
        self.tableau[1:, 0] = inv_basis_matrix @ b
        self.tableau[1:, 1:] = inv_basis_matrix @ A

        self.tableau[0, :] = np.hstack(
            [
                -1 * c[self.basis] @ self.tableau[1:, 0],
                c - c[self.basis] @ self.tableau[1:, 1:],
            ]
        )

    def pivot(self, pivot_row: int, pivot_col: int):
        """Perform pivot operation on tableau.

        Parameters
        ----------
        pivot_row : int
            Row in tableau to pivot on.
        pivot_col : int
            Col in tableau to pivot on.
        """
        self.basis[pivot_row - 1] = pivot_col - 1  # update basis
        # now preform elementary row operations to set column pivot_col of tableau to e_{pivot_row}
        self.tableau[pivot_row, :] /= self.tableau[
            pivot_row, pivot_col
        ]  # set self.tableau[pivot_row, pivot_col] to 1
        # now need to
        for i in range(self.tableau.shape[0]):
            if i == pivot_row:  # nothing to do entry is already 1
                continue
            self.tableau[i, :] -= (
                self.tableau[i, pivot_col] * self.tableau[pivot_row, :]
            )  # zero out self.tableau[i, pivot_col]


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


class DualTableauSimplexSolver:
    """Tableau implementation of Dual Simplex Algorithm."""

    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array) -> None:
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Parameters
        ----------
        c : np.array
            (1, n) cost vector
        A : np.array
            (m, n) matirx defining linear combinations subject to equality constraints.
        b : np.array
            (m, 1) vector defining the equality constraints.
        basis : np.array
            array of length `m` mapping columns in `A` to their indicies in the basic feasible solution (bfs).
        """
        self.tableau = Tableau(c, A, b, basis)

    def solve(self, maxiters=100):
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
        self.optimum = False
        while self.counter < maxiters:
            self.counter += 1
            self.tableau.tableau[0, 1:][
                self.tableau.basis
            ] = 0  # avoid numerical errors
            if (self.tableau.tableau[1:, 0].min() >= 0) or np.isclose(
                self.tableau.tableau[1:, 0].min(), 0
            ):  # check for termination condition
                self.optimum = True
                break

            pivot_row = np.argmax(self.tableau.tableau[1:, 0] < 0) + 1

            if self.tableau.tableau[pivot_row, 1:].min() >= 0:
                raise ValueError(
                    "`pivot_row` entries are all non negative. problem is unbounded."
                )

            self.tableau.tableau[pivot_row, 1:] < 0

            pivot_col = (
                np.argmin(
                    [
                        r if v < 0 else np.inf
                        for v, r in zip(
                            self.tableau.tableau[pivot_row, 1:],
                            primal_simplex_div(
                                self.tableau.tableau[0, 1:],
                                np.abs(self.tableau.tableau[pivot_row, 1:]),
                            ),
                        )
                    ]
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


class TwoPhaseTableauSimplexSolver:
    """
    Two Phase Simplex algorithm that implements Bland's selection rule to avoid cycling.
    """

    def __init__(self, c: np.array, A: np.array, b: np.array):
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

        res = self.artificial_tableau.solve(maxiters)
        if res.cost > 0:
            if res.optimum:
                raise ValueError("Problem is unfeasible.")
            else:
                raise ValueError("Phase one did not converge.")

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

    def solve(self, maxiters=100, return_phase_1_basis: bool = False):
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
