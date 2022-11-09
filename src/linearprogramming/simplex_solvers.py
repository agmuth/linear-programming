import numpy as np


safe_div0 = np.vectorize(
    # x/0 = sign(x) * inf, x != 0
    # 0/0 = inf
    lambda n, d: n / d if d != 0 else (np.sign(n) if n != 0 else 1) * np.inf
)


class NaiveSimplexSolver():
    """
        Naive Simplex algorithm that implements Bland's selection rule to avoid cycling. 
    """
    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)
        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
        """
        self.c, self.A, self.b = c, A, b
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = basis
    
        
    def solve(self, maxiters: int=100):
        inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = inv_basis_matrix @ self.b  # basic feasible soln

        for _ in range(maxiters):
            
            reduced_costs = self.c - self.c[self.basis] @ inv_basis_matrix @ self.A
            reduced_costs = reduced_costs.round(4)

            if reduced_costs.min() >= 0:
                # optimal solution found break
                break

            col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
            search_direction = inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] 

            if search_direction.max() <= 0:
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("reduced cost vector has all non-positive entries. problem is unbounded.")

            thetas = safe_div0(self.bfs, search_direction)
            thetas[search_direction <= 0] = np.inf
            col_in_basis_to_leave_basis = np.argmin(thetas)
            theta_star = thetas[col_in_basis_to_leave_basis]
            self.bfs -= theta_star * search_direction
            self.bfs[col_in_basis_to_leave_basis] = theta_star
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])

class RevisedSimplexSolver(NaiveSimplexSolver):
    def solve(self, maxiters: int=100):
        inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = inv_basis_matrix @ self.b  # basic feasible soln

        for _ in range(maxiters):
            
            reduced_costs = self.c - self.c[self.basis] @ inv_basis_matrix @ self.A

            if reduced_costs.min() >= 0:
                # optimal solution found break
                break

            col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
            search_direction = inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] 

            if search_direction.max() <= 0:
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("reduced cost vector has all non-positive entries. problem is unbounded.")

            thetas = safe_div0(self.bfs, search_direction)
            thetas[search_direction <= 0] = np.inf
            col_in_basis_to_leave_basis = np.argmin(thetas)
            theta_star = thetas[col_in_basis_to_leave_basis]
            self.bfs -= theta_star * search_direction
            self.bfs[col_in_basis_to_leave_basis] = theta_star
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis

            # update `inv_basis_matrix` w\o having to invert 
            inv_basis_matrix = np.hstack([inv_basis_matrix, np.expand_dims(search_direction, 1)])
            inv_basis_matrix[col_in_basis_to_leave_basis, :] /= search_direction[col_in_basis_to_leave_basis]
            for i in range(self.m):
                if i == col_in_basis_to_leave_basis:
                    continue
                inv_basis_matrix[i, :] -= search_direction[i] * inv_basis_matrix[col_in_basis_to_leave_basis, :] 
            inv_basis_matrix = inv_basis_matrix[:, :-1]


class TableauSimplexSolver(NaiveSimplexSolver):
    """
        Tableau implementation Simplex algorithm that implements Bland's selection rule to avoid cycling. 
    """
    def __init__(self, c: np.array, A: np.array, b: np.array, basis: np.array) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): 1, n vector cost vector. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            basis (np.array): array of length m mapping the columns of A to their indicies in the bfs 
        """

        super().__init__(c, A, b, basis)
        inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = inv_basis_matrix @ self.b  # basic feasible soln

        self.tableau = np.zeros((self.m+1, self.n+1))
        self.tableau[0, :] = np.hstack(
            [
                -1 * self.c[self.basis] @ self.bfs.T, 
                self.c - self.c[self.basis] @ inv_basis_matrix @ self.A
            ]
        )
        self.tableau[1:, 0] = inv_basis_matrix @ self.b
        self.tableau[1:, 1:] = inv_basis_matrix @ self.A


    def get_col_in_tableau_to_enter_basis(self):
        return np.argmax(self.tableau[0, 1:] < 0) + 1  # bland's rule


    def get_col_in_basis_to_leave_basis_and_theta_star(self, col_in_tableau_to_enter_basis: int):
        search_direction = self.tableau[1:, col_in_tableau_to_enter_basis]
        thetas = safe_div0(self.bfs, search_direction)
        thetas[search_direction <= 0] = np.inf
        col_in_basis_to_leave_basis = np.argmin(thetas)  # bland's rule
        theta_star = thetas[col_in_basis_to_leave_basis]
        return col_in_basis_to_leave_basis, theta_star


    def update_bfs_and_basis(self, col_in_basis_to_leave_basis: int, col_in_tableau_to_enter_basis: int, theta_star: float):
        self.bfs -= theta_star * self.tableau[1:, col_in_tableau_to_enter_basis] 
        self.bfs[col_in_basis_to_leave_basis] = theta_star
        self.basis[col_in_basis_to_leave_basis] = col_in_tableau_to_enter_basis - 1


    def pivot_tableau(self, col_in_basis_to_leave_basis: int, col_in_tableau_to_enter_basis: int):
        self.tableau[col_in_basis_to_leave_basis+1, :] /= self.tableau[1:, col_in_tableau_to_enter_basis][col_in_basis_to_leave_basis] 
        for i in range(self.tableau.shape[0]):
            if i == col_in_basis_to_leave_basis+1:
                continue
            self.tableau[i, :] -= self.tableau[i, col_in_tableau_to_enter_basis] * self.tableau[col_in_basis_to_leave_basis+1, :]


    def solve(self, maxiters: int=100):
        for _ in range(maxiters):
            if self.tableau[0, 1:].min() >= 0:  # 0^th row is reduced costs
                # optimal solution found break
                break
            
            col_in_tableau_to_enter_basis = self.get_col_in_tableau_to_enter_basis()

            if self.tableau[1:, col_in_tableau_to_enter_basis].max() <= 0:  # check to make sure search direction has at least one positive entry
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("reduced cost vector has all non-positive entries. problem is unbounded.")

            col_in_basis_to_leave_basis, theta_star = self.get_col_in_basis_to_leave_basis_and_theta_star(col_in_tableau_to_enter_basis)
            self.update_bfs_and_basis(col_in_basis_to_leave_basis, col_in_tableau_to_enter_basis, theta_star)
            self.pivot_tableau(col_in_basis_to_leave_basis, col_in_tableau_to_enter_basis)


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
        self.c, self.A, self.b = c, A, b
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1


    def solve(self, maxiters=100):

        # PHASE I --------------------------------------
        self.artificial_tableau = TableauSimplexSolver(
            c = np.hstack([np.zeros(self.n), np.ones(self.m)]),
            A = np.hstack([self.A, np.eye(self.m)]),
            b = self.b, 
            basis = np.arange(self.n, self.n+self.m)
        )

        self.artificial_tableau.solve(maxiters)

        # drive any remaining aritificial variables out of basis
        for i in range(self.m+self.n-1, self.n, -1):
            if i in self.artificial_tableau.basis:
                index_in_basis = np.argmax(self.artificial_tableau.basis == i)
                if (self.artificial_tableau.tableau[index_in_basis+1, 1:self.n] == 0).all():
                    # `index_in_basis`^th constraint is redundant -> drop
                    self.artificial_tableau.tableau = np.delete(self.artificial_tableau.tableau, index_in_basis+1, 0)
                    self.artificial_tableau.basis = np.delete(self.artificial_tableau.basis, index_in_basis, 0)
                    self.artificial_tableau.bfs = np.delete(self.artificial_tableau.bfs, index_in_basis, 0)

                else:
                    # need to pivot element out of basis
                    col_in_tableau_to_enter_basis = np.argmax(self.artificial_tableau.tableau[index_in_basis+1, 1:self.n] > 0) + 1
                    theta_star = self.artificial_tableau.tableau[index_in_basis+1, 0] / self.artificial_tableau.tableau[index_in_basis+1, col_in_tableau_to_enter_basis]
                    self.artificial_tableau.pivot_tableau(
                        col_in_basis_to_leave_basis=index_in_basis,
                        col_in_tableau_to_enter_basis=col_in_tableau_to_enter_basis,
                        theta_star=theta_star
                    )

        # PHASE II --------------------------------------
        self.tableau = TableauSimplexSolver(
            c = self.c,
            A = self.artificial_tableau.tableau[1:, 1:(self.n+1)],
            b = self.artificial_tableau.tableau[1:, 0], 
            basis = self.artificial_tableau.basis
        )

        self.tableau.solve(maxiters)
        print()

       