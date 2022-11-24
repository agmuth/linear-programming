import numpy as np


safe_div0 = np.vectorize(
    # x/0 = sign(x) * inf, x != 0
    # 0/0 = inf
    lambda n, d: n / d if d != 0 else (np.sign(n) if n != 0 else 1) * np.inf
)



class RevisedSimplexSolver():
    """
        Revised Simplex algorithm that implements Bland's selection rule to avoid cycling. 
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
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis)
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = self.inv_basis_matrix @ self.b  # basic feasible soln

    def solve(self, maxiters: int=100):
        for _ in range(maxiters):
            reduced_costs = self.c - self.c[self.basis] @ self.inv_basis_matrix @ self.A
            reduced_costs[self.basis] = 0  # numerical errors
            if reduced_costs.min() >= 0:
                # optimal solution found break
                break
            
            
            col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
            search_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] 

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

            # update `self.inv_basis_matrix` w\o having to invert 
            self.inv_basis_matrix = np.hstack([self.inv_basis_matrix, np.expand_dims(search_direction, 1)])
            self.inv_basis_matrix[col_in_basis_to_leave_basis, :] /= search_direction[col_in_basis_to_leave_basis]
            for i in range(self.m):
                if i == col_in_basis_to_leave_basis:
                    continue
                self.inv_basis_matrix[i, :] -= search_direction[i] * self.inv_basis_matrix[col_in_basis_to_leave_basis, :] 
            self.inv_basis_matrix = self.inv_basis_matrix[:, :-1]



class Tableau():
    """
        Tableau class for carrying out tableau operations of simplex algorithm that implements Bland's selection rule to avoid cycling. 
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

        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.copy(basis)
        
        inv_basis_matrix = np.linalg.inv(A[:, self.basis])
        

        self.tableau = np.zeros((self.m+1, self.n+1))
        self.tableau[1:, 0] = inv_basis_matrix @ b
        self.tableau[1:, 1:] = inv_basis_matrix @ A

        self.tableau[0, :] = np.hstack(
            [
                -1 * c[self.basis] @ self.tableau[1:, 0] , 
                c - c[self.basis] @ self.tableau[1:, 1:]
            ]
        )
        
 
    def pivot(self, pivot_row: int, pivot_col: int):
        self.basis[pivot_row-1] = pivot_col-1
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]
        for i in range(self.tableau.shape[0]):
            if i == pivot_row:
                continue
            self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]



class TableauSimplexSolver():
    """
        Tableau based Simplex algorithm.  
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
        self.tableau = Tableau(c, A, b, basis)
    
    def solve(self, maxiters=100):
        for _ in range(maxiters):
            self.tableau.tableau[0, 1:][self.tableau.basis] = 0  # avoid numerical errors
            if self.tableau.tableau[0, 1:].min() >= 0:  # 0^th row is reduced costs
                # optimal solution found break
                break

            pivot_col = np.argmax(self.tableau.tableau[0, 1:] < 0) + 1

            if self.tableau.tableau[1:, pivot_col].max() <= 0:  # check to make sure search direction has at least one positive entry
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("reduced cost vector has all non-positive entries. problem is unbounded.")

            pivot_row = np.argmin(
                [
                    xi if xi > 0 else np.inf for xi in
                    safe_div0(
                        self.tableau.tableau[1:, 0], 
                        self.tableau.tableau[1:, pivot_col]  # search direction
                    )
                ]
        
            ) \
            + 1  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]


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
                if (self.artificial_tableau.tableau.tableau[index_in_basis+1, 1:self.n] == 0).all():
                    # `index_in_basis`^th constraint is redundant -> drop
                    self.artificial_tableau.tableau.tableau = np.delete(self.artificial_tableau.tableau.tableau, index_in_basis+1, 0)
                    self.artificial_tableau.basis = np.delete(self.artificial_tableau.basis, index_in_basis, 0)
                    self.artificial_tableau.bfs = np.delete(self.artificial_tableau.bfs, index_in_basis, 0)

                else:
                    # need to pivot element out of basis
                    pivot_col = np.argmax(self.artificial_tableau.tableau.tableau[index_in_basis+1, 1:self.n] > 0) + 1
                    self.artificial_tableau.tableau.pivot(i, pivot_col)

        # PHASE II --------------------------------------
        self.tableau = TableauSimplexSolver(
            c = self.c,
            A = self.artificial_tableau.tableau.tableau[1:, 1:(self.n+1)],
            b = self.artificial_tableau.tableau.tableau[1:, 0], 
            basis = self.artificial_tableau.basis
        )

        self.tableau.solve(maxiters)
        self.basis = self.tableau.basis
        self.bfs = self.tableau.bfs


       

if __name__ == "__main__":
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    A = np.array(
        [
            [1, 0, 0, 3, 2, 1, 0, 0],
            [0, 1, 0, 5, 1, 1, 1, 0], 
            [0, 0, 1, 2, 5, 1, 0, 1]
        ]
    )
    b = np.array([1, 3, 4])
    basis_seq = np.array(
        [
           [0, 1, 2],  # starting
           [3, 1, 2],
           [4, 1, 2],
           [4, 6, 2], 
           [4, 6, 7],
        ]
    )

    solver = TableauSimplexSolver(c, A, b, basis_seq[0])
    # solver = TwoPhaseSimplexSolver(c, A, b)
    res = []
    for _, basis in enumerate(basis_seq[1:]):
        solver.solve(maxiters=1)
        res.append(np.array_equal(basis, solver.basis))
    solver.solve(maxiters=1) # check to make sure algorithm has terminated
    res.append(np.array_equal(basis_seq[-1], solver.basis))
    print(res)