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
        self.bfs = np.linalg.inv(self.A[:, self.basis]) @ self.b  # basic feasible soln
        inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])

        for _ in range(maxiters):
            
            reduced_costs = self.c - self.c[self.basis] @ inv_basis_matrix @ self.A

            if reduced_costs.min() >= 0:
                # optimal solution found break
                break

            col_in_A_to_enter_basis = np.argmin(reduced_costs)
            search_direction = inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] # need to find better name

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

        return self.bfs


class RevisedSimplexSolver(NaiveSimplexSolver):
    def solve(self, maxiters: int=100):
        self.bfs = np.linalg.inv(self.A[:, self.basis]) @ self.b  # basic feasible soln
        inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])

        for _ in range(maxiters):
            
            reduced_costs = self.c - self.c[self.basis] @ inv_basis_matrix @ self.A

            if reduced_costs.min() >= 0:
                # optimal solution found break
                break

            col_in_A_to_enter_basis = np.argmin(reduced_costs)
            search_direction = inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] # need to find better name

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

        return self.bfs


