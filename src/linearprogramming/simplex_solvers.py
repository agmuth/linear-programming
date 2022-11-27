import numpy as np

primal_simplex_div = np.vectorize(
    lambda n, d: n / d if d > 0 else np.inf
)

dual_simplex_div = np.vectorize(
    lambda n, d: -1 * n / d if d < 0 else np.inf
)



class PrimalNaiveSimplexSolver():
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
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.basis = np.array(basis)
        self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
        self.bfs = self.inv_basis_matrix @ self.b

    def _get_reduced_costs(self):
        reduced_costs = self.c - self.c[self.basis] @ self.inv_basis_matrix @ self.A
        reduced_costs[self.basis] = 0  #  avoid numerical errors
        return reduced_costs

    
    def _get_search_direction(self, col_in_A_to_enter_basis):
        search_direction = self.inv_basis_matrix @ self.A[:, col_in_A_to_enter_basis] 
        return search_direction


    def solve(self, maxiters: int=100):
        counter = 0
        while counter < maxiters:
            counter += 1

            reduced_costs = self._get_reduced_costs()
            if reduced_costs.min() >= 0:
                # optimal solution found break
                break
            
            col_in_A_to_enter_basis = np.argmax(reduced_costs < 0)
            search_direction = self._get_search_direction(col_in_A_to_enter_basis)

            if search_direction.max() <= 0:
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("Problem is unbounded.")

            thetas = primal_simplex_div(self.bfs, search_direction)
            col_in_basis_to_leave_basis = np.argmin(thetas)
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
            self.bfs = self.inv_basis_matrix @ self.b

        return {"x": self.bfs, "basis": self.basis, "cost": np.dot(self.c[self.basis], self.bfs), "iters": counter}



class DualNaiveSimplexSolver(PrimalNaiveSimplexSolver):
    def _get_search_direction(self, col_in_basis_to_leave_basis):
        search_direction = self.inv_basis_matrix[col_in_basis_to_leave_basis, :] @ self.A
        search_direction[self.basis] = 0  # avoid numerical errors
        return search_direction

    def solve(self, maxiters: int=100):
        counter = 0
        while counter < maxiters:
            counter += 1

            if self.bfs.min() >= 0:
                # optimal solution found break
                break

            col_in_basis_to_leave_basis = np.argmax(self.bfs < 0)
            search_direction = self._get_search_direction(col_in_basis_to_leave_basis)

            if search_direction.min() >= 0:
                raise ValueError("Problem is unbounded.")

            reduced_costs = self._get_reduced_costs()
            
            thetas = dual_simplex_div(reduced_costs, search_direction)
            col_in_A_to_enter_basis = np.argmin(thetas)
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            self.inv_basis_matrix = np.linalg.inv(self.A[:, self.basis])
            self.bfs = self.inv_basis_matrix @ self.b            

        return {"x": self.bfs, "basis": self.basis, "cost": np.dot(self.c[self.basis], self.bfs), "iters": counter}



class PrimalRevisedSimplexSolver():
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
        self.bfs = self.inv_basis_matrix @ self.b
        

    def _revised_update_of_inv_basis_matrix(self, search_direction, col_in_basis_to_leave_basis):
        self.inv_basis_matrix = np.hstack([self.inv_basis_matrix, np.expand_dims(search_direction, 1)])
        self.inv_basis_matrix[col_in_basis_to_leave_basis, :] /= search_direction[col_in_basis_to_leave_basis]
        for i in range(self.m):
            if i == col_in_basis_to_leave_basis:
                continue
            self.inv_basis_matrix[i, :] -= search_direction[i] * self.inv_basis_matrix[col_in_basis_to_leave_basis, :] 
        self.inv_basis_matrix = self.inv_basis_matrix[:, :-1]

   
    def solve(self, maxiters: int=100):
        counter = 0
        while counter < maxiters:
            counter += 1

            col_in_A_to_enter_basis = None
            search_direction = None

            for i in range(self.n):
                # generate reduced cost one at a time to save on computation/storage (probably not actually faster with numpy)
                if i in self.basis:
                    continue
                search_direction = self.inv_basis_matrix @ self.A[:, i]
                # check reduced cost
                if self.c[i] - self.c[self.basis] @ search_direction < 0:  # take first negative entry - bland's selection rule
                    col_in_A_to_enter_basis = i
                    break
            else:
                # reduced costs are all non-negative -> optimal soln found -> break
                break

            if search_direction.max() <= 0:
                # optimal cost is -inf -> problem is unbounded
                raise ValueError("Problem is unbounded.")

            thetas = primal_simplex_div(self.bfs, search_direction)
            
            # update bfs w\o matrix mult - again probably not any better as oppsed to numpy
            col_in_basis_to_leave_basis = np.argmin(thetas)
            theta_star = thetas[col_in_basis_to_leave_basis]
            self.bfs -= theta_star * search_direction
            self.bfs[col_in_basis_to_leave_basis] = theta_star
            self.basis[col_in_basis_to_leave_basis] = col_in_A_to_enter_basis
            
            # update basis and `self.inv_basis_matrix` w\o having to invert - again probably not any better as opposed to numpy
            self._revised_update_of_inv_basis_matrix(search_direction, col_in_basis_to_leave_basis)

        return {"x": self.bfs, "basis": self.basis, "cost": np.dot(self.c[self.basis], self.bfs), "iters": counter}


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



class PrimalTableauSimplexSolver():
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
        counter = 0
        while counter < maxiters:
            counter += 1
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
                    primal_simplex_div(
                        self.tableau.tableau[1:, 0], 
                        self.tableau.tableau[1:, pivot_col]  # search direction
                    )
                ]
        
            ) \
            + 1  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]
        return {"x": self.bfs, "basis": self.basis, "cost": self.tableau.tableau[0, 0], "iters": counter}



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
        self.c, self.A, self.b = np.array(c), np.array(A), np.array(b)
        self.m, self.n = A.shape
        self.row_offset = 1
        self.col_offset = 1
        self.artificial_tableau = None
        self.tableau = None


    def drive_artificial_variables_out_of_basis(self, maxiters=100):
        # PHASE I --------------------------------------
        self.artificial_tableau = TableauSimplexSolver(
            c = np.hstack([np.zeros(self.n), np.ones(self.m)]),
            A = np.hstack([self.A, np.eye(self.m)]),
            b = self.b, 
            basis = np.arange(self.n, self.n+self.m)
        )

        self.artificial_tableau.solve(maxiters)
        if self.artificial_tableau.tableau.tableau[0, 0] < 0:
            raise ValueError("Problem does not have any feasible solutions.")

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

    
    
    def solve(self, maxiters=100):
        self.drive_artificial_variables_out_of_basis(maxiters)
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



class DualTableauSimplexSolver():
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
        counter = 0
        while counter < maxiters:
            counter += 1
            self.tableau.tableau[0, 1:][self.tableau.basis] = 0  # avoid numerical errors
            if self.tableau.tableau[1:, 0].min() >= 0:  # check for termination condition
                break

            pivot_row = np.argmax(self.tableau.tableau[1:, 0] < 0) + 1

            if self.tableau.tableau[pivot_row, 1:].min() >= 0:
                raise ValueError("`pivot_row` entries are all non negative. problem is unbounded.")


            self.tableau.tableau[pivot_row, 1:] < 0

            pivot_col = np.argmin(
                [
                    r if v < 0 else np.inf for v, r in zip(
                        self.tableau.tableau[pivot_row, 1:],
                        primal_simplex_div(
                            self.tableau.tableau[0, 1:], 
                            np.abs(self.tableau.tableau[pivot_row, 1:])
                        )
                    )
                    
                ]
        
            ) \
            + 1  # bland's rule

            self.tableau.pivot(pivot_row, pivot_col)

        self.basis = self.tableau.basis
        self.bfs = self.tableau.tableau[1:, 0]
        return {"x": self.bfs, "basis": self.basis, "cost": self.tableau.tableau[0, 0], "iters": counter}


       

if __name__ == "__main__":
    pass