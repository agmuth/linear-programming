import numpy as np

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
        self.basis[pivot_row-1] = pivot_col-1  # update basis
        # now preform elementary row operations to set column pivot_col of tableau to e_{pivot_row}
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]  # set self.tableau[pivot_row, pivot_col] to 1
        # now need to 
        for i in range(self.tableau.shape[0]):
            if i == pivot_row:  # nothing to do entry is already 1
                continue
            self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]  # zero out self.tableau[i, pivot_col]

