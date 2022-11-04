import numpy as np
from typing import Tuple

class Tableau():
    """
    Tableau that implements Bland's selection rule to avoid cycling. 
    """
    def __init__(self, c: np.array, A: np.array, b: np.array, bfs: np.array, basis: np.array, cost_index: int=0) -> None:
        """
        Assumes LP is passed in in standard form (min c'x sbj. Ax = b, x >= 0)

        Args:
            c (np.array): marix of arbitrary number of rows by n columns. Used to define one or more cost vectors. 
            A (np.array): m by n matrix defining the linear combinations to be subject to equality constraints.
            b (np.array): m by 1 vector defining the equalies constraints.
            bfs (np.array): array of length n definining a basic feasible solution
            basis (np.array): array of length n non zero entries define the ording of columns in the basis corresponding to `bfs`
            cost_index (int, optional): which row of `c` to use as the cost. Defaults to 0.
        """

        self.m, self.n = A.shape
        self.row_offset = c.shape[0]
        self.col_offset = 1
        self.cost_index = cost_index

        # calculate relative costs of each column as well as cost of current bfs
        c_bar = np.hstack(
            [
                -1 * np.expand_dims(c @ bfs.T, 1), 
                c - c[:, basis > 0] @ np.linalg.inv(A[:, basis > 0]) @ A
            ]
        )

        # fill out tableau
        self.tableau = np.zeros((self.m+self.row_offset, self.n+self.col_offset))
        self.tableau[:self.row_offset, :] = c_bar
        self.tableau[self.row_offset:, [0]] = b
        self.tableau[self.row_offset:, self.col_offset:] = A
        self.tableau = self.tableau.astype(float)
        
        self.bfs = bfs.astype(float) 
        self.basis = basis.astype(int)

    
    def _get_preset_in_basis_index(self) -> np.array:
        # returns boolean index of whether or not a variable is in the current basis
        return self.basis > 0

    
    def _get_basis_ordering(self) -> np.array:
        # returns an integr index to rearrange self.basis[self.get_present_in_basis_index()] so that the i^{th} column of the resulting matrix is the variable corresponding to the i^{th} column in the basis
        present_in_basis_idx = self._get_preset_in_basis_index()
        column_order_in_basis = self.basis[present_in_basis_idx].astype(int) - 1
        basis_columns_idx = np.empty_like(column_order_in_basis)
        basis_columns_idx[column_order_in_basis] = np.arange(len(column_order_in_basis))
        return basis_columns_idx

      
    def calc_col_to_enter_basis(self) -> int:
        """
        choose nonbasis column to enter basis by first favoured column lexographicallly (Bland's selection rule)

        Returns:
            int: index of column to enter the basis (indexing done relative to `A`)
        """

        col_to_enter_basis = np.argmin(self.tableau[self.cost_index, self.col_offset:] >= 0)
        return col_to_enter_basis
    
    
    def calc_col_to_leave_basis_and_theta(self, col_to_enter_basis: int) -> Tuple[float, int]:
        """
        get column to leave basis and scalar (theta) to adjust bfs by in event of tie choose lowest number column to exit basis (Bland)

        Args:
            col_to_enter_basis (int): index of col to enter basis (indexing done relative to `A`) returned by `calc_col_to_enter_basis`

        Returns:
            Tuple[int, float]: returns the index of the column to leave basis (indexing done relative to `A`) and the amount to move in the direction of the column entering the basis
        """

        present_in_basis_idx, = self._get_preset_in_basis_index()
        basis_columns_idx = self._get_basis_ordering()
        thetas = self.bfs[present_in_basis_idx][basis_columns_idx] / self.tableau[self.row_offset:, self.col_offset+col_to_enter_basis]
        thetas[self.tableau[self.row_offset:, self.col_offset+col_to_enter_basis] < 0] = np.inf
        thetas[np.isnan(thetas)] = np.inf
        theta = np.min(thetas)    
        col_to_leave_basis = np.argmax(self.basis == (np.argmin(thetas) + 1))    
        return col_to_leave_basis, theta

    
    def _update_bfs_and_basis_after_pivot(self, col_to_leave_basis: int, col_to_enter_basis: int, theta: float) -> None:
        """
        update bfs and basis after pivot

        Args:
            col_to_leave_basis (int): index of column to leave the basis (indexing done relative to `A`)
            col_to_enter_basis (int): index of column to enter the basis (indexing done relative to `A`)
            theta (float): the amount to move in the direction of the column entering the basis
        """

        present_in_basis_idx = self._get_preset_in_basis_index()
        basis_columns_idx = self._get_basis_ordering()
        tableau_reordering_idx = np.empty_like(basis_columns_idx)
        tableau_reordering_idx[basis_columns_idx] = np.arange(len(basis_columns_idx))

        self.bfs[present_in_basis_idx] -= theta * self.tableau[self.row_offset:, self.col_offset+col_to_enter_basis][tableau_reordering_idx]  # move bfs theta in direction of entering column
        self.bfs[col_to_enter_basis] = theta  # new variable enters at level `theta`

        # update basis
        self.basis[col_to_enter_basis] = self.basis[col_to_leave_basis] 
        self.basis[col_to_leave_basis] = 0

    
    def pivot(self, col_to_enter_basis: int, col_to_leave_basis: int, theta: float) -> None:
        """
        pivot table based on col to enter and leave and then update bfs and basis

        Args:
            col_to_leave_basis (int): index of column to leave the basis (indexing done relative to `A`)
            col_to_enter_basis (int): index of column to enter the basis (indexing done relative to `A`)
            theta (float): the amount to move in the direction of the column entering the basis
        """
        
        # indexing here is done relative to `self.tableau`
        row_to_pivot_on = int(self.basis[col_to_leave_basis] -1 + self.row_offset)
        col_to_pivot_on = int(col_to_enter_basis + self.col_offset)

        self._update_bfs_and_basis_after_pivot(col_to_leave_basis, col_to_enter_basis, theta)
        
        self.tableau[row_to_pivot_on] /= self.tableau[row_to_pivot_on, col_to_pivot_on]  # the (row_to_pivot_on, col_to_pivot_on) entry is one after this operation
        for i in range(self.m+self.row_offset):
            if i == row_to_pivot_on: 
                continue
            # subtract multiple from all other rows to end up with an e_{self.basis[col_to_leave_basis] -1} vector in `self.tableau[self.row_offset:, col_to_pivot_on]`
            self.tableau[i] -= self.tableau[row_to_pivot_on] * self.tableau[i, col_to_pivot_on]  

    
    if __name__ == "__main__":
        pass