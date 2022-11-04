import numpy as np
from linearprogramming.tableau import Tableau 


def test_pivot_1():
    c = np.array([1, 1, 1, 1, 1])
    A = np.array(
    [
        [3, 2, 1, 0, 0],
        [2, -1, 0, 1, 0], 
        [-1, 3, 0, 0, 1]
    ]
    )
    b = np.array([1, 2, 3])
    c = np.expand_dims(c, axis=0)
    b = np.expand_dims(b, axis=1)

    bfs = np.array([1, 2, 3])
    basis = np.array([2, 3, 4])

    tbl = Tableau(c, A, b, bfs, basis)

    col_to_enter_basis = 1

    col_to_leave_basis_returned, theta_returned = tbl.calc_col_to_leave_basis_and_theta(col_to_enter_basis)

    col_to_leave_basis = 2
    theta = 0.5
    tbl.pivot(col_to_enter_basis, col_to_leave_basis, theta)

    tbl_after_pivot = np.array(
        [
            [-4.5, 1.5, 0.,  1.5, 0.,  0.],
            [ 0.5, 1.5, 1.,  0.5, 0.,  0.],
            [ 2.5, 3.5, 0., 0.5, 1.,  0.],
            [ 1.5, -5.5, 0., -1.5, 0.,  1.]
        ]

    )
    
    assert all(
        [
            col_to_leave_basis_returned == col_to_leave_basis,
            theta_returned == theta,
            np.array_equal(tbl.tableau, tbl_after_pivot)
        ]
    )