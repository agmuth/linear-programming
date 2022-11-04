import numpy as np 
from simplex import BaseSimplexSolver
from tableau import Tableau 


def test_base_simplex_solver_1():
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    A = np.array(
        [
            [1, 0, 0, 3, 2, 1, 0, 0],
            [0, 1, 0, 5, 1, 1, 1, 0], 
            [0, 0, 1, 2, 5, 1, 0, 1]
        ]
    )
    b = np.array([1, 3, 4])
    c = np.expand_dims(c, axis=0)
    b = np.expand_dims(b, axis=1)

    bfs = np.array([1, 3, 4])
    basis = np.array([0, 1, 2])
    tableau = Tableau(c, A, b, bfs, basis)
    solver = BaseSimplexSolver(tableau)

    bfs_seq = np.array(
        [
        #    [1, 3, 4],  # starting
           [1/3, 4/3, 10/3],
           [1/2, 5/2, 3/2],
           [1/2, 5/2, 3/2], 
           [1/2, 5/2, 3/2],
        ]
    )

    basis_seq = np.array(
        [
        #    [0, 1, 2],  # starting
           [3, 1, 2],
           [4, 1, 2],
           [4, 6, 2], 
           [4, 6, 7],
        ]
    )

    bfs_res = []
    basis_res = []
    tol = 1e-2

    for i in range(bfs_seq.shape[0]):
        col_to_enter_basis = solver.tableau.calc_col_to_enter_basis()
        col_to_leave_basis, theta = solver.tableau.calc_col_to_leave_basis_and_theta(col_to_enter_basis)
        solver.tableau.pivot(col_to_enter_basis, col_to_leave_basis, theta)

        bfs_res.append(np.abs(bfs_seq[i] - solver.tableau.bfs).max() < tol)
        basis_res.append(np.abs(basis_seq[i] - solver.tableau.basis).max() < tol)

    assert all(
        [
            all(bfs_res), 
            all(basis_res),
            solver.tableau.tableau[solver.tableau.cost_index, 1:].min() >= 0  # check to see algo would terminate
        ]
    )