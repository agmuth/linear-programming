import numpy as np

from linprog.special_solvers import BoundedVariablePrimalSimplexSolver


def test_1():
    # ex. 5.6 linear programming and network flows Bazara
    # TODO: move to problems.py
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
    assert np.isclose(res.x, np.array([2 / 3, 6, 8 / 3, 0, 0])).all()
