import numpy as np
from linearprogramming.general_solvers import *
import pytest
from tests.problems import PRIMAL_BASE_SOLVER_PROBLEMS

TOL = 1e-2
SOLVERS = [TwoPhaseSimplexSolver]


@pytest.mark.parametrize("problem", PRIMAL_BASE_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_general_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b)
    res = solver.solve()
    assert np.array_equal(res["basis"], problem.optimal_basis)
