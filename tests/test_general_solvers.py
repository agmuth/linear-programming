import numpy as np
import pytest

from linprog.general_solvers import *
from tests.constants import TOL
from tests.problems import PRIMAL_BASE_SOLVER_PROBLEMS

SOLVERS = [TwoPhaseSimplexSolver]


@pytest.mark.parametrize("problem", PRIMAL_BASE_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_general_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b)
    res = solver.solve()
    assert np.array_equal(res.basis, problem.optimal_basis)
