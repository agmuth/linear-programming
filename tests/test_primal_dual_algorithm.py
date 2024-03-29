import numpy as np
import pytest

from linprog.special_solvers import PrimalDualAlgorithm
from tests.constants import TOL
from tests.problems import PRIMAL_DUAL_SOLVER_PROBLEMS

SOLVERS = [PrimalDualAlgorithm]


@pytest.mark.parametrize("problem", PRIMAL_DUAL_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_general_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b)
    res = solver.solve()
    assert np.array_equal(res.x, problem.optimal_bfs)
