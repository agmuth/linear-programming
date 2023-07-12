import numpy as np
import pytest

from linprog.primal_dual_alogrithm import *
from tests.problems import PRIMAL_DUAL_SOLVER_PROBLEMS
from tests.constants import TOL

SOLVERS = [PrimalDualAlgorithm]


@pytest.mark.parametrize("problem", PRIMAL_DUAL_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_general_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b)
    res = solver.solve()
    assert np.array_equal(res.x, problem.optimal_bfs)


if __name__ == "__main__":
    test_general_simplex_solver_for_correct_soln(
        PRIMAL_DUAL_SOLVER_PROBLEMS[0],
        SOLVERS[0]
    )