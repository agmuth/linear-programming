import numpy as np
import pytest

from linprog.dual_simplex_solvers import *
from tests.problems import *
from tests.constants import TOL

SOLVERS = [
    DualNaiveSimplexSolver,
    DualRevisedSimplexSolver,
    DualTableauSimplexSolver,
]


@pytest.mark.parametrize("problem", DUAL_BASE_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_base_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b, problem.starting_basis)
    res = solver.solve()
    assert np.linalg.norm(res.x[res.basis] - problem.optimal_bfs, 2) < TOL and np.array_equal(
        res.basis, problem.optimal_basis
    )


if __name__ == "__main__":
    test_base_simplex_solver_for_correct_soln(
        DUAL_BASE_SOLVER_PROBLEMS[0],
        SOLVERS[0]
    )