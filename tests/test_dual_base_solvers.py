import numpy as np
import pytest

from linprog.dual_solvers import *
from linprog.tableau import DualTableauSimplexSolver
from tests.constants import TOL
from tests.problems import *

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
    assert np.linalg.norm(
        res.x[res.basis] - problem.optimal_bfs, 2
    ) < TOL and np.array_equal(res.basis, problem.optimal_basis)
