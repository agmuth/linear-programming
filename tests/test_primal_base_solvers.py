import numpy as np
from linearprogramming.simplex_solvers import *
import pytest
from problems import *

TOL = 1e-2
SOLVERS = [PrimalNaiveSimplexSolver, PrimalRevisedSimplexSolver, PrimalTableauSimplexSolver]


@pytest.mark.parametrize("problem", PRIMAL_BASE_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_base_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b, problem.starting_basis)
    res = solver.solve()
    assert np.linalg.norm(res["x"] - problem.optimal_bfs, 2) < TOL and np.array_equal(res["basis"], problem.optimal_basis)


@pytest.mark.parametrize("problem", PRIMAL_BASE_SOLVER_BLANDS_SEQUENCE_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_base_simplex_solver_for_blands_seq(problem, solver):
    solver = solver(problem.c, problem.A, problem.b, problem.basis_seq[0])
    path = []
    for _, basis in enumerate(problem.basis_seq[1:]):
        res = solver.solve(maxiters=1)
        path.append(np.array_equal(basis, res["basis"]))
    res = solver.solve(maxiters=1)  # check to make sure algorithm has terminated
    path.append(np.array_equal(problem.basis_seq[-1], res["basis"]))
    assert all(path)