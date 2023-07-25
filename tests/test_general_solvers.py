import numpy as np
import pytest

from linprog.primal_solvers import TwoPhaseRevisedSimplexSolver
from linprog.tableau import TwoPhaseTableauSimplexSolver
from tests.constants import TOL
from tests.problems import PRIMAL_BASE_SOLVER_PROBLEMS

SOLVERS = [TwoPhaseTableauSimplexSolver, TwoPhaseRevisedSimplexSolver]


@pytest.mark.parametrize("problem", PRIMAL_BASE_SOLVER_PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_general_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b)
    res = solver.solve()
    assert np.array_equal(res.basis, problem.optimal_basis)


@pytest.mark.parametrize("solver", SOLVERS)
def test_redundent_constraints(solver):
    c = np.array([-1, 2, -3, 0])
    A = np.array(
        [
            [1, 1, 1, 0],
            [-1, 1, 2, 0],
            [0, 2, 3, 0],
            [0, 0, 1, 1],
        ]
    )
    b = np.array([6, 4, 10, 2])
    solver = solver(c, A, b)
    res = solver.solve()
    assert np.array_equal(res.x[res.basis], np.array([2, 2, 2]))


@pytest.mark.parametrize("solver", SOLVERS)
def test_infeasible_constraints(solver):
    c = np.array([-3, 4, 0, 0])
    A = np.array(
        [
            [1, 1, 1, 0],
            [2, 3, 0, -1],
        ]
    )
    b = np.array([4, 18])
    solver = solver(c, A, b)
    with pytest.raises(Exception):
        solver.solve()


if __name__ == "__main__":
    test_redundent_constraints(solver=TwoPhaseRevisedSimplexSolver)
