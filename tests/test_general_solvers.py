import numpy as np
from linearprogramming.general_solvers import *
import pytest
from linearprogramming.utils import gen_lp_problem

TOL = 1e-2
SOLVERS = [TwoPhaseSimplexSolver]

class LinearProgrammignProblem():
    def __init__(self, c, A, b, optimal_basis):
        self.c, self.A, self.b = c, A, b
        self.optimal_basis = optimal_basis

problem1 = LinearProgrammignProblem(
    *gen_lp_problem(
        c=np.ones(2),
        k=2, 
        u=1*np.ones(2),
        G = np.array(
            [
                [1, 0],
                [0, 1], 
                [1, 1]
            ]
        )
    )
)

PROBLEMS = [problem1]


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_general_simplex_solver_for_correct_soln(problem, solver):
    solver = solver(problem.c, problem.A, problem.b)
    res = solver.solve()
    assert np.array_equal(res["basis"], problem.optimal_basis)
