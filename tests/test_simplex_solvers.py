import numpy as np
from simplex_solvers import *
import pytest


TOL = 1e-2


class StandardFormLPBaseSolverProblem():
    def __init__(self, c, A, b, starting_basis, optimal_bfs, optimal_basis):
        self.c, self.A, self.b = c, A, b
        self.starting_basis = starting_basis
        self.optimal_bfs = optimal_bfs
        self.optimal_basis = optimal_basis


base_solver_problem1 = StandardFormLPBaseSolverProblem(
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0]),
    A = np.array(
        [
            [1, 0, 0, 3, 2, 1, 0, 0],
            [0, 1, 0, 5, 1, 1, 1, 0], 
            [0, 0, 1, 2, 5, 1, 0, 1]
        ]
    ),
    b = np.array([1, 3, 4]),
    starting_basis = np.array([0, 1, 2]),
    optimal_bfs = np.array([1/2, 5/2, 3/2]),
    optimal_basis = np.array([4, 6, 7])
)

base_solver_problem2 = StandardFormLPBaseSolverProblem(
    #linear and nonlinear programming 3rd ed. pg. 48
    c = -1*np.array([3, 1, 3, 0, 0, 0]),
    A = np.array(
        [
            [2, 1, 1, 1, 0, 0],
            [1, 2, 3, 0, 1, 0], 
            [2, 2, 1, 0, 0, 1]
        ]
    ),
    b = np.array([2, 5, 6]),
    starting_basis = np.array([3, 4, 5]),
    optimal_bfs = np.array([1/5, 8/5, 4]),
    optimal_basis = np.array([0, 2, 5])
)

base_solver_problem3 = StandardFormLPBaseSolverProblem(
    #linear prgamming and networkflows ed.2 pg. 110
    c = np.array([-1, -3, 0, 0]),
    A = np.array(
        [
            [2, 3, 1, 0],
            [-1, 1, 0, 1]
        ]
    ),
    b = np.array([6, 1]),
    starting_basis = np.array([2, 3]),
    optimal_bfs =  np.array([3/5, 8/5]),
    optimal_basis =np.array([0, 1])
)

base_solver_problem4 = StandardFormLPBaseSolverProblem(
    #linear prgamming and networkflows ed.2 pg. 117
    c = np.array([1, 1, -4, 0, 0, 0]),
    A = np.array(
        [
            [1, 1, 2, 1, 0, 0],
            [1, 1, -1, 0 ,1, 0],
            [-1, 1, 1, 0, 0, 1],
        ]
    ),
    b = np.array([9, 2, 4]),
    starting_basis = np.array([3, 4, 5]),
    optimal_bfs = np.array([1/3, 6, 13/3]),
    optimal_basis = np.array([0, 4, 2])
)



class StandardFormLPBaseSolverBlandsSequence():
    def __init__(self, c, A, b, basis_seq):
        self.c, self.A, self.b = c, A, b
        self.basis_seq = basis_seq

base_solver_blands_sequence_problem1 = StandardFormLPBaseSolverBlandsSequence(
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0]),
    A = np.array(
        [
            [1, 0, 0, 3, 2, 1, 0, 0],
            [0, 1, 0, 5, 1, 1, 1, 0], 
            [0, 0, 1, 2, 5, 1, 0, 1]
        ]
    ),
    b = np.array([1, 3, 4]),
    basis_seq = np.array(
        [
           [0, 1, 2],  # starting
           [3, 1, 2],
           [4, 1, 2],
           [4, 6, 2], 
           [4, 6, 7],
        ]
    ),
)

# -------------------------------------------------------------------
base_solver_problems = [v for k, v in globals().items() if "base_solver_problem" in k]
base_solver_blands_sequence_problems = [v for k, v in globals().items() if "base_solver_blands_sequence_problem" in k]

base_solvers = [RevisedSimplexSolver, TableauSimplexSolver]

@pytest.mark.parametrize("problem", base_solver_problems)
@pytest.mark.parametrize("solver", base_solvers)
def test_base_simplex_solver_for_correct_soln(problem: StandardFormLPBaseSolverProblem, solver):
    solver = solver(problem.c, problem.A, problem.b, problem.starting_basis)
    solver.solve()
    assert np.linalg.norm(solver.bfs - problem.optimal_bfs, 2) < TOL and np.array_equal(solver.basis, problem.optimal_basis)


@pytest.mark.parametrize("problem", base_solver_blands_sequence_problems)
@pytest.mark.parametrize("solver", base_solvers)
def test_base_simplex_solver_for_blandss_seq(problem: StandardFormLPBaseSolverBlandsSequence, solver):
    solver = solver(problem.c, problem.A, problem.b, problem.basis_seq[0])
    res = []
    for _, basis in enumerate(problem.basis_seq[1:]):
        solver.solve(maxiters=1)
        res.append(np.array_equal(basis, solver.basis))
    solver.solve(maxiters=1) # check to make sure algorithm has terminated
    res.append(np.array_equal(problem.basis_seq[-1], solver.basis))
    assert all(res)
