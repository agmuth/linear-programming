import numpy as np
from simplex_solvers import *

class TestProblem1():
    tol = 1e-2
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    A = np.array(
        [
            [1, 0, 0, 3, 2, 1, 0, 0],
            [0, 1, 0, 5, 1, 1, 1, 0], 
            [0, 0, 1, 2, 5, 1, 0, 1]
        ]
    )
    b = np.array([1, 3, 4])
    starting_basis = np.array([0, 1, 2])

    optimal_bfs = np.array([1/2, 5/2, 3/2])
    optimal_basis = np.array([4, 6, 7])

    def test_naive_simplex_solver(self):
        solver = NaiveSimplexSolver(self.c, self.A, self.b, self.starting_basis)
        solver.solve()
        assert np.linalg.norm(solver.bfs - self.optimal_bfs, 2) < self.tol and np.array_equal(solver.basis, self.optimal_basis)

    
    def test_revised_simplex_solver(self):
        solver = RevisedSimplexSolver(self.c, self.A, self.b, self.starting_basis)
        solver.solve()
        assert np.linalg.norm(solver.bfs - self.optimal_bfs, 2) < self.tol and np.array_equal(solver.basis, self.optimal_basis)


    def test_tableau_simplex_solver(self):
        solver = TableauSimplexSolver(self.c, self.A, self.b, self.starting_basis)
        solver.solve()
        assert np.linalg.norm(solver.bfs - self.optimal_bfs, 2) < self.tol and np.array_equal(solver.basis, self.optimal_basis)

    
    def test_two_phase_simplex_solver(self):
        solver = TwoPhaseSimplexSolver(self.c, self.A, self.b)
        solver.solve()
        assert np.linalg.norm(solver.tableau.bfs - self.optimal_bfs, 2) < self.tol and np.array_equal(solver.tableau.basis, self.optimal_basis)

