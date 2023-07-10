import numpy as np


class StandardFormLPP:
    def __init__(self, c, A, b, starting_basis, optimal_bfs, optimal_basis):
        self.c, self.A, self.b = c, A, b
        self.starting_basis = starting_basis
        self.optimal_bfs = optimal_bfs
        self.optimal_basis = optimal_basis


class StandardFormLPBlandsSequence:
    def __init__(self, c, A, b, basis_seq):
        self.c, self.A, self.b = c, A, b
        self.basis_seq = basis_seq


primal_standard_form_lpp1 = StandardFormLPP(
    # Combinatorial Optimization - Algorithms and Complexity Papadimitriou pg. 57
    c=np.array([1, 1, 1, 0, 0, 0, 0, 0]),
    A=np.array(
        [[1, 0, 0, 3, 2, 1, 0, 0], [0, 1, 0, 5, 1, 1, 1, 0], [0, 0, 1, 2, 5, 1, 0, 1]]
    ),
    b=np.array([1, 3, 4]),
    starting_basis=np.array([0, 1, 2]),
    optimal_bfs=np.array([1 / 2, 5 / 2, 3 / 2]),
    optimal_basis=np.array([4, 6, 7]),
)

primal_standard_form_lpp2 = StandardFormLPP(
    # linear and nonlinear programming 3rd ed. pg. 48
    c=-1 * np.array([3, 1, 3, 0, 0, 0]),
    A=np.array([[2, 1, 1, 1, 0, 0], [1, 2, 3, 0, 1, 0], [2, 2, 1, 0, 0, 1]]),
    b=np.array([2, 5, 6]),
    starting_basis=np.array([3, 4, 5]),
    optimal_bfs=np.array([1 / 5, 8 / 5, 4]),
    optimal_basis=np.array([0, 2, 5]),
)

primal_standard_form_lpp3 = StandardFormLPP(
    # linear prgamming and networkflows ed.2 pg. 110
    c=np.array([-1, -3, 0, 0]),
    A=np.array([[2, 3, 1, 0], [-1, 1, 0, 1]]),
    b=np.array([6, 1]),
    starting_basis=np.array([2, 3]),
    optimal_bfs=np.array([3 / 5, 8 / 5]),
    optimal_basis=np.array([0, 1]),
)

primal_standard_form_lpp4 = StandardFormLPP(
    # linear prgamming and networkflows ed.2 pg. 117
    c=np.array([1, 1, -4, 0, 0, 0]),
    A=np.array(
        [
            [1, 1, 2, 1, 0, 0],
            [1, 1, -1, 0, 1, 0],
            [-1, 1, 1, 0, 0, 1],
        ]
    ),
    b=np.array([9, 2, 4]),
    starting_basis=np.array([3, 4, 5]),
    optimal_bfs=np.array([1 / 3, 6, 13 / 3]),
    optimal_basis=np.array([0, 4, 2]),
)


dual_standard_form_lpp1 = StandardFormLPP(
    # Introduction to Linear Programming Bertimas pg. 162
    c=np.array([1, 1, 0, 0]),
    A=np.array(
        [
            [-1, -2, 1, 0],
            [-1, 0, 0, 1],
        ]
    ),
    b=np.array([-2, -1]),
    starting_basis=np.array([2, 3]),
    optimal_bfs=np.array([0.5, 1.0]),
    optimal_basis=np.array([1, 0]),
)

dual_standard_form_lpp2 = StandardFormLPP(
    # Linear and Nonlinear Programming LuenBerger pg. 93
    c=np.array([3, 4, 5, 0, 0]),
    A=np.array(
        [
            [-1, -2, -3, 1, 0],
            [-2, -2, -1, 0, 1],
        ]
    ),
    b=np.array([-5, -6]),
    starting_basis=np.array([3, 4]),
    optimal_bfs=np.array([1.0, 2.0]),
    optimal_basis=np.array([0, 1]),
)


primal_solver_blands_sequence_problem1 = StandardFormLPBlandsSequence(
    c=np.array([1, 1, 1, 0, 0, 0, 0, 0]),
    A=np.array(
        [[1, 0, 0, 3, 2, 1, 0, 0], [0, 1, 0, 5, 1, 1, 1, 0], [0, 0, 1, 2, 5, 1, 0, 1]]
    ),
    b=np.array([1, 3, 4]),
    basis_seq=np.array(
        [
            [0, 1, 2],  # starting
            [3, 1, 2],
            [4, 1, 2],
            [4, 6, 2],
            [4, 6, 7],
        ]
    ),
)

primal_dual_test_problem1 = StandardFormLPP(
    # example 6.8 pg 272 linear programming and network flows
    c=np.array([3, 4, 6, 7, 5, 0, 0]),
    A=np.array(
        [
            [2, -1, 1, 6, -5, -1, 0],
            [1, 1, 2, 1, 2, 0, -1],
        ]
    ),
    b=np.array([6, 3]),
    starting_basis=None,
    optimal_bfs=np.array([3, 0, 0, 0, 0, 0, 0]),
    optimal_basis=None,
)

primal_dual_test_problem2 = StandardFormLPP(
    # pg 96 linear and nonlinear programming
    c=np.array([2, 1, 4]),
    A=np.array(
        [
            [1, 1, 2],
            [2, 1, 3],
        ]
    ),
    b=np.array([3, 5]),
    starting_basis=None,
    optimal_bfs=np.array([2, 1, 0]),
    optimal_basis=None,
)


primal_dual_test_problem3 = StandardFormLPP(
    c=np.array([-2, 1, -1, 0, 0]),
    A=np.array([[1, 1, 1, 1, 0], [-1, 2, 0, 0, 1]]),
    b=np.array([6, 4]),
    starting_basis=None,
    optimal_bfs=np.array([6.0, 0.0, 0.0, 0.0, 10.0]),
    optimal_basis=None,
)


PRIMAL_BASE_SOLVER_PROBLEMS = [
    v for k, v in globals().items() if "primal_standard_form_lpp" in k
]
DUAL_BASE_SOLVER_PROBLEMS = [
    v for k, v in globals().items() if "dual_standard_form_lpp" in k
]
PRIMAL_BASE_SOLVER_BLANDS_SEQUENCE_PROBLEMS = [
    v for k, v in globals().items() if "primal_solver_blands_sequence_problem" in k
]
PRIMAL_DUAL_SOLVER_PROBLEMS = [
    v for k, v in globals().items() if "primal_dual_test_problem" in k
]
