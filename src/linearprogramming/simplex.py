import numpy as np
from tableau import Tableau


class BaseSimplexSolver():
    def __init__(self, tableau: Tableau) -> None:
        """_summary_

        Args:
            tableau (Tableau): _description_
        """

        self.tableau = tableau


    def solve(self, maxiters: int=100) -> None:
        for _ in range(maxiters):
            if self.tableau.tableau[self.tableau.cost_index, 1:].min() >= 0: 
                break
            col_to_enter_basis = self.tableau.calc_col_to_enter_basis()
            col_to_leave_basis, theta = self.tableau.calc_col_to_leave_basis_and_theta(col_to_enter_basis)
            self.tableau.pivot(col_to_enter_basis, col_to_leave_basis, theta)


if __name__ == "__main__":
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    A = np.array(
        [
            [1, 0, 0, 3, 2, 1, 0, 0],
            [0, 1, 0, 5, 1, 1, 1, 0], 
            [0, 0, 1, 2, 5, 1, 0, 1]
        ]
    )
    b = np.array([1, 3, 4])
    c = np.expand_dims(c, axis=0)
    b = np.expand_dims(b, axis=1)

    bfs = np.array([1, 3, 4])
    basis = np.array([0, 1, 2])

    tableau = Tableau(c, A, b, bfs, basis)
    solver = BaseSimplexSolver(tableau)
    solver.solve()
    print(solver.tableau.bfs)