# example of diet problem 
# ref: http://documentation.sas.com/doc/en/orcdc/14.2/ormpug/ormpug_lpsolver_examples01.htm

import numpy as np
from linprog.general_solvers import TwoPhaseRevisedSimplexSolver

# nutritional content
# arrays indexed ('Bread', 'Milk', 'Cheese', 'Potato', 'Fish', 'Yogurt')
foods = ('Bread', 'Milk', 'Cheese', 'Potato', 'Fish', 'Yogurt')
costs = np.array([2.0, 3.5, 8.0, 1.5, 11.0, 1.0, ])
protien = np.array([4.0, 8.0, 7.0, 1.3, 8.0, 9.2, ])
fat = np.array([1.0, 5.0, 9.0, 0.1, 7.0, 1.0, ])
carbohydrates = np.array([15.0, 11.7, 0.4, 22.6, 0.0, 17.0, ])
calories = np.array([0.90, 12, 10.6, 9.7, 13, 18, ])  # divided by 10 throughout

# nutritional constraints/requirements
min_calories = 30
max_protien = 10
min_carbohydrates = 10
min_fat = 8

# upper and lower bounds for specific foods
fish_lb = 0.5
milk_ub = 1.0

# setup problem 
A = np.vstack(
    [
        np.concatenate([calories, np.array([-1, 0, 0, 0, 0, 0])]),
        np.concatenate([protien, np.array([0, 1, 0, 0, 0, 0])]),
        np.concatenate([carbohydrates, np.array([0, 0, -1, 0, 0, 0])]),
        np.concatenate([fat, np.array([0, 0, 0, -1, 0, 0])]),
        np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), # max milk bound
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1]) # min fish bound
    ]
)
m, n = A.shape

c = np.concatenate([np.array(costs), np.zeros(m)])
b = np.array([min_calories, max_protien, min_carbohydrates, min_fat, milk_ub, fish_lb])

solver = TwoPhaseRevisedSimplexSolver(c, A, b)
res = solver.solve()

print(f"\nOptimal Diet Cost: {res.cost}")
print("-"*40)
print("Optimal Diet:")
for food, quantity in zip(foods, res.x[:len(foods)]):
    print(f"{food}: {quantity}")







