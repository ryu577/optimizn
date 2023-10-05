import numpy as np
from optimizn.combinatorial.algorithms.knapsack.bnb_knapsack\
    import KnapsackParams, ZeroOneKnapsackProblem
from tests.combinatorial.algorithms.check_sol_utils import check_bnb_sol,\
    check_sol


def test_bnb_zeroone_knapsack():
    TEST_CASES = [
        # test case: (weights, values, capacity, initial solution, optimal
        # solution)
        (np.array([1, 25, 12, 12]), np.array([1, 24, 12, 12]), 25,
         (np.array([0, 1, 0, 0]), -1), [1, 0, 1, 1]),
        (np.array([10, 10, 15, 1]), np.array([20, 12, 54, 21]), 25,
         (np.array([1, 1, 0, 0]), -1), [0, 0, 1, 1]),
        (np.array([1, 3, 2, 5, 4]), np.array([10, 35, 20, 25, 5]), 4,
         (np.array([0, 0, 1, 0, 0]), -1), [1, 1, 0, 0, 0])
    ]
    for weights, values, capacity, init_sol, opt_sol in TEST_CASES:
        for bnb_type in [0, 1]:
            params = KnapsackParams(values, weights, capacity, init_sol)
            kp = ZeroOneKnapsackProblem(params)
            init_cost = kp.best_cost
            kp.solve(1000, 100, 120, bnb_type)

            # check final solution
            check_bnb_sol(kp, init_cost, bnb_type, params)

            # check final solution optimality
            sol = list(kp.best_solution[0])
            check_sol(sol, opt_sol)
