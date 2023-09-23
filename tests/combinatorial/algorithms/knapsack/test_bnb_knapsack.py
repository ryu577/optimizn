import numpy as np
from optimizn.combinatorial.algorithms.knapsack.bnb_knapsack\
    import KnapsackParams, ZeroOneKnapsackProblem


def test_bnb_zeroone_knapsack():
    weights = [
        np.array([1, 25, 12, 12]),
        np.array([10, 10, 15, 1]),
        np.array([1, 3, 2, 5, 4]),
    ]
    values = [
        np.array([1, 24, 12, 12]),
        np.array([20, 12, 54, 21]),
        np.array([10, 35, 20, 25, 5]),
    ]
    capacity = [
        25,
        25,
        4
    ]
    init_sol = [
        (np.array([0, 1, 0, 0]), -1),
        (np.array([1, 1, 0, 0]), -1),
        (np.array([0, 0, 1, 0, 0]), -1)
    ]
    true_sol = [
        np.array([1, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0, 0])
    ]

    for i in range(len(init_sol)):
        for bnb_type in [0, 1]:
            print('\n=====================')
            print(f'TEST CASE {i+1}\n')
            params = KnapsackParams(
                values[i], weights[i], capacity[i], init_sol[i])
            sol, score = ZeroOneKnapsackProblem(params).solve(
                1000, 100, 120, bnb_type)
            print(f'\nScore: {-1 * score}')
            print(f'Solution: {sol[0]}')
            print(f'True solution: {true_sol[i]}')
            assert list(sol[0]) == list(true_sol[i]), f'Test case {i} failed'
            print('=====================\n')
