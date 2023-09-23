from optimizn.combinatorial.algorithms.binpacking.bnb_binpacking import\
    BinPackingParams, BinPackingProblem


def test_param_equality():
    TEST_CASES = [
        (
            BinPackingParams([1, 2, 3, 4], [6]),
            None,
            False
        ),
        (
            BinPackingParams([1, 2, 3, 4], [7]),
            BinPackingParams([1, 2, 3, 4], [6]),
            False
        ),
        (
            BinPackingParams([1, 2, 3, 4], [6]),
            BinPackingParams([1, 2, 3, 4], [6]),
            True
        )
    ]
    for params1, params2, equal in TEST_CASES:
        assert (params1 == params2) == equal


def test_constructor():
    TEST_CASES = [
        ([1, 2, 3], 3, {1: {3}, 2: {1, 2}}),
        ([7, 8, 2, 3], 15, {1: {2, 1}, 2: {4, 3}})
    ]
    for weights, capacity, expected in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check capacity
        assert bpp.capacity == capacity

        # check item weights
        for i in range(len(weights)):
            assert bpp.item_weights[i + 1] == weights[i]

        # check sorted item weights
        for i in range(len(bpp.sorted_item_weights)):
            weight, item = bpp.sorted_item_weights[i]
            assert bpp.item_weights[item] == weight
            if i > 0:
                assert weight < bpp.sorted_item_weights[i - 1][0]

        # check initial solution
        assert bpp.best_solution[0] == expected
        assert bpp.best_solution[1] == -1
    print('Constructor tests passed')


def test_is_feasible():
    TEST_CASES = [
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}, 3: {1}}, 1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}}, 1), True),
        ([1, 2, 3], 3, ({1: {1, 2, 3}}, -1), False),
        ([1, 2, 3], 3, ({1: {3, 1}, 2: {2}}, -1), False),
        ([1, 2, 3], 3, ({1: {3, 2}, 2: {1}}, 1), False),
        ([1, 2, 3], 3, ({1: {3, 2}, 2: {1}}, 1), False)
    ]
    for weights, capacity, sol, feasible in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        assert bpp.is_feasible(sol) == feasible


def test_is_complete():
    TEST_CASES = [
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}, 3: {1}}, 1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}}, 1), False),
        ([1, 2, 3], 3, ({1: {3}, 2: {1}}, 1), False),
        ([1, 2, 3], 3, ({1: {2}, 2: {1}}, 1), False)
    ]
    for weights, capacity, sol, complete in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        assert bpp.is_complete(sol) == complete


def test_cost():
    TEST_CASES = [
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), 2),
        ([1, 2, 3], 3, ({1: {1, 2}, 2: {3}}, -1), 2),
        ([1, 2, 3], 3, ({1: {2}, 2: {3}, 3: {1}}, 2), 3),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}, 3: {1}}, 2), 3)
    ]
    for weights, capacity, sol, cost in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check cost
        assert bpp.cost(sol) == cost
    print('Cost tests passed')


def test_lbound():
    TEST_CASES = [
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), 2),
        ([1, 2, 3], 3, ({1: {3}}, 0), 2),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}}, 1), 2),
        ([1, 2, 3], 3, ({1: {3}, 2: {2, 1}}, 2), 2),
    ]
    for weights, capacity, sol, lb in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check lower bounds
        assert bpp.lbound(sol) == lb

    print('Lower bound tests passed')


def test_branch():
    TEST_CASES = [
        ([1, 2, 3], 3, [
            ({1: {3}, 2: {2}}, 1),
        ], ({1: {3}}, 0)),
        ([7, 8, 2, 3], 15, [
            ({1: {1, 2}}, 1),
            ({1: {2}, 2: {1}}, 1),
        ], ({1: {2}}, 0)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16, [
            ({1: {6}, 2: {5, 10}, 3: {4}}, 3),
            ({1: {6}, 2: {5}, 3: {4, 10}}, 3),
            ({1: {6}, 2: {5}, 3: {4}, 4: {10}}, 3)
        ], ({1: {6}, 2: {5}, 3: {4}}, 2)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16, [
            ({1: {6}}, 0)
        ], ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, -1)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16, [
            ({1: {6}, 2: {5}}, 1)
        ], ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, 0))
    ]
    for weights, capacity, expected, init_sol in TEST_CASES:
        # check branch
        params = BinPackingParams(
            weights,
            capacity
        )
        bpp = BinPackingProblem(params)
        new_sols = bpp.branch(init_sol)
        for new_sol in new_sols:
            assert new_sol in expected
        for exp_sol in expected:
            assert exp_sol in new_sols

    print('Branch tests passed')


def test_complete_solution():
    TEST_CASES = [
        ([1, 2, 3], 3, ({1: {3}}, 0), ({1: {3}, 2: {1, 2}}, 0)),
        ([7, 8, 2, 3], 15, ({1: {2}}, 0), ({1: {2, 1}, 2: {3, 4}}, 0)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16,
         ({1: {6}, 2: {5}, 3: {4}}, 2),
         ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, 2)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16,
         (dict(), -1),
         ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, -1))
    ]
    for weights, capacity, incomplete_sol, complete_sol in TEST_CASES:
        # check branch
        params = BinPackingParams(
            weights,
            capacity
        )
        bpp = BinPackingProblem(params)
        sol = bpp.complete_solution(incomplete_sol)
        assert sol == complete_sol

def test_bnb_binpacking():
    # weights, capacity, min bins (optimal solution)
    TEST_CASES = [
        ([1, 2, 3], 3, 2),
        ([7, 8, 2, 3], 15, 2),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12, 5),
        ([49, 41, 34, 33, 29, 26, 26, 22, 20, 19], 100, 3),
        ([49, 41, 34, 33, 29, 26, 26, 22, 20, 19] * 2, 100, 6)
    ]
    for weights, capacity, min_bins in TEST_CASES:
        for bnb_type in [0, 1]:
            print('-----------------')
            params = BinPackingParams(weights, capacity)
            bpp = BinPackingProblem(params)
            print('Sorted item weights (w, i):', bpp.sorted_item_weights)
            print('Item weights:', bpp.item_weights)
            bpp.solve(1000, 100, 120, bnb_type)
            print('\nItem weight dictionary:', bpp.item_weights)
            print('Final solution:', bpp.best_solution[0])
            print('Score:', bpp.best_cost)
            print('Optimal solution reached: ', min_bins == bpp.best_cost)
            print('-----------------')
