from optimizn.combinatorial.algorithms.suitcase_reshuffle\
    .bnb_suitcasereshuffle import SuitcaseReshuffleProblem
from copy import deepcopy
from optimizn.combinatorial.algorithms.suitcase_reshuffle.suitcases\
    import SuitCases


def test_constructor():
    TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], [13, 11], -1)
    ]
    for config, capacities, cost in TEST_CASES:
        sc = SuitCases(config)
        params = {
            'init_sol': sc,
        }
        srp = SuitcaseReshuffleProblem(params)
        init_sol = srp.best_solution
        init_config = init_sol[0].config
        init_caps = init_sol[0].capacities
        init_suitcase_num = init_sol[1]
        assert init_config == config, 'Initial solution configs do not '\
            + f'match. Expected: {config}, Actual: {init_config}'
        assert init_caps == capacities, 'Initial solution capacities '\
            + f'do not match. Expected: {capacities}, Actual: {init_caps}'
        assert srp.best_cost == cost, 'Initial solution scores do not '\
            + f'match. Expected: {cost}, Actual: {srp.best_cost}'
        assert init_suitcase_num == 0, 'Initial solution suitcase number '\
            + f'for branching is not 0: {init_suitcase_num}'
    print('constructor tests passed')


def test_cost():
    TEST_CASES = [
        (([[7, 5, 1], [4, 6, 1]], 0), -1),
        (([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]], 1), -4),
        (([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]], 2), -4)
    ]
    for sol, cost in TEST_CASES:
        sc = SuitCases(sol[0])
        params = {
            'init_sol': sc
        }
        srp = SuitcaseReshuffleProblem(params)
        sol_cost = srp.cost((sc, sol[1]))
        assert sol_cost == cost, f'Computed cost of solution {sol} is '\
            + f'incorrect. Expected: {cost}, Actual: {sol_cost}'
    print('cost tests passed')


def test_lbound():
    TEST_CASES = [
        (([[7, 5, 1], [4, 6, 1]], 0), -2),
        (([[7, 5, 3], [4, 6, 1]], 0), -4),
        (([[7, 5, 1], [4, 6, 4]], 0), -5)
    ]
    for sol, lbound in TEST_CASES:
        sc = SuitCases(sol[0])
        params = {
            'init_sol': sc
        }
        srp = SuitcaseReshuffleProblem(params)
        sol_lb = srp.lbound((sc, sol[1]))
        assert sol_lb == lbound, f'Computed cost of solution {sol} is '\
            + f'incorrect. Expected: {lbound}, Actual: {sol_lb}'
    print('lbound tests passed')


def test_is_feasible():
    TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], 0, True),
        ([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]], 1, True),
        ([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]], 2, True),
    ]
    for config, suitcase_num, valid_sol in TEST_CASES:
        sc1 = SuitCases(config)
        new_config = deepcopy(config)
        for i in range(len(new_config)):
            new_config[i][0] += 1
        sc2 = SuitCases(new_config)
        for sc, v_sol in [(sc1, valid_sol), (sc2, False)]:
            params = {
                'init_sol': sc
            }
            srp = SuitcaseReshuffleProblem(params)
            sol = (sc, suitcase_num)
            is_feasible = srp.is_feasible((sc, suitcase_num))
            assert valid_sol == is_feasible, f'Validation of solution {sol} '\
                + f'failed. Expected {v_sol}, Actual: {is_feasible}'
    OTHER_TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], [[7, 5, -1], [4, 6, 1]], 0, False),
        ([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]],
            [[7, 5, 1], [4, 6, 1], [12, -12, -4], [11, 10, 2]], 1, False),
        ([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]],
            [[7, 5, -1], [4, 6, 1], [12, 12, 4], [11, 10, -2]], 2, False),
    ]
    for valid_config, config, suitcase_num, valid_sol in OTHER_TEST_CASES:
        vsc = SuitCases(valid_config)
        sc = SuitCases(config)
        params = {
            'init_sol': vsc
        }
        srp = SuitcaseReshuffleProblem(params)
        sol = (sc, suitcase_num)
        is_feasible = srp.is_feasible((sc, suitcase_num))
        assert valid_sol == is_feasible, f'Validation of solution {sol} '\
            + f'failed. Expected {v_sol}, Actual: {is_feasible}'
    print('is_feasible tests passed')


def test_branch():
    TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], 0, [
            [[7, 5, 1], [4, 6, 1]],
            [[6, 5, 2], [4, 7, 0]],
            [[7, 4, 2], [5, 6, 0]],
            [[7, 6, 0], [4, 5, 2]],
        ]),
        ([[7, 5, 1], [4, 6, 1]], 1, []),
        ([[7, 5, 1], [4, 6, 1], [5, 5, 1]], 0, [
            [[7, 5, 1], [4, 6, 1], [5, 5, 1]],
            [[6, 5, 2], [4, 7, 0], [5, 5, 1]],
            [[7, 4, 2], [5, 6, 0], [5, 5, 1]],
            [[7, 6, 0], [4, 5, 2], [5, 5, 1]],
        ]),
        ([[7, 5, 1], [4, 6, 1], [5, 5, 1]], 1, [
            [[7, 5, 1], [4, 6, 1], [5, 5, 1]],
            [[7, 5, 1], [5, 6, 0], [4, 5, 2]],
            [[7, 5, 1], [5, 6, 0], [5, 4, 2]],
            [[7, 5, 1], [4, 5, 2], [6, 5, 0]],
            [[7, 5, 1], [4, 5, 2], [5, 6, 0]],
        ]),
        ([[7, 5, 4], [4, 6, 1], [5, 5, 1]], 0, [
            [[7, 5, 4], [4, 6, 1], [5, 5, 1]],
            [[7, 5, 4, 0], [6, 5], [5, 5, 1]],
            [[6, 5, 5], [4, 7, 0], [5, 5, 1]],
            [[7, 4, 5], [5, 6, 0], [5, 5, 1]],
            [[7, 6, 3], [4, 5, 2], [5, 5, 1]]
        ])
    ]
    for config, suitcase_num, branch_sols in TEST_CASES:
        sc = SuitCases(config)
        params = {
            'init_sol': sc
        }
        srp = SuitcaseReshuffleProblem(params)
        sol = (sc, suitcase_num)
        new_sols = srp.branch(sol)
        assert len(new_sols) == len(branch_sols), 'Length of branched '\
            + f'solutions for solution {sol} are incorrect. Expected: '\
            + f'{branch_sols}, Actual: {new_sols}'
        for i in range(len(new_sols)):
            new_sc, new_sc_num = new_sols[i]
            assert new_sc_num == suitcase_num + 1, 'Incorrect suitcase index '\
                + f'for next branching. Expected: {suitcase_num + 1}, Actual:'\
                + f' {new_sc_num}'
            assert branch_sols[i] == new_sc.config, 'Branched solution for '\
                + f'solution {sol} is incorrect. Expected: {branch_sols[i]}'\
                + f', Actual: {new_sc.config}'
    print('branch tests passed')


def test_bnb_suitcasereshuffle():
    TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], -2),
        ([[7, 5, 4], [4, 6, 1], [5, 5, 1]], -6),
        ([[1, 4, 3, 6, 4, 2], [2, 4, 7, 1, 0], [1, 7, 3, 8, 3, 4]], -6),
        ([
            [12, 52, 34, 23, 17, 18, 22, 10],
            [100, 21, 36, 77, 82, 44, 40],
            [1, 5, 2, 8, 22, 34, 50]
        ], -100)
    ]
    for config, final_cost in TEST_CASES:
        for bnb_type in [0, 1]:
            sc = SuitCases(config)
            params = {
                'init_sol': sc
            }
            srp = SuitcaseReshuffleProblem(params)
            srp.solve(1000, 100, 120, bnb_type)
            # srp.persist() # does not work
            print('Best solution: ', srp.best_solution[0].config)
            print(f'Expected cost: {final_cost}, Actual cost: {srp.best_cost}')
