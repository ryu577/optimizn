from optimizn.combinatorial.branch_and_bound import BnBProblem
from copy import deepcopy


class SuitCases():
    def __init__(self, config):
        """
        The configuration of the suitcases
        is an array of arrays. The last element
        of each array must be the amount of empty space.
        This means that the sum of each array is the 
        capacity of that suitcase.
        """
        self.config = config
        self.capacities = []
        for ar in config:
            self.capacities.append(sum(ar))

    def __eq__(self, other):
        return (
            self.config == other.config
            and self.capacities == other.capacities
        )


class SuitcaseReshuffleProblem(BnBProblem):
    '''
    Solution Format:
    2-tuple
    1. SuitCases object containing suitcases, weights, and empty space
    2. Suitcase number - for branching, determines which suitcase to start the
    swaps from (0-indexed)

    Branching strategy:
    For suitcase number i, create a new solution for each swap between an item 
    in suitcase i and an item in suitcase i + 1. Stop once i is two less than 
    the number of suitcases
    '''

    def __init__(self, params):
        self.init_sol = params['init_sol']
        super().__init__(params)

    def get_candidate(self):
        return (self.init_sol, 0)

    def cost(self, sol):
        suitcases = sol[0]
        max_empty_space = float('-inf')
        for suitcase in suitcases.config:
            max_empty_space = max(max_empty_space, suitcase[-1])
        return -1 * max_empty_space

    def lbound(self, sol):
        suitcases = sol[0]
        empty_space = 0
        for suitcase in suitcases.config:
            empty_space += suitcase[-1]
        return -1 * empty_space

    def is_sol(self, sol):
        suitcases = sol[0]
        for i in range(len(suitcases.config)):
            suitcase = suitcases.config[i]
            # weights and extra space must be non-negative
            suitcase_sum = 0
            for item in suitcase:
                suitcase_sum += item
                if item < 0:
                    return False
            # weights and extra space must equal original capacity
            capacity = suitcases.capacities[i]
            if suitcase_sum != capacity:
                return False
        return True

    def branch(self, sol):
        suitcases = sol[0]
        curr = sol[1]
        if curr > len(suitcases.config) - 2:
            return []

        # produce a solution where no items are swapped or moved between
        # suitcases curr and curr + 1
        new_sols = []
        new_sols.append((suitcases, curr + 1))

        # produce solutions where only one item is moved from suitcase
        # swap_from to suitcase swap_from + 1, and vice versa
        for from_ix, to_ix in [
                (curr, curr + 1), (curr + 1, curr)]:
            for i in range(len(suitcases.config[from_ix]) - 1):
                new_suitcases = deepcopy(suitcases)

                # compute empty space change, see if move is possible
                empty_space_change = new_suitcases.config[from_ix][i]
                new_suitcases.config[from_ix][-1] += empty_space_change
                new_suitcases.config[to_ix][-1] -= empty_space_change
                if (new_suitcases.config[from_ix][-1] < 0 or
                        new_suitcases.config[to_ix][-1] < 0):
                    continue

                # move item
                new_suitcases.config[to_ix].insert(
                    -1, new_suitcases.config[from_ix][i])
                del new_suitcases.config[from_ix][i]

                # create new solution
                new_sol = (new_suitcases, curr + 1)
                new_sols.append(new_sol)

        # produce a new solution for each swap between suitcases
        # swap_from and swap_from + 1
        for i1 in range(len(suitcases.config[curr]) - 1):
            for i2 in range(len(suitcases.config[curr + 1]) - 1):
                new_suitcases = deepcopy(suitcases)

                # compute empty space change, see if swap is possible
                empty_space_change = new_suitcases.config[curr + 1][i2] \
                    - new_suitcases.config[curr][i1]
                new_suitcases.config[curr][-1] -= empty_space_change
                new_suitcases.config[curr + 1][-1] += empty_space_change
                if (new_suitcases.config[curr][-1] < 0 or
                        new_suitcases.config[curr + 1][-1] < 0):
                    continue

                # swap items
                temp = new_suitcases.config[curr][i1]
                new_suitcases.config[curr][i1] = new_suitcases.config[
                    curr + 1][i2]
                new_suitcases.config[curr + 1][i2] = temp

                # create new solution
                new_sol = (new_suitcases, curr + 1)
                new_sols.append(new_sol)
        return new_sols


def test_constructor():
    TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], [13, 11], -1)
    ]
    for config, capacities, cost in TEST_CASES:
        sc = SuitCases(config)
        params = {
            'init_sol': sc,
            'iters_limit': 1000,
            'print_iters': 100,
            'time_limit': 600
        }
        srp = SuitcaseReshuffleProblem(params)
        init_sol = srp.best_solution
        init_config = init_sol[0].config
        init_caps = init_sol[0].capacities
        init_suitcase_num = init_sol[1]
        assert srp.iters_limit == params['iters_limit'], 'Incorrect '\
            + f'iters_limit. Expected: {params["iters_limit"]}, '\
            + f'Actual: {srp.iters_limit}'
        assert srp.print_iters == params['print_iters'], 'Incorrect '\
            + f'print_iters. Expected: {params["print_iters"]}, '\
            + f'Actual: {srp.print_iters}'
        assert srp.time_limit == params['time_limit'], 'Incorrect '\
            + f'time_limit. Expected: {params["time_limit"]}, '\
            + f'Actual: {srp.time_limit}'
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


def test_is_sol():
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
            is_sol = srp.is_sol((sc, suitcase_num))
            assert valid_sol == is_sol, f'Validation of solution {sol} '\
                + f'failed. Expected {v_sol}, Actual: {is_sol}'
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
        is_sol = srp.is_sol((sc, suitcase_num))
        assert valid_sol == is_sol, f'Validation of solution {sol} '\
            + f'failed. Expected {v_sol}, Actual: {is_sol}'
    print('is_sol tests passed')


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
        # ([[7, 5, 4], [4, 6, 1], [5, 5, 1]], -6),
        # ([[1, 4, 3, 6, 4, 2], [2, 4, 7, 1, 0], [1, 7, 3, 8, 3, 4]], -6),
        # ([
        #     [12, 52, 34, 23, 17, 18, 22, 10],
        #     [100, 21, 36, 77, 82, 44, 40],
        #     [1, 5, 2, 8, 22, 34, 50]
        # ], -100)
    ]
    for config, final_cost in TEST_CASES:
        sc = SuitCases(config)
        params = {
            'init_sol': sc
        }
        srp = SuitcaseReshuffleProblem(params)
        srp.solve()
        # srp.persist() # does not work
        print('Best solution: ', srp.best_solution[0].config)
        print(f'Expected cost: {final_cost}, Actual cost: {srp.best_cost}')


if __name__ == '__main__':
    print('Unit tests:')
    test_constructor()
    test_cost()
    test_lbound()
    test_is_sol()
    test_branch()
    print('-----------------\n')

    test_bnb_suitcasereshuffle()
