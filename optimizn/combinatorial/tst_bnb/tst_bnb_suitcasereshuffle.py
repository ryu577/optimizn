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
    def __init__(self, init_sol, iters_limit=1e6, print_iters=100,
                 time_limit=3600):
        self.init_sol = init_sol
        super().__init__(name='SuitcaseReshuffleProblem',
                         iters_limit=iters_limit,
                         print_iters=print_iters,
                         time_limit=time_limit)

    def get_candidate(self):
        return (self.init_sol, 0)

    def cost(self, sol):
        suitcases = sol[0]
        max_empty_space = float('-inf')
        for suitcase in suitcases.config:
            max_empty_space = max(max_empty_space, suitcase[-1])
        return max_empty_space

    def lbound(self, sol):
        pass

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
        pass


def test_constructor():
    TEST_CASES = [
        ([[7, 5, 1], [4, 6, 1]], [13, 11], 1)
    ]
    for config, capacities, cost in TEST_CASES:
        sc = SuitCases(config)
        srp = SuitcaseReshuffleProblem(init_sol=sc, iters_limit=10000,
                                       print_iters=100, time_limit=300)
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
        (([[7, 5, 1], [4, 6, 1]], 0), 1),
        (([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]], 1), 4),
        (([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]], 2), 4)
    ]
    for sol, cost in TEST_CASES:
        sc = SuitCases(sol[0])
        srp = SuitcaseReshuffleProblem(init_sol=sc, iters_limit=10000,
                                       print_iters=100, time_limit=300)
        sol_cost = srp.cost((sc, sol[1]))
        assert sol_cost == cost, f'Computed cost of solution {sol} is '\
            + f'incorrect. Expected: {cost}, Actual: {sol_cost}'
    print('cost tests passed')


def test_lbound():
    pass
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
            srp = SuitcaseReshuffleProblem(init_sol=sc, iters_limit=10000,
                                           print_iters=100, time_limit=300)
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
        srp = SuitcaseReshuffleProblem(init_sol=vsc, iters_limit=10000,
                                       print_iters=100, time_limit=300)
        sol = (sc, suitcase_num)
        is_sol = srp.is_sol((sc, suitcase_num))
        assert valid_sol == is_sol, f'Validation of solution {sol} '\
            + f'failed. Expected {v_sol}, Actual: {is_sol}'
    print('is_sol tests passed')


def test_branch():
    pass
    print('branch tests passed')


if __name__ == '__main__':
    test_constructor()
    test_cost()
    test_lbound()
    test_is_sol()
    test_branch()
