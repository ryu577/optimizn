from optimizn.combinatorial.branch_and_bound import BnBProblem
from functools import reduce
import copy
import math


class BinPackingParams:
    def __init__(self, weights, capacity):
        self.weights = weights
        self.capacity = capacity

    def __eq__(self, other):
        return (
            other is not None
            and self.weights == other.weights
            and self.capacity == other.capacity
        )


# References:
# http://www.or.deis.unibo.it/knapsack.html (See PDF for Chapter 8,
#   http://www.or.deis.unibo.it/kp/Chapter8.pdf)
class BinPackingProblem(BnBProblem):
    '''
    Solution format:
    1. Allocation of items to bins (dict, keys are integers representing bins
    (starting from 1, so 1 represents the first bin, 2 is the second bin, etc.)
    and values are sets of integers that represent the items (1 represents
    first item in weights list, 2 represents second item in weights list, and
    so on))
    2. Index of last allocated item in sorted-by-decreasing-weight list of
    items (int)

    Branching strategy:
    Each level of the solution space tree corresponds to an item. Items are
    considered in order of decreasing weight. Each solution in a level 
    corresponds to the item being placed in a bin that it can fit in. The
    remaining items can be put in bins in decreasing order of weight, into
    the first bin that can fit it. New bins created as needed
    '''
    def __init__(self, params):
        self.item_weights = {}  # mapping of items to weights
        self.sorted_item_weights = []  # sorted (weight, item) tuples (desc)
        for i in range(1, len(params.weights) + 1):
            self.item_weights[i] = params.weights[i - 1]
            self.sorted_item_weights.append((params.weights[i - 1], i))
        self.sorted_item_weights.sort(reverse=True)
        self.capacity = params.capacity
        super().__init__(params)
    
    def get_candidate(self):
        return (self._pack_rem_items(dict(), -1), -1)

    def _pack_rem_items(self, bin_packing, last_item_idx):
        next_item_idx = last_item_idx + 1
        for i in range(next_item_idx, len(self.sorted_item_weights)):
            next_item_weight, next_item = self.sorted_item_weights[i]
            bins = set(bin_packing.keys())
            item_packed = False
            for bin in bins:
                # check if bin has space
                bin_weight = sum(
                    list(map(
                        lambda x: self.item_weights[x],
                        bin_packing[bin])
                    ))
                if next_item_weight > self.capacity - bin_weight:
                    continue

                # put item in bin
                bin_packing[bin].add(next_item)
                item_packed = True
                break

            # create new bin if needed
            if not item_packed:
                new_bin = 1
                if len(bins) != 0:
                    new_bin = max(bins) + 1
                bin_packing[new_bin] = set()
                bin_packing[new_bin].add(next_item)
        return bin_packing

    def _filter_items(self, bin_packing, last_item_idx):
        # remove items that have not been considered yet
        considered_items = set(map(
            lambda x: x[1], self.sorted_item_weights[0:last_item_idx + 1]))
        new_bin_packing = {}
        for bin in bin_packing.keys():
            new_bin = set(filter(
                lambda x: x in considered_items, bin_packing[bin]))
            if len(new_bin) != 0:
                new_bin_packing[bin] = new_bin
        return new_bin_packing

    def lbound(self, sol):
        bin_packing = sol[0]
        last_item_idx = sol[1]

        # remove items that have not been considered yet
        bin_packing = self._filter_items(bin_packing, last_item_idx)
        curr_bin_ct = len(bin_packing.keys())

        # get free capacity in bin packing
        curr_weight = sum(list(map(
            lambda x: self.sorted_item_weights[x][0],
            list(range(last_item_idx + 1))
        )))
        free_capacity = self.capacity * curr_bin_ct - curr_weight

        # get weights of remaining items
        rem_weight = sum(list(map(
            lambda x: self.sorted_item_weights[x][0],
            list(range(last_item_idx + 1, len(self.sorted_item_weights)))
        )))        

        return curr_bin_ct + math.ceil(
            (rem_weight - free_capacity) / self.capacity)

    def cost(self, sol):
        bin_packing = sol[0]
        return len(bin_packing.keys())

    def branch(self, sol):
        # determine next item and its weight
        bin_packing = sol[0]
        last_item_idx = sol[1]
        next_item_idx = last_item_idx + 1
        if next_item_idx >= len(self.sorted_item_weights):
            return []
        next_item_weight, next_item = self.sorted_item_weights[
            next_item_idx]

        # remove items that have not been considered yet
        bin_packing = self._filter_items(bin_packing, last_item_idx)

        # pack items in bins
        new_sols = []
        extra_bin = 1
        if len(bin_packing.keys()) != 0:
            extra_bin = max(bin_packing.keys()) + 1
        bins = set(bin_packing.keys()).union({extra_bin})
        for bin in bins:
            # create new bin if considering new bin index
            new_bin_packing = copy.deepcopy(bin_packing)
            if bin not in new_bin_packing.keys():
                new_bin_packing[bin] = set()

            # check if bin has space
            bin_weight = sum(
                list(map(
                    lambda x: self.item_weights[x],
                    new_bin_packing[bin])
                ))
            if next_item_weight > self.capacity - bin_weight:
                continue

            # pack item in bin
            new_bin_packing[bin].add(next_item)
            new_sols.append((new_bin_packing, next_item_idx))
        return new_sols

    def is_feasible(self, sol):
        bin_packing = sol[0]

        # check that packed items are valid
        items = set(reduce(
            (lambda s1, s2: s1.union(s2)),
            list(map(lambda b: bin_packing[b], bin_packing.keys()))
        ))
        if len(items.difference(set(range(1, len(self.item_weights)+1)))) != 0:
            return False

        # check that for each bin, the weight is not exceeded
        for bin in bin_packing.keys():
            bin_weight = sum(
                list(map(lambda x: self.item_weights[x], bin_packing[bin])))
            if bin_weight > self.capacity:
                return False

        return True

    def is_complete(self, sol):
        bin_packing = sol[0]

        # check that all items are packed
        items = set(reduce(
            (lambda s1, s2: s1.union(s2)),
            list(map(lambda b: bin_packing[b], bin_packing.keys()))
        ))
        if items != set(range(1, len(self.item_weights)+1)):
            return False

        # check that for each bin, the weight is not exceeded
        for bin in bin_packing.keys():
            bin_weight = sum(
                list(map(lambda x: self.item_weights[x], bin_packing[bin])))
            if bin_weight > self.capacity:
                return False

        return True

    def complete_solution(self, sol):
        return (self._pack_rem_items(copy.deepcopy(sol[0]), sol[1]), sol[1])


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


def test_is_sol():
    TEST_CASES = [
        ([1, 2, 3], 3)
    ]
    for weights, capacity in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check valid solutions
        assert bpp.is_sol(({1: {1, 2}, 2: {3}}, -1))
        assert bpp.is_sol(({1: {3}, 2: {1, 2}}, -1))
        assert bpp.is_sol(({1: {3}, 2: {2}, 3: {1}}, 1))
        assert bpp.is_sol(({1: {3}, 2: {2}, 3: {1}}, 1))

        # check invalid solutions
        assert not bpp.is_sol(({1: {1, 2, 3}}, -1))
        assert not bpp.is_sol(({1: {3, 1}, 2: {2}}, -1))
        assert not bpp.is_sol(({1: {3, 2}, 2: {1}}, 1))
        assert not bpp.is_sol(({1: {3, 2}, 2: {1, 4}}, 1))
        assert not bpp.is_sol(({1: {3}, 2: {1}}, 1))

    print('Is solution tests passed')


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


if __name__ == '__main__':
    # unit tests
    print('Unit Tests:')
    print('==============')
    test_constructor()
    test_is_feasible()
    test_is_complete()
    test_complete_solution()
    test_cost()
    test_lbound()
    test_branch()
    print('==============\n')

    # main test
    print('Test Cases:')
    print('==============')
    test_bnb_binpacking()
    print('==============')
