from optimizn.combinatorial.branch_and_bound import BnBProblem
from functools import reduce
import copy
import math


class BinPackingParams:
    def __init__(self, weights, capacity, iters_limit=1000, print_iters=200, 
                 time_limit=300):
        self.weights = weights
        self.capacity = capacity
        self.iters_limit = iters_limit
        self.print_iters = print_iters
        self.time_limit = time_limit

    def __eq__(self, other):
        return (
            other is not None
            and self.weights == other.weights
            and self.capacity == other.capacity
            and self.iters_limit == other.iters_limit
            and self.print_iters == other.print_iters
            and self.time_limit == other.time_limit
        )


# References:
# http://www.or.deis.unibo.it/knapsack.html (See PDF for Chapter 8,
#   http://www.or.deis.unibo.it/kp/Chapter8.pdf)
class BinPackingProblem1D(BnBProblem):
    '''
    Solution format:
    1. Allocation of items up to last allocated item to bins (dict)
    2. Last allocated item (int)

    Branching strategy:
    Each level of the solution space tree corresponds to an item. Each
    solution in a level corresponds to the item being placed in a bin
    that it can fit in. The remaining items can be put in bins in
    decreasing order of weight, into the first bin that can fit it.
    New bins created as needed
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
        return (self._pack_rem_items({}, -1), -1)

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

        # get weights of remaining items
        rem_weight = sum(list(map(
            lambda x: self.sorted_item_weights[x][0],
            list(range(last_item_idx + 1, len(self.sorted_item_weights)))
        )))        

        return curr_bin_ct + math.ceil(rem_weight / self.capacity)

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
            new_bin_packing = self._pack_rem_items(
                new_bin_packing, next_item_idx)
            new_sols.append((new_bin_packing, next_item_idx))
        return new_sols

    def is_sol(self, sol):
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
                list(map(
                    lambda x: self.item_weights[x],
                    bin_packing[bin]
                )))
            if bin_weight > self.capacity:
                return False

        return True


def test_param_equality():
    TEST_CASES = [
        (
            BinPackingParams([1, 2, 3, 4], [6]),
            None,
            False
        ),
        (
            None,
            BinPackingParams([1, 2, 3, 4], [6]),
            False
        ),
        (
            BinPackingParams([1, 2, 3, 4], [6]),
            BinPackingParams([1, 2, 3, 4], [6], 1000, 200, 300),
            True
        ),
        (
            BinPackingParams([1, 2, 3, 4], [6], iters_limit=1100),
            BinPackingParams([1, 2, 3, 4], [6], 1000, 200, 300),
            False
        ),
        (
            BinPackingParams([1, 2, 3, 4], [6], print_iters=300),
            BinPackingParams([1, 2, 3, 4], [6], 1000, 300, 300),
            True
        ),
        (
            BinPackingParams([1, 2, 3, 4], [6], 1200, 400, 300),
            BinPackingParams([1, 2, 3, 4], [6], 1000, 300, 300),
            False
        )
    ]
    for params1, params2, equal in TEST_CASES:
        assert (params1 == params2) == equal


def test_constructor():
    TEST_CASES = [
        ([1, 2, 3], 3, {1: {3}, 2: {1, 2}}),
        ([7, 8, 2, 3], 15, {1: {2, 1}, 2: {4, 3}})
    ]
    ITERS_LIMIT = 1000
    PRINT_ITERS = 200
    TIME_LIMIT = 300
    for weights, capacity, expected in TEST_CASES:
        params = BinPackingParams(
            weights,
            capacity,
            iters_limit=ITERS_LIMIT,
            print_iters=PRINT_ITERS,
            time_limit=TIME_LIMIT
        )
        bpp = BinPackingProblem1D(params)

        # check iters_limit, print_iters, and time_limit
        assert bpp.iters_limit == ITERS_LIMIT
        assert bpp.print_iters == PRINT_ITERS
        assert bpp.time_limit == TIME_LIMIT

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


def test_is_sol():
    TEST_CASES = [
        ([1, 2, 3], 3)
    ]
    for weights, capacity in TEST_CASES:
        params = BinPackingParams(
            weights,
            capacity,
            iters_limit=1000,
            print_iters=200,
            time_limit=300
        )
        bpp = BinPackingProblem1D(params)

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
        ([1, 2, 3], 3)
    ]
    for weights, capacity in TEST_CASES:
        params = BinPackingParams(
            weights,
            capacity,
            iters_limit=1000,
            print_iters=200,
            time_limit=300
        )
        bpp = BinPackingProblem1D(params)

        # check cost
        assert bpp.cost(({1: {3}, 2: {1, 2}}, -1)) == 2
        assert bpp.cost(({1: {1, 2}, 2: {3}}, -1)) == 2
        assert bpp.cost(({1: {2}, 2: {3}, 3: {1}}, 1)) == 3
        assert bpp.cost(({1: {3}, 2: {2}, 3: {1}}, 1)) == 3

    print('Cost tests passed')


def test_lbound():
    TEST_CASES = [
        ([1, 2, 3], 3)
    ]
    for weights, capacity in TEST_CASES:
        params = BinPackingParams(
            weights,
            capacity
        )
        bpp = BinPackingProblem1D(params)

        # check lower bounds
        assert bpp.lbound(({1: {3}, 2: {1, 2}}, -1)) <= 2
        assert bpp.lbound(({1: {3}, 2: {1, 2}}, 0)) <= 2
        assert bpp.lbound(({1: {3}, 2: {1}, 3: {2}}, 0)) <= 2
        assert bpp.lbound(({1: {3}, 2: {1}, 3: {2}}, 1)) <= 3
        assert bpp.lbound(({1: {3}, 2: {1}, 3: {2}}, 2)) <= 3

    print('Lower bound tests passed')


def test_branch():
    TEST_CASES = [
        ([1, 2, 3], 3, [
                ({1: {3}, 2: {2, 1}}, 2),
                ({1: {3}, 2: {2}, 3: {1}}, 2),
            ],
            ({1: {3}, 2: {2, 1}}, 1)),
        ([7, 8, 2, 3], 15, [
                ({1: {2, 1}, 2: {4, 3}}, 1),
                ({1: {2, 4, 3}, 2: {1}}, 1),
            ],
            ({1: {2, 1}, 2: {4, 3}}, 0)),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12, [
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 4}, 4: {7, 5}, 5: {6, 1}}, 5),
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 4}, 4: {7, 1}, 5: {6, 5}}, 5),
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 4}, 4: {7, 1}, 5: {6}, 6: {5}}, 5)
            ],
            ({1: {10}, 2: {9}, 3: {8}, 4: {7}, 5: {6}, 6: {5, 4}, 7: {2, 1}}, 4)),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12, [
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 4}, 4: {7, 1}, 5: {6}, 6: {5}}, 6),
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 1}, 4: {7, 4}, 5: {6}, 6: {5}}, 6),
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 1}, 4: {7}, 5: {6, 4}, 6: {5}}, 6),
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 1}, 4: {7}, 5: {6}, 6: {5, 4}}, 6),
                ({1: {10, 2}, 2: {9, 3}, 3: {8, 1}, 4: {7}, 5: {6}, 6: {5}, 7: {4}}, 6)
            ],
            ({1: {10}, 2: {9}, 3: {8}, 4: {7}, 5: {6}, 6: {5, 4}, 7: {2, 1}}, 5))
    ]
    for weights, capacity, expected, init_sol in TEST_CASES:
        # check branch
        params = BinPackingParams(
            weights,
            capacity
        )
        bpp = BinPackingProblem1D(params)
        new_sols = bpp.branch(init_sol)
        assert new_sols == expected

    print('Branch tests passed')


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
        print('-----------------')
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem1D(params)
        print('Sorted item weights (w, i):', bpp.sorted_item_weights)
        print('Item weights:', bpp.item_weights)
        bpp.solve()
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
    test_is_sol()
    test_cost()
    test_lbound()
    test_branch()
    print('==============\n')

    # main test
    print('Test Cases:')
    print('==============')
    test_bnb_binpacking()
    print('==============')
