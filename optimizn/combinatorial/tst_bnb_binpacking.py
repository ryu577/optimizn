from branch_and_bound import BnBProblem
from functools import reduce
import copy


class BinPackingProblem(BnBProblem):
    '''
    Solution format:
    1. Allocation of items up to last allocated item to bins (dict)
    3. Last allocated item (int)

    Branching strategy:
    Each level of the solution space tree corresponds to an item. Each
    solution in a level corresponds to the item being placed in a bin
    that it can fit in. The remaining items can be put in bins in
    decreasing order of weight, into the first bin that can fit it.
    New bins created as needed
    '''
    def __init__(self, weights, capacity):
        self.item_weights = {}  # mapping of items to weights
        self.sorted_item_weights = []  # sorted (weight, item) tuples (desc)
        for i in range(len(weights)):
            self.item_weights[i + 1] = weights[i]
            self.sorted_item_weights.append((weights[i], i+1))
        self.sorted_item_weights.sort(reverse=True)
        self.capacity = capacity

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
                new_bin = max(bins) + 1
                bin_packing[new_bin].add(next_item)
        return bin_packing

    def _filter_items(self, bin_packing, last_item_idx):
        # remove items that have not been considered yet
        considered_items = set(map(
            lambda x: x[1], self.sorted_item_weights[0:last_item_idx + 1]))
        for bin in bin_packing.keys():
            bin_packing[bin] = set(filter(
                lambda x: x in considered_items, bin_packing[bin]))
        return bin_packing

    def lbound(self, sol):
        bin_packing = sol[0]
        last_item_idx = sol[1]

        # remove items that have not been considered yet
        bin_packing = self._filter_items(bin_packing, last_item_idx)
        curr_bin_ct = len(bin_packing.keys())

        # get weights of remaining items
        rem_weight = sum(list(
            lambda x: self.sorted_item_weights[x][1],
            list(range(last_item_idx + 1, len(self.sorted_item_weights)))
        ))

        return curr_bin_ct + rem_weight / self.capacity

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
        next_item_weight, next_item = self.sorted_item_weights[next_item_idx]

        # remove items that have not been considered yet
        bin_packing = self._filter_items(bin_packing, last_item_idx)

        new_sols = []
        extra_bin = 1
        if len(bin_packing.keys()) != 0:
            extra_bin = max(bin_packing.keys()) + 1
        bins = set(bin_packing.keys()).union({extra_bin})
        for bin in bins:
            # check if bin has space
            bin_weight = sum(
                list(map(
                    lambda x: self.item_weights[x],
                    bin_packing[bin])
                ))
            if next_item_weight > self.capacity - bin_weight:
                continue

            # pack item in bin
            new_bin_packing = copy.deepcopy(bin_packing)
            if bin not in new_bin_packing.keys():
                new_bin_packing[bin] = set()
            new_bin_packing[bin].add(next_item)
            new_bin_packing = self._pack_rem_items(
                new_bin_packing, next_item)
            new_sols.append((new_bin_packing, next_item))
        return new_sols

    def is_sol(self, sol):
        bin_packing = sol[0]

        # check that all items are packed
        items = set(reduce(
            (lambda s1, s2: s1.union(s2)),
            list(map(lambda b: bin_packing[b], bin_packing.keys()))
        ))
        if items != set(range(1, len(self.weights)+1)):
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
