import numpy as np
from optimizn.combinatorial.branch_and_bound import BnBProblem


class KnapsackParams:
    def __init__(self, values, weights, capacity, init_sol):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.init_sol = init_sol

    def __eq__(self, other):
        return (
            other is not None
            and (len(self.values) == len(other.values)
                 and (self.values == other.values).all())
            and (len(self.weights) == len(other.weights)
                 and (self.weights == other.weights).all())
            and self.capacity == other.capacity
            and (len(self.init_sol[0]) == len(other.init_sol[0])
                 and (self.init_sol[0] == other.init_sol[0]).all())
            and self.init_sol[1] == self.init_sol[1]
        )


class ZeroOneKnapsackProblem(BnBProblem):
    '''
    Branch and bound implementation for the 0/1 knapsack problem, where each
    item is either taken or omitted in its entirety.

    This branch and bound implementation is based on the demonstration shown in
    the following source.

    Source:

    [1]
    Title: 7.2 0/1 Knapsack using Branch and Bound
    Author: Abdul Bari
    URL: https://www.youtube.com/watch?v=yV1d-b_NeK8
    Date published: February 26, 2018
    Date accessed: December 16, 2022
    '''
    def __init__(self, params):
        self.values = np.array(params.values)
        self.weights = np.array(params.weights)
        self.capacity = params.capacity

        # value/weight ratios, in decreasing order
        vw_ratios = self.values / self.weights
        vw_ratios_ixs = []
        for i in range(len(vw_ratios)):
            vw_ratios_ixs.append((vw_ratios[i], i))
        self.sorted_vw_ratios = sorted(vw_ratios_ixs)
        self.sorted_vw_ratios.reverse()
        self.init_sol = params.init_sol
        super().__init__(params)

    def get_candidate(self):
        return self.init_sol

    def lbound(self, sol):
        value = 0
        weight = 0

        # consider items already taken
        for i in range(0, sol[1] + 1):
            if sol[0][i] == 1:
                value += self.values[i]
                weight += self.weights[i]

        # greedily take other items
        for vw_ratio, ix in self.sorted_vw_ratios:
            if ix < sol[1] + 1:
                continue
            rem_cap = self.capacity - weight
            if rem_cap <= 0:
                break
            item_weight = min(rem_cap, self.weights[ix])
            value += item_weight * vw_ratio
            weight += item_weight

        return -1 * value

    def cost(self, sol):
        return -1 * np.sum(sol[0] * self.values)

    def branch(self, sol):
        exp_idx = sol[1] + 1
        if exp_idx >= len(self.weights):
            return []

        new_sols = []
        for val in [0, 1]:
            new_sol = np.zeros(len(sol[0]))
            new_sol[0:exp_idx] = sol[0][0:exp_idx]
            new_sol[exp_idx] = val
            weight = np.sum(new_sol * self.weights)

            # greedily take other items
            for _, ix in self.sorted_vw_ratios:
                if ix < exp_idx + 1:
                    continue
                rem_cap = self.capacity - weight
                if rem_cap <= 0:
                    break
                if self.weights[ix] <= rem_cap:
                    new_sol[ix] = 1
                    weight += self.weights[ix]

            new_sols.append((new_sol, exp_idx))
        return new_sols

    def is_feasible(self, sol):
        # check that array is not longer than the number of weights/values
        check_length1 = len(sol[0]) <= len(self.weights)
        check_length2 = len(sol[0]) <= len(self.values)
        check_length = check_length1 and check_length2

        # check that the only values in the array are 0 and 1
        check_values = len(set(sol[0].tolist()).difference({0, 1})) == 0

        # check that the weight of the values in the array is not greater
        # than the capacity
        check_weight = np.sum(sol[0] * self.weights) <= self.capacity

        return check_length and check_values and check_weight

    def is_complete(self, sol):
        # check that array length is the same as the number of weights/values
        check_length1 = len(sol[0]) == len(self.weights)
        check_length2 = len(sol[0]) == len(self.values)
        check_length = check_length1 and check_length2

        return check_length

    def complete_solution(self, sol):
        # greedily add other items to array
        knapsack = list(sol[0])
        value = 0
        weight = 0
        for i in range(len(knapsack)):
            if knapsack[i] == 1:
                value += self.values[i]
                weight += self.weights[i]

        # greedily take other items
        for _, ix in self.sorted_vw_ratios:
            if ix < sol[1] + 1:
                continue
            rem_cap = self.capacity - weight
            if rem_cap < self.weights[ix]:
                knapsack.append(0)
                continue
            value += self.values[ix]
            weight += self.weights[ix]
            knapsack.append(1)
        
        return (np.array(knapsack), sol[1])
