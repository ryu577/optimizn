from optimizn.combinatorial.branch_and_bound import BnBProblem
from copy import deepcopy


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
    
    def is_feasible(self, sol):
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

    def is_complete(self, sol):
        # return True, since all solutions are complete by design
        return True

    def complete_solution(self, sol):
        # return solution, since all solutions are complete by design
        return sol

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
