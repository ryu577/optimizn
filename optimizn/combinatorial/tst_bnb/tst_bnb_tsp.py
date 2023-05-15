from copy import deepcopy
from optimizn.combinatorial.tst_anneal.trav_salsmn import CityGraph
from optimizn.combinatorial.branch_and_bound import BnBProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np


class TravelingSalesmanProblem(BnBProblem):
    def __init__(self, params):
        self.input_graph = params['input_graph']
        # sort all distance values, for computing lower bounds
        self.sorted_dists = list(self.input_graph.dists.flatten())
        self.sorted_dists.sort()
        self.sorted_dists = self.sorted_dists[
            self.input_graph.dists.shape[0]:]
        super().__init__(params)

    def _get_closest_city(self, city, visited):
        # get the unvisited city closest to the one provided
        min_city = None
        min_dist = float('inf')
        dists = self.input_graph.dists[city]
        for i in range(len(dists)):
            if i != city and i not in visited and dists[i] < min_dist:
                min_city = i
                min_dist = dists[i]
        return min_city

    def _complete_path(self, sol):
        # complete the path greedily, iteratively adding the unvisited city
        # closest to the last city in the accmulated path
        path = sol[0]
        visited = set(path)
        complete_path = deepcopy(path)
        while len(complete_path) != self.input_graph.dists.shape[0]:
            if len(complete_path) == 0:
                next_city = 0
            else:
                last_city_idx = 0 if len(complete_path) == 0 else\
                    complete_path[-1]
                next_city = self._get_closest_city(last_city_idx, visited)
            visited.add(next_city)
            complete_path.append(next_city)
        return complete_path

    def complete_solution(self, sol):
        # greedily complete the path using the remaining/unvisited cities
        return (self._complete_path(sol[0]), sol[1])

    def get_candidate(self):
        # greedily assemble a path from scratch
        return (self._complete_path([]), -1)

    def cost(self, sol):
        # sum of distances between adjacent cities in path, and from last
        # city to first city in path
        path = sol[0]
        path_cost = 0
        for i in range(self.input_graph.num_cities - 1):
            path_cost += self.input_graph.dists[path[i], path[i + 1]]
        path_cost += self.input_graph.dists[
            path[self.input_graph.num_cities - 1], path[0]]
        return path_cost

    def lbound(self, sol):
        # sum of distances between confirmed cities and smallest distances
        # to account for remaining cities and start city
        path = sol[0]
        branch_idx = sol[1]
        lb_path_cost = 0
        for i in range(branch_idx):
            lb_path_cost += self.input_graph.dists[path[i], path[i + 1]]
        if branch_idx + 1 == self.input_graph.num_cities:
            lb_path_cost += self.input_graph.dists[
                path[branch_idx], path[0]]
        else:
            lb_path_cost += sum(self.sorted_dists[
                :self.input_graph.num_cities - (branch_idx + 1)])
        return lb_path_cost

    def is_complete(self, sol):
        # check that all cities covered once and path length is equal to the
        # number of cities
        path = sol[0]
        return len(path) == self.input_graph.num_cities and\
            len(path) == len(set(path))

    def is_feasible(self, sol):
        # check that all cities are covered once, path length is less than
        # or equal to the number of cities, and branch index is valid
        path = sol[0]
        branch_idx = sol[1]
        return len(path) <= self.input_graph.num_cities and\
            len(path) == len(set(path)) and branch_idx < len(path)\
            and branch_idx >= -1

    def branch(self, sol):
        # build the path from the last confirmed city, by creating a new
        # solution where each uncovered city is the next confirmed city
        path = sol[0]
        branch_idx = sol[1]
        visited = set(path)
        new_sols = []
        for new_city in range(self.input_graph.dists.shape[0]):
            if new_city not in visited:
                new_sols.append((path + [new_city], branch_idx + 1))
        return new_sols


class MockCityGraph:
    def __init__(self, dists):
        self.dists = dists
        self.num_cities = len(dists)


def test_get_closest_city():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    TEST_CASES = [
        (0, {}, 3),
        (3, {0}, 2),
        (2, {0, 3}, 1),
        (1, {0, 3, 2}, None),
        (1, {0, 3, 2, 1}, None)
    ]
    for city, visited, close_city in TEST_CASES:
        tsp = TravelingSalesmanProblem(params)
        edge = tsp._get_closest_city(city, visited)
        assert edge == close_city, f'Vertex mismatch: {edge} != {close_city}'
    print('_get_closest_city tests passed')


def test_is_feasible():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([0, 1, 2, 3], True),
        ([1, 0, 2, 3], True),
        ([1, 2, 2], False),
        ([1, 2, 3], True),
        ([1, 2, 3, 3], False),
        ([1, 2, 3, 0, 1], False)
    ]
    for sol, valid_sol in TEST_CASES:
        assert valid_sol == tsp.is_feasible(sol), f'{sol} is solution: '\
            + f'{valid_sol}'
    print('is_feasible tests passed')


def test_is_complete():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([0, 1, 2, 3], True),
        ([1, 0, 2, 3], True),
        ([1, 2, 2], False),
        ([1, 2, 3], False),
        ([1, 2, 3, 3], False),
        ([1, 2, 3, 0, 1], False)
    ]
    for sol, valid_sol in TEST_CASES:
        assert valid_sol == tsp.is_complete(sol), f'{sol} is solution: '\
            + f'{valid_sol}'
    print('is_complete tests passed')


def test_complete_path():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([], [0, 3, 2, 1]),
        ([0], [0, 3, 2, 1]),
        ([0, 1], [0, 1, 2, 3]),
        ([1, 3], [1, 3, 0, 2]),
        ([0, 3, 2, 1], [0, 3, 2, 1])
    ]
    for path, complete_path in TEST_CASES:
        comp_path = tsp._complete_path(path)
        assert comp_path == complete_path, 'Completed paths do no match, '\
            + f'Expected: {complete_path}, Actual: {comp_path}'
    print('_complete_path tests passed')


def test_get_candidate():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    exp_init_sol = [0, 3, 2, 1]
    assert tsp.best_solution == exp_init_sol, 'Invalid initial solution. '\
        + f'Expected: {exp_init_sol}. Actual: {tsp.best_solution}'
    print('get_candidate tests passed')


def test_complete_solution():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([], [0, 3, 2, 1]),
        ([0], [0, 3, 2, 1]),
        ([0, 1], [0, 1, 2, 3]),
        ([1, 3], [1, 3, 0, 2]),
        ([0, 3, 2, 1], [0, 3, 2, 1])
    ]
    for path, complete_path in TEST_CASES:
        comp_path = tsp.complete_solution(path)
        assert comp_path == complete_path, 'Completed paths do no match, '\
            + f'Expected: {complete_path}, Actual: {comp_path}'
    print('complete_solution tests passed')


def test_cost():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([0, 3, 2, 1], 10),
        ([0, 1, 2, 3], 10),
        ([0, 1, 3, 2], 12)
    ]
    for sol, cost in TEST_CASES:
        sol_cost = tsp.cost(sol)
        assert sol_cost == cost, f'Incorrect cost. Expected: {cost}, '\
            + f'Actual: {sol_cost}'
    print('cost tests passed')


def test_lbound():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([0, 3, 2, 1], 10),
        ([0, 1, 2, 3], 10),
        ([0, 1, 3, 2], 12),
        ([0, 3], 5),
        ([0, 3, 2], 5),
        ([0], 6)
    ]
    for sol, lower_bound in TEST_CASES:
        lb = tsp.lbound(sol)
        assert lb == lower_bound, 'Incorrect lower bound. Expected: '\
            + f'{lower_bound}, Actual: {lb}'
    print('lbound tests passed')


def test_branch():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        ([0, 3, 2, 1], []),
        ([0, 2, 1, 3], []),
        ([0, 1], [[0, 1, 2], [0, 1, 3]]),
        ([0, 1, 2], [[0, 1, 2, 3]]),
        ([1], [[1, 0], [1, 2], [1, 3]])
    ]
    for sol, branch_sols in TEST_CASES:
        new_sols = tsp.branch(sol)
        assert branch_sols == new_sols, 'Incorrect branched solutions for '\
            + f'solution: {sol}. Expected: {branch_sols}, Actual: {new_sols}'
    print('branch tests passed')


def test_bnb_tsp():
    graph = CityGraph()
    permutation, distance = solve_tsp_simulated_annealing(
        graph.dists, perturbation_scheme='ps2', alpha=0.99)
    params = {
        'input_graph': graph,
    }
    tsp1 = TravelingSalesmanProblem(params)
    sol1 = tsp1.solve(1e20, 1e20, 120, 0)
    tsp2 = TravelingSalesmanProblem(params)
    sol2 = tsp2.solve(1e20, 1e20, 120, 1)
    print('Extra library produced path: ', permutation)
    print('Distance of external-library-produced distance: ', distance)
    print('Traditional-BnB produced path: ', sol1[0])
    print(f'Iters: {tsp1.total_iters}')
    print('Distance of Traditional-BnB produced path: ', sol1[1])
    print('Modified-BnB-produced path: ', sol2[0])
    print(f'Iters: {tsp2.total_iters}')
    print('Distance of Modified-BnB-produced path: ', sol2[1])


if __name__ == '__main__':
    # unit tests
    print('Unit tests:')
    test_get_candidate()
    test_is_complete()
    test_is_feasible()
    test_get_closest_city()
    test_complete_path()
    test_complete_solution()
    test_cost()
    test_lbound()
    test_branch()
    print('=================')
    test_bnb_tsp()
