from copy import deepcopy
from optimizn.combinatorial.algorithms.travelling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.branch_and_bound import BnBProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np


class TravelingSalesmanProblem(BnBProblem):
    def __init__(self, params):
        self.input_graph = params['input_graph']
        # sort all distance values, for computing lower bounds
        self.sorted_dists = []
        for i in range(self.input_graph.dists.shape[0]):
            for j in range(0, i):
                self.sorted_dists.append(self.input_graph.dists[i, j])
        self.sorted_dists.sort()
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

    def _complete_path(self, path):
        # complete the path greedily, iteratively adding the unvisited city
        # closest to the last city in the accumulated path
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

    def get_candidate(self):
        # greedily assemble a path from scratch
        # solution format is 2-tuple, first element is the path itself and
        # the second element is the index of the last confirmed city (last
        # confirmed index), which is used for branching
        return (self._complete_path([]), -1)

    def complete_solution(self, sol):
        # greedily complete the path using the remaining/unvisited cities
        return (self._complete_path(sol[0]), sol[1])

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
        last_confirmed_idx = sol[1]
        lb_path_cost = 0
        for i in range(last_confirmed_idx):
            lb_path_cost += self.input_graph.dists[path[i], path[i + 1]]
        if last_confirmed_idx + 1 == self.input_graph.num_cities:
            lb_path_cost += self.input_graph.dists[
                path[last_confirmed_idx], path[0]]
        else:
            lb_path_cost += sum(self.sorted_dists[
                :self.input_graph.num_cities - last_confirmed_idx])
        return lb_path_cost

    def is_complete(self, sol):
        # check that all cities covered once, path length is equal to the
        # number of cities
        path = sol[0]
        check_all_cities_covered = set(path) == set(
            range(self.input_graph.num_cities))
        check_cities_covered_once = len(path) == len(set(path))
        check_path_length = len(path) == self.input_graph.num_cities
        return (check_path_length and check_cities_covered_once and
                check_all_cities_covered)

    def is_feasible(self, sol):
        # check that covered cities are valid, covered cities are only covered
        # once, path length is less than or equal to the number of cities, and
        # last confirmed index is valid
        path = sol[0]
        last_confirmed_idx = sol[1]
        check_covered_cities = len(set(path).difference(
                set(range(self.input_graph.num_cities)))) == 0
        check_cities_covered_once = len(path) == len(set(path))
        check_path_length = len(path) <= self.input_graph.num_cities
        check_last_confirmed_index = last_confirmed_idx < len(path)\
            and last_confirmed_idx >= -1
        return (check_covered_cities and check_cities_covered_once and
                check_path_length and check_last_confirmed_index)

    def branch(self, sol):
        # build the path from the last confirmed city, by creating a new
        # solution where each uncovered city is the next confirmed city
        path = sol[0]
        last_confirmed_idx = sol[1]
        if last_confirmed_idx >= self.input_graph.num_cities - 1:
            return []
        visited = set(path[:last_confirmed_idx + 1])
        new_sols = []
        for new_city in range(self.input_graph.dists.shape[0]):
            if new_city not in visited:
                new_sols.append((path[:last_confirmed_idx + 1] + [new_city],
                                 last_confirmed_idx + 1))
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
        (([0, 1, 2, 3], 3), True),
        (([1, 0, 2, 3], 2), True),
        (([1, 0, 2, 3], 4), False),
        (([1, 2, 2], 2), False),
        (([1, 2, 3], 2), True),
        (([1, 2, 3, 3], 2), False),
        (([1, 2, 3, 0, 1], 3), False)
    ]
    for sol, valid_sol in TEST_CASES:
        assert valid_sol == tsp.is_feasible(sol), f'{sol} is feasible: '\
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
        (([0, 1, 2, 3], 3), True),
        (([0, 1, 2, 3], 2), True),
        (([1, 0, 2, 3], 3), True),
        (([1, 0, 2, 3], 2), True),
        (([1, 2], 1), False),
        (([1, 2, 3], 2), False),
        (([1, 2, 3, 0], 3), True),
        (([1, 2, 3, 0, 1], 3), False)
    ]
    for sol, valid_sol in TEST_CASES:
        assert valid_sol == tsp.is_complete(sol), f'{sol} is complete: '\
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


def test_get_candidate_sorted_dists():
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
    exp_init_sol = ([0, 3, 2, 1], -1)
    exp_sorted_dists = [1, 2, 2, 3, 4, 4]
    assert tsp.best_solution == exp_init_sol, 'Invalid initial solution. '\
        + f'Expected: {exp_init_sol}. Actual: {tsp.best_solution}'
    assert tsp.sorted_dists == exp_sorted_dists, 'Invalid sorted distances. '\
        + f'Expected: {tsp.sorted_dists}. Actual: {exp_sorted_dists}'
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
        (([], -1), ([0, 3, 2, 1], -1)),
        (([0], 0), ([0, 3, 2, 1], 0)),
        (([0, 1], 1), ([0, 1, 2, 3], 1)),
        (([1, 3], 1), ([1, 3, 0, 2], 1)),
        (([0, 3, 2, 1], 3), ([0, 3, 2, 1], 3))
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
        (([0, 3, 2, 1], 2), 10),
        (([0, 3, 2, 1], 3), 10),
        (([0, 1, 2, 3], 1), 10),
        (([0, 1, 3, 2], 0), 12)
    ]
    for sol, cost in TEST_CASES:
        sol_cost = tsp.cost(sol)
        assert sol_cost == cost, f'Incorrect cost for solution {sol}. '\
            + f'Expected: {cost}. Actual: {sol_cost}'
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
        (([0, 3, 2, 1], 3), 10),
        (([0, 3, 2, 1], 1), 6),
        (([0, 3, 2, 1], 2), 6),
        (([0, 1, 2, 3], 3), 10),
        (([0, 1, 3, 2], 3), 12),
        (([0, 3], 1), 6),
        (([0, 3, 2], 2), 6),
        (([0], 0), 8)
    ]
    for sol, lower_bound in TEST_CASES:
        lb = tsp.lbound(sol)
        assert lb == lower_bound, 'Incorrect lower bound for solution '\
            + f'{sol}. Expected: {lower_bound}. Actual: {lb}'
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
        (([0, 3, 2, 1], -1), [([0], 0), ([1], 0), ([2], 0), ([3], 0)]),
        (([0, 3, 2, 1], 3), []),
        (([0, 2, 1, 3], 4), []),
        (([0, 2, 1, 3], 1), [([0, 2, 1], 2), ([0, 2, 3], 2)]),
        (([0, 1], 1), [([0, 1, 2], 2), ([0, 1, 3], 2)]),
        (([0, 1, 2], 2), [([0, 1, 2, 3], 3)]),
        (([1], 0), [([1, 0], 1), ([1, 2], 1), ([1, 3], 1)])
    ]
    for sol, branch_sols in TEST_CASES:
        new_sols = tsp.branch(sol)
        assert branch_sols == new_sols, 'Incorrect branched solutions for '\
            + f'solution: {sol}. Expected: {branch_sols}, Actual: {new_sols}'
    print('branch tests passed')


def test_bnb_tsp():
    graph = CityGraph()
    params = {
        'input_graph': graph,
    }
    tsp1 = TravelingSalesmanProblem(params)
    permutation, distance = solve_tsp_simulated_annealing(
        graph.dists, x0=tsp1.best_solution[0], perturbation_scheme='ps2',
        alpha=0.99)
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
    test_get_candidate_sorted_dists()
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
