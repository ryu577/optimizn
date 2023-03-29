from optimizn.combinatorial.tst_anneal.trav_salsmn import CityGraph
from optimizn.combinatorial.branch_and_bound import BnBProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np


class TravelingSalesmanProblem(BnBProblem):
    '''
    Solution format:
    (path, last_vert_idx)
    path - list of vertices
    last_vert_idx - index of last confirmed vertex, rest of vertices are added
    to path greedily

    Branching strategy:
    From a current solution, branch into x solutions, where
    x is the number of unconfirmed vertices that are reachable from the last
    confirmed vertex. Each branched solution corresponds to an extra confirmed
    vertex, reachable from the last confirmed vertex of the current solution
    '''
    def __init__(self, params):
        self.input_graph = params['input_graph']
        # sort all distance values, for computing lower bounds
        self.sorted_dists = list(self.input_graph.dists.flatten())
        self.sorted_dists.sort()
        self.sorted_dists = self.sorted_dists[
            self.input_graph.dists.shape[0]:]
        super().__init__(params)

    def _get_closest_vert(self, vert, visited):
        # get the unvisited vertex closest to the one provided
        min_vert = None
        min_dist = float('inf')
        dists = self.input_graph.dists[vert]
        for i in range(len(dists)):
            if i != vert and i not in visited and dists[i] < min_dist:
                min_vert = i
                min_dist = dists[i]
        return min_vert

    def _complete_path(self, path):
        # complete the path greedily, iteratively adding the unvisited vertex
        # closest to the last vertex in the accmulated path
        visited = set(path)
        while len(path) != self.input_graph.dists.shape[0]:
            if len(path) == 0:
                next_vert = 0
            else:
                last_vert_idx = 0 if len(path) == 0 else path[-1]
                next_vert = self._get_closest_vert(last_vert_idx, visited)
            visited.add(next_vert)
            path.append(next_vert)
        return path

    def get_candidate(self):
        # greedily assemble a path from scratch
        return (self._complete_path([]), -1)

    def cost(self, sol):
        # sum of distances between adjacent vertices in path, and from last
        # vertex to first vertex in path
        path = sol[0]
        path_cost = 0
        for i in range(len(path) - 1):
            path_cost += self.input_graph.dists[path[i], path[i + 1]]
        path_cost += self.input_graph.dists[path[len(path) - 1], path[0]]
        return path_cost

    def lbound(self, sol):
        # sum of distances between confirmed vertices and smallest distances
        # to account for edges between remaining vertices and start vertex
        path, last_vert_idx = sol
        lb_path_cost = 0
        num_cov_edges = 0
        for i in range(last_vert_idx):
            lb_path_cost += self.input_graph.dists[path[i], path[i + 1]]
            num_cov_edges += 1
        lb_path_cost += sum(self.sorted_dists[:len(path) - num_cov_edges])
        return lb_path_cost

    def is_sol(self, sol):
        # check path length and all vertices covered once
        path = sol[0]
        return len(path) == self.input_graph.num_cities and\
            len(path) == len(set(path))

    def branch(self, sol):
        # build the path from the last confirmed vertex, by creating a new
        # solution where each uncovered vertex is the next confirmed vertex
        path, last_vert_idx = sol
        main_path = path[0:last_vert_idx + 1]
        visited = set(main_path)
        new_sols = []
        for new_vert in range(self.input_graph.dists.shape[0]):
            if new_vert not in visited:
                new_path = self._complete_path(main_path + [new_vert])
                new_sols.append((new_path, last_vert_idx + 1))
        return new_sols


class MockCityGraph:
    def __init__(self, dists):
        self.dists = dists
        self.num_cities = len(dists)


def test_get_closest_vert():
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
        (0, {}, 3),
        (3, {0, 3}, 2),
        (2, {0, 3, 2}, 1),
        (1, {0, 3, 2, 1}, None)
    ]
    for vert, visited, close_vert in TEST_CASES:
        edge = tsp._get_closest_vert(vert, visited)
        assert edge == close_vert, f'Vertex mismatch: {edge} != {close_vert}'
    print('_get_closest_vert tests passed')


def test_is_sol():
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
        (([0, 1, 2, 3], -1), True),
        (([0, 2, 1, 3], -1), True),
        (([1, 0, 2, 3], 1), True),
        (([1, 2, 3], 1), False),
        (([1, 2, 3, 3], 1), False),
        (([1, 2, 3, 0, 1], 1), False),
    ]
    for sol, valid_sol in TEST_CASES:
        assert valid_sol == tsp.is_sol(sol), f'{sol} is solution: {valid_sol}'
    print('is_sol tests passed')


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
        (([0, 3, 2, 1], 0), 10),
        (([0, 3, 2, 1], -1), 10),
        (([0, 1, 2, 3], 1), 10),
        (([0, 1, 3, 2], 3), 12)
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
        (([0, 3, 2, 1], -1), 6),
        (([0, 3, 2, 1], 0), 6),
        (([0, 3, 2, 1], 1), 5),
        (([0, 3, 2, 1], 2), 5),
        (([0, 3, 2, 1], 3), 7),
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
        (([0, 3, 2, 1], -1), [
            ([0, 3, 2, 1], 0), ([1, 2, 0, 3], 0), ([2, 0, 3, 1], 0),
            ([3, 0, 2, 1], 0)
        ]),
        (([0, 2, 1, 3], 0), [
            ([0, 1, 2, 3], 1), ([0, 2, 3, 1], 1), ([0, 3, 2, 1], 1),
        ]),
        (([0, 1, 2, 3], 1), [
            ([0, 1, 2, 3], 2), ([0, 1, 3, 2], 2)
        ]),
        (([0, 1, 2, 3], 3), []),
        (([0, 1, 2, 3], 4), [])
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
    tsp = TravelingSalesmanProblem(params)
    sol = tsp.solve(1000, 100, 120)
    print('Extra library produced path: ', permutation)
    print('Distance of external-library-produced distance: ', distance)
    print('BnB-produced path: ', sol[0][0])
    print('Distance of BnB-produced path: ', sol[1])


if __name__ == '__main__':
    # unit tests
    print('Unit tests:')
    test_is_sol()
    test_get_closest_vert()
    test_complete_path()
    test_cost()
    test_lbound()
    test_branch()
    print('=================')
    test_bnb_tsp()
