from optimizn.combinatorial.tst_anneal.trav_salsmn import CityGraph
from optimizn.combinatorial.branch_and_bound import BnBProblem
from python_tsp.heuristics import solve_tsp_local_search
import numpy as np


class TravelingSalesmanProblem(BnBProblem):
    '''
    Solution format:
    (path, last_edge)
    path - list of 2-tuples, represents a sequential ordering of edges
    last_edge_idx - index of last edge added to path before solution was 
                    completed by greedy technique (least-cost edges to 
                    unvisited vertices)
    
    Branching strategy:
    From the initial solution, branch into x solutions, where x is the number
    of edges that contain the first vertex. Each solution corresponds to
    one of these edges being chosen in the path, and the path is completed
    via a greedy technique (iteratively choose the least-cost edges to
    unvisited vertices). Branching continues on these solutions, and so on
    '''
    def __init__(self, input_graph, iters_limit, print_iters, time_limit):
        self.input_graph = input_graph
        self.sorted_dists = []
        for i in range(self.input_graph.dists.shape[0]):
            for j in range(self.input_graph.dists.shape[1]):
                if i != j:
                    self.sorted_dists.append(self.input_graph.dists[i, j])
        self.sorted_dists.sort()
        super().__init__(
            name='TravelingSalesmanProblem',
            iters_limit=iters_limit,
            print_iters=print_iters,
            time_limit=time_limit)

    def _get_closest_vert(self, vert, visited):
        dists = self.input_graph.dists[vert]
        min_edge = None
        min_dist = float('inf')
        if len(visited) == self.input_graph.dists.shape[0]:
            visited -= {0}
        for i in range(len(dists)):
            if i != vert and i not in visited and dists[i] < min_dist:
                min_edge = (vert, i)
                min_dist = dists[i]
        return min_edge

    def _complete_path(self, path):
        visited = set()
        for v1, v2 in path:
            visited.add(v1)
            visited.add(v2)
        while len(path) != self.input_graph.dists.shape[0]:
            last_vert = 0 if len(path) == 0 else path[-1][1]
            next_edge = self._get_closest_vert(last_vert, visited)
            visited.add(next_edge[0])
            visited.add(next_edge[1])
            path.append(next_edge)
        return path

    def get_candidate(self):
        return (self._complete_path([]), -1)

    def cost(self, sol):
        path = sol[0]
        path_cost = 0
        for v1, v2 in path:
            path_cost += self.input_graph.dists[v1, v2]
        return path_cost

    def lbound(self, sol):
        path = sol[0]
        last_edge_idx = sol[1]
        path_cost = 0
        for i in range(0, last_edge_idx + 1):
            v1, v2 = path[i]
            path_cost += self.input_graph.dists[v1, v2]
        for j in range(0, len(path) - last_edge_idx - 1):
            path_cost += self.sorted_dists[j]
        return path_cost

    def is_sol(self, sol):
        path = sol[0]
        counts = dict()
        for edge in path:
            for vert in edge:
                if vert in counts.keys():
                    counts[vert] += 1
                else:
                    counts[vert] = 1

        # check if all vertices visited
        all_verts_visited = counts.keys() == set(range(
            self.input_graph.dists.shape[0]))

        # check if path forms cycle
        path_is_cycle = True
        for vert in counts.keys():
            if counts[vert] != 2:
                return False
        return all_verts_visited and path_is_cycle

    def branch(self, sol):
        path = sol[0]
        last_edge_idx = sol[1]
        main_path = path[0:last_edge_idx + 1]
        last_vert = 0 if len(main_path) == 0 else main_path[-1][1]
        visited = set()
        for v1, v2 in main_path:
            visited.add(v1)
            visited.add(v2)
        new_sols = []
        for new_vert in range(self.input_graph.dists.shape[0]):
            if new_vert not in visited and new_vert != last_vert:
                new_path = main_path + [(last_vert, new_vert)]
                new_path = self._complete_path(new_path)
                new_sols.append((new_path, last_edge_idx + 1))
        return new_sols


class MockCityGraph():
    def __init__(self, dists):
        self.dists = dists


def test_get_closest_vert():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    tsp = TravelingSalesmanProblem(mcg, 1000, 100, 600)
    TEST_CASES = [
        (0, {}, (0, 3)),
        (3, {0, 3}, (3, 2)),
        (2, {0, 3, 2}, (2, 1)),
        (1, {0, 3, 2, 1}, (1, 0))
    ]
    for vert, visited, close_edge in TEST_CASES:
        edge = tsp._get_closest_vert(vert, visited)
        assert edge == close_edge, f'Edge mismatch: {edge} != {close_edge}'
    print('_get_closest_vert tests passed')


def test_is_sol():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    tsp = TravelingSalesmanProblem(mcg, 1000, 100, 600)
    TEST_CASES = [
        (([(0, 1), (1, 2), (2, 3), (3, 0)], -1), True),
        (([(0, 2), (2, 1), (1, 3), (3, 0)], -1), True),
        (([(1, 0), (0, 2), (2, 3), (3, 1)], 1), True),
        (([(1, 0), (0, 2), (2, 3), (3, 0)], 2), False),
        (([(1, 2), (2, 3), (3, 1)], 1), False),
        (([(1, 2), (2, 3), (3, 0)], 0), False),
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
    tsp = TravelingSalesmanProblem(mcg, 1000, 100, 600)
    TEST_CASES = [
        ([], [(0, 3), (3, 2), (2, 1), (1, 0)]),
        ([(0, 3)], [(0, 3), (3, 2), (2, 1), (1, 0)]),
        ([(0, 1)], [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ([(0, 1), (1, 3)], [(0, 1), (1, 3), (3, 2), (2, 0)]),
        ([(0, 3), (3, 2), (2, 1), (1, 0)], [(0, 3), (3, 2), (2, 1), (1, 0)])
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
    tsp = TravelingSalesmanProblem(mcg, 1000, 100, 600)
    TEST_CASES = [
        (([(0, 3), (3, 2), (2, 1), (1, 0)], 0), 10),
        (([(0, 3), (3, 2), (2, 1), (1, 0)], -1), 10),
        (([(0, 1), (1, 2), (2, 3), (3, 0)], 1), 10),
        (([(0, 1), (1, 3), (3, 2), (2, 0)], 3), 12)
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
    tsp = TravelingSalesmanProblem(mcg, 1000, 100, 600)
    TEST_CASES = [
        (([(0, 3), (3, 2), (2, 1), (1, 0)], -1), 6),
        (([(0, 3), (3, 2), (2, 1), (1, 0)], 0), 5),
        (([(0, 3), (3, 2), (2, 1), (1, 0)], 1), 5),
        (([(0, 3), (3, 2), (2, 1), (1, 0)], 2), 7),
        (([(0, 3), (3, 2), (2, 1), (1, 0)], 3), 10),
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
    tsp = TravelingSalesmanProblem(mcg, 1000, 100, 600)
    TEST_CASES = [
        (([(0, 3), (3, 2), (2, 1), (1, 0)], -1), [
            ([(0, 1), (1, 2), (2, 3), (3, 0)], 0),
            ([(0, 2), (2, 3), (3, 1), (1, 0)], 0),
            ([(0, 3), (3, 2), (2, 1), (1, 0)], 0)
        ]),
        (([(0, 2), (2, 1), (1, 3), (3, 0)], 0), [
            ([(0, 2), (2, 1), (1, 3), (3, 0)], 1),
            ([(0, 2), (2, 3), (3, 1), (1, 0)], 1)
        ]),
        (([(0, 1), (1, 2), (2, 3), (3, 0)], 1), [
            ([(0, 1), (1, 2), (2, 3), (3, 0)], 2)
        ]),
        (([(0, 1), (1, 2), (2, 3), (3, 0)], 3), []),
        (([(0, 1), (1, 2), (2, 3), (3, 0)], 4), [])
    ]
    for sol, branch_sols in TEST_CASES:
        new_sols = tsp.branch(sol)
        assert branch_sols == new_sols, 'Incorrect branched solutions for '\
            + f'solution: {sol}. Expected: {branch_sols}, Actual: {new_sols}'
    print('branch tests passed')


def test_bnb_tsp():
    graph = CityGraph()
    permutation, distance = solve_tsp_local_search(graph.dists)

    tsp = TravelingSalesmanProblem(graph, 1000, 500, 600)
    sol = tsp.solve()
    print('BnB-produced path: ', sol[0][0])
    print('Distance of BnB-produced path: ', sol[1])
    print('Distance of external-library-produced distance: ', distance)


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
