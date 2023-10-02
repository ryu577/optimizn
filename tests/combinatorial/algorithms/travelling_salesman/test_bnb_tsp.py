from optimizn.combinatorial.algorithms.travelling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.algorithms.travelling_salesman.bnb_tsp import\
    TravelingSalesmanProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np


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
