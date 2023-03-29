from optimizn.combinatorial.tst_anneal.trav_salsmn import CityGraph, TravSalsmn
from optimizn.combinatorial.tst_bnb.tst_bnb_tsp import TravelingSalesmanProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
from optimizn.combinatorial.opt_problem import load_latest_pckl
import time
import sys


def run_tsp_experiments(num_cities=50, compute_time_mins=1, num_trials=3):
    # specify maximum number of iterations
    MAX_ITERS = sys.maxsize  # int max, highest possible bound on iterations
    # since algorithms should use up all compute time

    # for collecting results
    results = dict()
    results['sa1'] = []
    results['sa1_time'] = []
    results['bnb'] = []
    results['bnb_time'] = []
    results['sa2'] = []
    results['sa2_time'] = []

    # create traveling salesman problem parameters
    city_graph = CityGraph(num_cities)

    # run simulated annealing 1
    tsp_sa = TravSalsmn(city_graph)
    s = time.time()
    tsp_sa.anneal(n_iter=MAX_ITERS, time_limit=compute_time_mins * 60)
    e = time.time()
    tsp_sa.persist()
    results['sa1'].append(tsp_sa.best_cost)
    results['sa1_time'].append(e - s)

    # run branch and bound
    tsp_bnb = TravelingSalesmanProblem({'input_graph': city_graph})
    s = time.time()
    tsp_bnb.solve(iters_limit=MAX_ITERS, print_iters=200,
                  time_limit=compute_time_mins * 60)
    e = time.time()
    tsp_bnb.persist()
    results['bnb'].append(tsp_bnb.best_cost)
    results['bnb_time'].append(e - s)

    # run simulated annealing 2
    opt_permutation = None
    opt_dist = float('inf')
    s = time.time()
    e = time.time()
    while (e - s) < (compute_time_mins * 60):
        permutation, distance = solve_tsp_simulated_annealing(
            city_graph.dists,
            max_processing_time=(compute_time_mins * 60) - (e - s),
            alpha=0.99, x0=opt_permutation, perturbation_scheme='ps2')
        if opt_dist > distance:
            opt_dist = distance
            opt_permutation = permutation
        e = time.time()
    results['sa2'].append(opt_dist)
    results['sa2_time'].append(e - s)

    # repeat run from previous solutions, for remaining trials
    for _ in range(num_trials - 1):
        # run simulated annealing 1
        tsp_sa = load_latest_pckl(path1='Data/TravSalsmn/DailyOpt')
        if tsp_sa is None:
            raise Exception('No saved instance for TSP simulated annealing')
        s = time.time()
        tsp_sa.anneal(n_iter=MAX_ITERS, time_limit=compute_time_mins * 60)
        e = time.time()
        tsp_sa.persist()
        results['sa1'].append(tsp_sa.best_cost)
        results['sa1_time'].append(e - s)

        # run branch and bound
        tsp_bnb = load_latest_pckl(
            path1='Data/TravelingSalesmanProblem/DailyOpt')
        if tsp_bnb is None:
            raise Exception('No saved instance for TSP branch and bound')
        s = time.time()
        tsp_bnb.solve(iters_limit=MAX_ITERS, print_iters=200,
                      time_limit=compute_time_mins * 60)
        e = time.time()
        tsp_bnb.persist()
        results['bnb'].append(tsp_bnb.best_cost)
        results['bnb_time'].append(e - s)

        # run simulated annealing 2
        s = time.time()
        e = time.time()
        while (e - s) < (compute_time_mins * 60):
            permutation, distance = solve_tsp_simulated_annealing(
                city_graph.dists,
                max_processing_time=(compute_time_mins * 60) - (e - s),
                alpha=0.99, x0=opt_permutation, perturbation_scheme='ps2')
            if opt_dist > distance:
                opt_dist = distance
                opt_permutation = permutation
            e = time.time()
        results['sa2'].append(opt_dist)
        results['sa2_time'].append(e - s)

    # return results
    return results


if __name__ == '__main__':
    exp1_results = run_tsp_experiments(50, 1, 3)
    exp2_results = run_tsp_experiments(100, 2, 3)
    exp3_results = run_tsp_experiments(200, 4, 3)

    # print results
    results = [exp1_results, exp2_results, exp3_results]
    for i in range(len(results)):
        print(f'Results for Experiment {i}:\n')
        print(f'Simulated annealing:\n{results[i]["sa"]}')
        print(f'Branch and bound:\n{results[i]["bnb"]}')
        print(f'Local search heuristic:\n{results[i]["ls"]}')
