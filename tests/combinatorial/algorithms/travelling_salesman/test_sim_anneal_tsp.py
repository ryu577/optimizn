# TODO: fix ModuleNotFound errors, uncomment below code

# # pip install python-tsp
# # https://github.com/fillipe-gsm/python-tsp
# from python_tsp.heuristics import solve_tsp_local_search
# ## https://developers.google.com/optimization/routing/tsp
# # Their solution didn't work, some cpp error.
# from optimizn.combinatorial.algorithms.travelling_salesman.city_graph\
#     import CityGraph
# from optimizn.combinatorial.algorithms.travelling_salesman.sim_anneal_tsp\
#     import TravSalsmn


# def tst1():
#     # import optimizn.combinatorial.tst_anneal.trav_salsmn as ts
#     tt = CityGraph()
#     ts1 = TravSalsmn(tt)
#     print("Best solution with external library: ")
#     #permutation, distance = solve_tsp_dynamic_programming(tt.dists)
#     permutation, distance = solve_tsp_local_search(tt.dists)
#     print(distance)
#     ts1.anneal()
#     print("Best solution: " + str(ts1.best_cost))
#     return ts1
