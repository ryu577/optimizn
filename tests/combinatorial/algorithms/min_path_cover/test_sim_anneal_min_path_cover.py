# TODO: fix ModuleNotFound errors, uncomment below code

# import numpy as np
# from graphing.special_graphs.neural_trigraph.neural_trigraph\
#     import NeuralTriGraph
# from graphing.special_graphs.neural_trigraph.rand_graph import rep_graph
# from graphing.special_graphs.neural_trigraph.path_cover import \
#     min_cover_trigraph
# import os
# from optimizn.combinatorial.algorithms.min_path_cover\
#     .sim_anneal_min_path_cover import MinPathCover_NTG


# def test_1(edges1=None, edges2=None, n_iter=20000, swtch=1):
#     # edges1, edges2 = neur_trig_edges(8, 10, 14)
#     if edges1 is None:
#         edges1, edges2 = rep_graph(8, 10, 14, reps=4)
#     opt_paths = min_cover_trigraph(edges1, edges2)
#     print("Optimal solution: " + str(len(opt_paths)))
#     ntg = NeuralTriGraph(edges1, edges2)
#     # print(ntg.g1.adj)
#     mpc = MinPathCover_NTG(ntg, swtch=swtch)
#     paths = mpc.get_candidate()
#     # mpc.candidate = np.concatenate((mpc.candidate, mpc.candidate))
#     print("Current solution: " + str(len(mpc.candidate)))
#     mpc.anneal(n_iter)
#     print("Best solution: " + str(mpc.best_cost))
#     # Now, can we get the min path cover for this?
#     return mpc


# def test_2(n_iter=20000, swtch=1):
#     dirname = os.path.dirname(__file__)
#     edges1_path = os.path.join(dirname, './edges1.csv')
#     edges2_path = os.path.join(dirname, './edges2.csv')
#     edges1 = np.loadtxt(edges1_path)
#     edges1 = edges1.astype(int)
#     edges2 = np.loadtxt(edges2_path)
#     edges2 = edges2.astype(int)
#     return test_1(edges1, edges2, n_iter, swtch=swtch)
