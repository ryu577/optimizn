# TODO: fix ModuleNotFound errors, uncomment below code

# import numpy as np
# from graphing.special_graphs.neural_trigraph.rand_graph import rep_graph
# from graphing.special_graphs.neural_trigraph.path_cover import \
#     min_cover_trigraph
# from optimizn.combinatorial.algorithms.min_path_cover.bnb_min_path_cover\
#     import MinPathCoverParams, MinPathCoverProblem1, MinPathCoverProblem2


# def test_bnb_minpathcover():
#     EDGES = [
#         (
#             np.array([[1, 4], [2, 4], [2, 5], [3, 5]]),
#             np.array([[4, 6], [4, 7], [5, 8]])
#         ),
#         rep_graph(8, 10, 14, reps=4),
#         # NOTE: uncomment to run below test cases
#         # rep_graph(10, 14, 10, reps=4),
#         # rep_graph(20, 40, 20, reps=4),
#         # rep_graph(20, 40, 20, reps=8),
#         # rep_graph(40, 50, 60, reps=10)
#     ]
#     LENGTHS = [
#         3,
#         len(min_cover_trigraph(EDGES[1][0], EDGES[1][1])),
#         # NOTE: uncomment to run below test cases
#         # len(min_cover_trigraph(EDGES[2][0], EDGES[2][1])),
#         # len(min_cover_trigraph(EDGES[3][0], EDGES[3][1])),
#         # len(min_cover_trigraph(EDGES[4][0], EDGES[4][1])),
#         # len(min_cover_trigraph(EDGES[5][0], EDGES[5][1])),
#     ]
#     for i in range(len(EDGES)):
#         for bnb_type in [0, 1]:
#             print('\n=============================')
#             print(f'TEST CASE {i}')
#             edges1 = EDGES[i][0]
#             edges2 = EDGES[i][1]

#             # first approach
#             params = MinPathCoverParams(edges1, edges2)
#             mpc1 = MinPathCoverProblem1(params)
#             sol1, scr1 = mpc1.solve(1000, 100, 120, bnb_type)

#             # second approach
#             mpc2 = MinPathCoverProblem2(params)
#             sol2, scr2 = mpc2.solve(1000, 100, 120, bnb_type)

#             if bnb_type == 0:
#                 print('\nFirst Approach (Traditional BnB):')
#             else:
#                 print('\nFirst Approach (Modified BnB):')
#             solution1 = []
#             for j in range(len(sol1[0].astype(int))):
#                 if sol1[0][j] == 1:
#                     solution1.append(mpc1.all_paths[j])
#             solution1 = np.array(solution1)
#             print(f'Score: {scr1}\nSolution: {solution1}')
#             if len(solution1) != LENGTHS[i]:
#                 print(f'Paths: {len(solution1)} '
#                     + f'Optimal number of paths: {LENGTHS[i]}')
#             else:
#                 print(f'Optimal number of paths reached: {len(solution1)}')
#             assert set(solution1.flatten()) == mpc1.vertices, \
#                 'Not all vertices covered'
#             print('All vertices covered')

#             if bnb_type == 0:
#                 print('\nSecond Approach (Traditional BnB):')
#             else:
#                 print('\nSecond Approach (Modified BnB):')
#             solution2 = np.concatenate((sol2[0], sol2[1]), axis=0)
#             print(f'Score: {scr2}\nSolution: {solution2}')
#             if len(solution2) != LENGTHS[i]:
#                 print(f'Paths: {len(solution2)} '
#                       + f'Optimal number of paths: {LENGTHS[i]}')
#             else:
#                 print(f'Optimal number of paths reached: {len(solution2)}')
#             assert set(solution2.flatten()) == mpc2.vertices, \
#                 'Not all vertices covered'
#             print('All vertices covered')
#             print('=============================\n')
