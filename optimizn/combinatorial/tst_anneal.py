import numpy as np
from optimizn.combinatorial.anneal import OptProblem
from graphing.special_graphs.neural_trigraph.rand_graph import *
from graphing.graph import Graph
from graphing.traversal.clr_traversal import Graph1
from graphing.special_graphs.neural_trigraph.path_cover import\
    complete_paths
from graphing.special_graphs.neural_trigraph.neural_trigraph import\
    NeuralTriGraph
from graphing.special_graphs.neural_trigraph.path_cover import \
    min_cover_trigraph
from copy import deepcopy

# For demonstration purposes. 
# We pick the min path cover problem 
# where we have an algorithm for computing
# the optimal solution.
class MinPathCover_NTG(OptProblem):
    """
    Finding the min path cover of a neural trigraph.
    """
    def __init__(self, ntg, swtch=1):
        self.ntg = ntg
        self.adj = ntg.g1.adj
        self.swtch = swtch
        super().__init__()

    def get_candidate(self):
        """
        A candidate is going to be an array of
        arrays, where each array is a full path
        from the left-most layer of the graph
        to the right-most layer.
        """
        paths = []
        self.covered = {}
        ixs = np.arange(1, self.ntg.max_ix+1)
        ixs = np.random.permutation(ixs)
        for i in ixs:
            if i not in self.covered:
                path = self.add_path(i)
                paths.append(path[0])
        self.candidate = paths
        return paths

    def next_candidate_v2(self):
        paths = self.candidate
        covered = deepcopy(self.covered)
        del_paths = []
        for i in range(4):
            ix = np.random.choice(range(len(paths)))
            del_paths.append(paths[ix])
            paths = np.delete(paths, ix, 0)
        for del_path in del_paths:
            for ixx in del_path:
                covered[ixx] -= 1
                if covered[ixx] == 0:
                    path = complete_paths([[ixx]],
                        self.ntg.left_edges,
                        self.ntg.right_edges)
                    path = path[0]
                    for pa in path:
                        covered[pa] += 1
                        #breakpoint()
                    paths = np.concatenate((paths, [path]))
                    #breakpoint()
        return paths

    def next_candidate(self):
        if self.swtch == 0:
            return self.get_candidate()
        else:
            return self.next_candidate_v2()

    def add_path(self, i):
        path = complete_paths([[i]],
            self.ntg.left_edges, self.ntg.right_edges)
        for j in path[0]:
            if j in self.covered:
                self.covered[j] += 1
            else:
                self.covered[j] = 1
        #self.candidate.append(path)
        return path

    def cost(self, candidate):
        ''' 
        Gets the cost for candidate solution.
        '''
        return(len(candidate))

    def update_candidate(self, candidate, cost):
        ## TODO: This can be made more efficient
        ## by updating existing covered set.
        self.covered = {}
        for path in candidate:
            for j in path:
                if j in self.covered:
                    self.covered[j] += 1
                else:
                    self.covered[j] = 1
        super().update_candidate(candidate, cost)



# Scipy simulated annealing can't be used 
# because it expects a 1-d array
# and probably permutes the 1-d array to switch between solutions.
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html

def tst1(edges1=None, edges2=None, n_iter=20000, swtch=1):
    #edges1, edges2 = neur_trig_edges(8, 10, 14)
    if edges1 is None:
        edges1, edges2 = rep_graph(8, 10, 14, reps=4)
    opt_paths = min_cover_trigraph(edges1, edges2)
    print("Optimal solution: " + str(len(opt_paths)))
    ntg = NeuralTriGraph(edges1, edges2)
    #print(ntg.g1.adj)
    mpc = MinPathCover_NTG(ntg, swtch=swtch)
    paths = mpc.get_candidate()
    #mpc.candidate = np.concatenate((mpc.candidate, mpc.candidate))
    print("Current solution: " + str(len(mpc.candidate)))
    mpc.anneal(n_iter)
    print("Best solution: " + str(mpc.best_cost))
    #Now, can we get the min path cover for this?
    return mpc


def tst2(n_iter=20000, swtch=1):
    edges1 = np.loadtxt('edges1.csv')
    edges1 = edges1.astype(int)
    edges2 = np.loadtxt('edges2.csv')
    edges2 = edges2.astype(int)
    return tst1(edges1, edges2, n_iter, swtch=swtch)

