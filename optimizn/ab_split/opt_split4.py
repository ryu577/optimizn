from optimizn.ab_split.opt_split import form_arrays, unionTrees
from optimizn.ab_split.testing.cluster_vmsku import cluster_vmsku
from optimizn.ab_split.opt_split3 import optimize7, create_sparse_tree, prepare_data, find_a_path
from optimizn.ab_split.opt_split2 import optimize6, OptProblem2, \
                                        OrderedTuple, manhattan_dist
from optimizn.ab_split.evaluation import calc_sol_delta
from optimizn.ab_split.trees.lvlTrees import Tree2, intrsctAllTrees
import numpy as np
import random
from heapq import heappop, heappush


def clean_data(arrays):
    # # Now some data cleaning.
    # Remove arrays where total nodes less than 10 and ones
    # where only one cluster has the hardware since splitting
    # by cluster in those cases won't make sense.
    arrays = arrays[np.sum(arrays, axis=1) > 10]
    arrays = arrays[np.sum(arrays != 0, axis=1) > 1]
    # Now we remove the clusters (columns) where its all zeros.
    mask = np.sum(arrays, axis=0) != 0
    arrays2 = []
    for arr in arrays:
        arrays2.append(arr[mask])
    arrays = arrays2
    return arrays


def tst2():
    arrays, vm_ix, cl_ix =\
        form_arrays(cluster_vmsku, "Usage_VMSize")
    arrays = clean_data(arrays)


def optimize8(arrays, n_iter=500):
    min_cost = np.infty
    best_split = []
    for i in range(n_iter):
        arrays1 = random.sample(arrays, 7)
        path1 = optimize7(arrays1, max_cand=50)
        if path1 is not None and len(path1) > 0:
            delta1 = calc_sol_delta(arrays, path1)
            print(delta1)
            if delta1 < min_cost:
                min_cost = delta1
                best_split = path1
    return best_split


class OptProblem3(OptProblem2):
    def __init__(self, arrays, matrices, targets,
                 target_cands,
                 opt_fn):
        super().__init__(arrays, matrices, targets,
                         target_cands,
                         opt_fn)
        self.trees = []
        self.covered_upto = np.zeros(len(arrays))
        self.verbose = True
        for ix in range(len(self.arrays)):
            arr = arrays[ix]
            mat = matrices[ix]
            target = targets[ix]
            tree1 = create_sparse_tree(arr, mat, target)
            self.trees.append(tree1)

    def itr_arrays_heap(self):
        heap1 = []
        u1 = np.zeros(len(self.target_cands)).astype(int)
        # The distance isn't really 0, but it doesn't matter
        # since this is the first tuple.
        ot = OrderedTuple(0, u1)
        heappush(heap1, ot)
        itr = 0
        while heap1 and not self.stop_looking and itr <= self.max_cand:
            itr += 1
            u_op = heappop(heap1)
            u = u_op.arr
            expand_ix = []
            for ix in range(len(u)):
                if self.covered_upto[ix] < u[ix]:
                    self.covered_upto[ix] = u[ix]
                    expand_ix.append(ix)
            dist = u_op.key
            u_arr = self.ix_arr_to_arr(u)
            if u_arr is not None:
                if len(expand_ix) > 0:
                    self.update_trees(u_arr, expand_ix)
                    path1 = self.find_path()
                    if self.verbose:
                        print("evaluating: " + str(u_arr) + " at dist: "
                              + str(dist))
                    if path1 is not None:
                        if len(path1) > 0:
                            self.path1 = path1
                            self.stop_looking = True
                            break
            for ix in range(len(self.target_cands)):
                delta = np.zeros(len(self.target_cands)).astype(int)
                delta[ix] = 1
                v1 = u + delta
                v = self.ix_arr_to_arr(v1)
                if v is not None:
                    dist = manhattan_dist(self.targets, v)
                    ot1 = OrderedTuple(dist, v1)
                    heappush(heap1, ot1)

    def update_trees(self, u_arr, expand_ix):
        for ix in expand_ix:
            arr = self.arrays[ix]
            mat1 = self.matrices[ix]
            target = u_arr[ix]
            tree1 = create_sparse_tree(arr, mat1, target)
            tree2 = self.trees[ix]
            tree2 = unionTrees(tree2, tree1)

    def find_path(self):
        tree1 = intrsctAllTrees(self.trees)
        tree1.find_best_path()
        return tree1.path1


def optimize9(arrays):
    op = prepare_data(arrays, find_a_path)
    op.itr_arrays_heap()
    return op.path1


def tst1():
    arrs = [np.array([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0, 82,  0,  0,  0,  0,  0, 57,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0]),
            np.array([0,  0,  0,  0,  0,  0,  2,  0,  0,  5,  0,  0,  0,  0,  0,  0,  0,
            2,  0, 65, 10,  1,  0, 16,  4,  0,  2,  0,  0,  8,  0,  0,  0,  0,
            29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0]),
            np.array([0,   0,   0,   0,   0,   0,   0,  18,   4,   0,   1,   0,   0,
            0,   0,   0,   7,   0,  23,  61,  11,  52,  38, 108,  14,  29,
            56,   1,   9,  47,   3,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1,
            0, 4, 1, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0,   0,   0,   0,   0,   0,   2, 171,  98,  29,  86,  10,  62,
            75,   0,   0,  33,   2,   0,   0,   0,   0,   0,   0,   0,   0,
            0,  74,   0,   0, 107,   0,   0,   0,   0,   0,   4,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])]
    path1 = optimize6(arrs)
    #path2 = optimize7(arrs)
    print(path1)
    #print(path2)
    return path1


def tst2():
    arr = [
            [3, 34, 4, 12, 5, 2],
            [0, 25, 4, 12, 5, 2],
            [22, 10, 4, 12, 5, 2],
        ]
    path1 = optimize9(arr)
    print(path1)


if __name__ == "__main__":
    tst2()
