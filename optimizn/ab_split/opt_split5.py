import numpy as np
from copy import deepcopy
from optimizn.ab_split.testing.cluster_vmsku import cluster_vmsku
from optimizn.ab_split.trees.lvlTrees import Node1
from optimizn.ab_split.opt_split2 import OrderedTuple, manhattan_dist, \
    OptProblem2
from optimizn.ab_split.opt_split4 import DataContainer, clean_data
from optimizn.ab_split.opt_split2 import optimize6
from heapq import heappop, heappush
from optimizn.ab_split.opt_split import form_arrays
import random


class OptSplit(OptProblem2):
    def __init__(self, dc):
        self.dc = dc
        self.arrays, self.matrices, self.targets, \
            self.target_cands = dc.arrays, dc.matrices, \
            dc.targets, dc.target_cands
        self.n = len(self.matrices)
        self.n_clstrs = len(self.matrices[0])
        self.target_sets = [[i] for i in self.targets]
        self.verbose = True
        self.stop = False
        self.covered_upto = np.ones(self.n)*-1
        self.max_cand = np.inf
        self.cols = self.target_sets
        self.path1 = []

    def mk_tree(self, ro, cols, path1=[]):
        # print(str(ro) + "," + str(cols))
        if self.stop:
            return
        if ro < -1:
            print(path1)
            self.path1 = deepcopy(path1)
            self.stop = True
            return
        for i in range(self.n):
            mat = self.matrices[i]
            col = cols[i]
            cnt1 = 0
            for jj in col:
                if not mat[ro+1][jj] or jj < 0:
                    cnt1 += 1
            if cnt1 == len(col):
                return
        node1 = Node1(ro)
        cnt = 0
        for i in range(self.n):
            mat = self.matrices[i]
            col = cols[i]
            for jj in col:
                if mat[ro+1-1][jj]:
                    cnt += 1
                    break
        if cnt == self.n:
            node1.right = self.mk_tree(ro-1, cols, path1)
        cnt = 0
        col_deltas = np.zeros(self.n).astype(int)
        for i in range(self.n):
            mat = self.matrices[i]
            col_set = cols[i]
            arr = self.arrays[i]
            for col in col_set:
                if col - arr[ro] >= 0 and \
                        mat[ro+1-1][col - arr[ro]]:
                    cnt += 1
                    col_deltas[i] = arr[ro]
                    break
        if cnt == self.n:
            self.target_set_add(-col_deltas)
            cols = self.target_sets
            path1.append(ro)
            node1.left = self.mk_tree(ro-1, cols, path1)
            print(path1)
            path1.pop()
            self.target_set_add(col_deltas)
            cols = self.target_sets
        return node1

    def itr_arrays_heap(self):
        heap1 = []
        u1 = np.zeros(len(self.target_cands)).astype(int)
        # The distance isn't really 0, but it doesn't matter
        # since this is the first tuple.
        ot = OrderedTuple(0, u1)
        heappush(heap1, ot)
        itr = 0
        while heap1 and not self.stop and itr <= self.max_cand:
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
                    self.update_sets(u_arr, expand_ix)
                    self.mk_tree(self.n_clstrs-2, self.target_sets)
                    if self.verbose:
                        print("evaluating: " + str(u_arr) + " at dist: "
                              + str(dist))
                    if self.path1 is not None:
                        if len(self.path1) > 0:
                            self.path1 = self.path1
                            self.stop = True
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

    def update_sets(self, u_arr, expand_ix):
        for ix in expand_ix:
            target = u_arr[ix]
            if target not in self.target_sets[ix]:
                self.target_sets[ix].append(target)

    def target_set_add(self, col_deltas):
        for ix in range(len(self.target_sets)):
            for ix1 in range(len(self.target_sets[ix])):
                self.target_sets[ix][ix1] += col_deltas[ix]


def optimize11(arrays):
    dc = DataContainer(arrays)
    os1 = OptSplit(dc)
    os1.itr_arrays_heap()
    return os1.path1


def tst2():
    arr = [
            [3, 34, 4, 12, 5, 2],
            [0, 25, 4, 12, 5, 2],
            [22, 10, 4, 12, 5, 2],
        ]
    path1 = optimize11(arr)
    print(path1)


def tst3():
    arr = [
            [3, 34, 4, 12, 5, 2],
            [3, 34, 4, 12, 5, 2],
            [3, 34, 4, 12, 5, 2],
        ]
    path1 = optimize6(arr)
    print(path1)


def tst4():
    arr = [
            [2, 3],
            [2, 3]
        ]
    path1 = optimize11(arr)
    print(path1)


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
    sum_arrs = sum(arrs)
    arrs = [arr1[sum_arrs > 0] for arr1 in arrs]
    path1 = optimize11(arrs)
    print(path1)
    return path1


def tst5():
    arrays, vm_ix, cl_ix =\
        form_arrays(cluster_vmsku, "Usage_VMSize")
    arrays = clean_data(arrays)
    sum_arrs = sum(arrays)
    arrays = [arr1[sum_arrs > 100] for arr1 in arrays]
    arrays = random.sample(arrays, 10)
    path1 = optimize11(arrays)
    print(path1)


if __name__ == "__main__":
    tst5()
