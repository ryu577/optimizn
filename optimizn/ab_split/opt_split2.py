from optimizn.ab_split.opt_split import Tree, \
    form_arrays, create_matr
from optimizn.ab_split.trees.lvlTrees import Node1
import numpy as np
from copy import deepcopy
import queue
from optimizn.ab_split.testing.cluster_hw import df1
from heapq import heappop, heappush


class Tree1(Tree):
    """
    The arrays per hardware are very sparse. They have a few non
    zero entries and lots of zero entries. Creating the tree for
    these kinds of arrays makes it explode since it doesn't matter
    which group the zero entries go to. To mitigate this issue, we
    can construct a combined tree across all dp matrices in one shot
    instead of one dp matrix at a time and then taking intersection.
    """
    def __init__(self, arrays, matrices, targets):
        self.arrays = arrays
        self.matrices = matrices
        self.targets = targets
        self.cols = deepcopy(targets)
        self.n = len(matrices)
        self.n_clstrs = len(self.matrices[0])
        self.stop = False
        self.path1 = []
        path = []
        self.root = self.mk_tree(ro=self.n_clstrs-2,
                                 cols=self.cols, path1=path)

    def mk_tree(self, ro, cols, path1=[]):
        if self.stop:
            return
        if ro < -1:
            print(path1)
            self.path1 = deepcopy(path1)
            self.stop = True
            return
        # print(str(ro)+",")
        for i in range(self.n):
            mat = self.matrices[i]
            col = cols[i]
            if not mat[ro+1][col] or col < 0:
                return
        node1 = Node1(ro)
        cnt = 0
        for i in range(self.n):
            mat = self.matrices[i]
            col = cols[i]
            if mat[ro+1-1][col]:
                cnt += 1
        if cnt == self.n:
            node1.right = self.mk_tree(ro-1, cols, path1)
        cnt = 0
        col_deltas = np.zeros(self.n).astype(int)
        for i in range(self.n):
            mat = self.matrices[i]
            col = cols[i]
            arr = self.arrays[i]
            if col - arr[ro] >= 0 and \
                    mat[ro+1-1][col - arr[ro]]:
                cnt += 1
                col_deltas[i] = arr[ro]
        if cnt == self.n:
            cols = cols - col_deltas
            path1.append(ro)
            node1.left = self.mk_tree(ro-1, cols, path1)
            path1.pop()
            cols = cols + col_deltas
        return node1

    def find_1path(self, node, depth=0, path=...):
        return super().find_1path(node, depth, path)


class OptProblm():
    """
    Does data cleaning of the array, removes the zeros, etc.
    """
    def __init__(self):
        # This file path isn't used since we import the dataframe
        # from a pandas file.
        self.arrays, self.hws_ix, self.cl_ix =\
            form_arrays(df1)
        # Remove arrays where total nodes less than 10 and ones
        # where only one cluster has the hardware since splitting
        # by cluster in those cases won't make sense.
        self.arrays = self.arrays[np.sum(self.arrays, axis=1) > 10]
        self.arrays = self.arrays[np.sum(self.arrays != 0, axis=1) > 1]
        # Now we remove the clusters (columns) where its all zeros.
        self.mask = np.sum(self.arrays, axis=0) != 0
        self.arrays2 = []
        for arr in self.arrays:
            self.arrays2.append(arr[self.mask])
        self.arrays = self.arrays2
        self.path1 = optimize3(self.arrays)

    def optimize(self):
        tr = Tree1(self.arrays, self.matrices, self.targets)
        self.tree = tr


def optimize1(arrays):
    matrices = []
    targets = []
    target_cands = []
    for arr in arrays:
        sum1 = np.sum(arr)
        matr = create_matr(arr, sum1)
        last_ro = matr[len(matr)-1]
        all_trgts = np.arange(len(matr[0]))[last_ro]
        target = sum1//2
        (all_trgts - target)
        target_cands.append(all_trgts)
        for x in range(sum1//2-1):
            if last_ro[sum1//2-x]:
                target = sum1//2-x
                break
            if last_ro[sum1//2+x]:
                target = sum1//2+x
                break
        targets.append(target)
        matrices.append(matr)
    tr = Tree1(arrays, matrices, targets)
    return tr.path1


def find_path1(arrays, matrices, targets):
    tr = Tree1(arrays, matrices, targets)
    return tr.path1


def prepare_data(arrays, fn1=find_path1):
    """
    Superceded by prepare_data2
    """
    matrices = []
    targets = []
    target_cands = []
    for arr in arrays:
        sum1 = np.sum(arr)
        matr = create_matr(arr, sum1)
        last_ro = matr[len(matr)-1]
        all_trgts = np.arange(len(matr[0]))[last_ro]
        target = sum1//2
        deltas = abs(all_trgts - target)
        deltainds = deltas.argsort()
        all_trgts = all_trgts[deltainds[::1]]
        target_cands.append(all_trgts)
        targets.append(target)
        matrices.append(matr)
    op = OptProblem2(arrays, matrices, targets, target_cands, fn1)
    return op


def optimize3(arrays):
    op = prepare_data(arrays)
    # op.itr_arrays()
    op.itr_arrays_bfs()
    return op.path1


def optimize6(arrays):
    op = prepare_data(arrays)
    op.itr_arrays_heap()
    return op.path1


def itr_arrays(arrays, lvl=0, targets=[]):
    """
    Given an array of arrays (could be jagged),
    All possible arrays formed by taking one entry
    from each of the arrays.
    """
    if lvl == len(arrays):
        yield targets
    for xx in arrays[lvl]:
        targets.append(xx)
        itr_arrays(arrays, lvl+1, targets)
        targets.pop()


class OrderedTuple():
    def __init__(self, key, arr):
        self.key = key
        self.arr = arr

    def __eq__(self, other):
        return self.key == other.key

    def __gt__(self, other):
        return self.key > other.key


class OptProblem2():
    def __init__(self, arrays, matrices, targets,
                 target_cands,
                 opt_fn=find_path1):
        self.arrays = arrays
        self.matrices = matrices
        self.targets = targets
        self.target_cands = target_cands
        self.opt_fn = opt_fn
        self.stop_looking = False
        self.verbose = False
        self.path1 = None
        self.max_cand = np.inf

    def itr_arrays(self, lvl=0, targets=[]):
        """
        Given an array of arrays (could be jagged),
        All possible arrays formed by taking one entry
        from each of the arrays. This one should be used
        when we have a strong priority order between the arrays.
        The first array in the list of arrays has the highest priority
        the second one has the second highest and so on.
        """
        if self.stop_looking:
            return
        if lvl == len(self.target_cands):
            # tr = Tree1(self.arrays, self.matrices, targets)
            path1 = find_path1(self.arrays, self.matrices, targets)
            if len(path1) > 0:
                self.path1 = path1
                # Could have used yield here as well.
                self.stop_looking = True
            return
        for xx in self.target_cands[lvl]:
            if not self.stop_looking:
                targets.append(xx)
                self.itr_arrays(lvl+1, targets)
                targets.pop()

    def itr_arrays_bfs(self):
        """
        Don't use. Use itr_arrays_heap instead.
        """
        q = queue.Queue()
        u1 = np.zeros(len(self.target_cands)).astype(int)
        # init_arr = [self.target_cands[ix][0] for ix in u1]
        init_arr = self.ix_arr_to_arr(u1)
        q.put(u1)
        while q and not self.stop_looking:
            u = q.get()
            u_arr = self.ix_arr_to_arr(u)
            if u_arr is not None:
                tr = Tree1(self.arrays, self.matrices, u_arr)
                if len(tr.path1) > 0:
                    self.path1 = tr.path1
                    self.stop_looking = True
                    break
            dists = []
            vs = []
            for ix in range(len(self.target_cands)):
                delta = np.zeros(len(self.target_cands)).astype(int)
                delta[ix] = 1
                v1 = u + delta
                v = self.ix_arr_to_arr(v1)
                if v is not None:
                    dist = manhattan_dist(init_arr, v)
                    dists.append(dist)
                    vs.append(v1)
            dists = np.array(dists)
            vs = np.array(vs)
            deltainds = dists.argsort()
            for dix in deltainds:
                q.put(vs[dix])

    def itr_arrays_heap(self):
        q = []
        u1 = np.zeros(len(self.target_cands)).astype(int)
        # The distance isn't really 0, but it doesn't matter
        # since this is the first tuple.
        ot = OrderedTuple(0, u1)
        heappush(q, ot)
        itr = 0
        while q and not self.stop_looking and itr <= self.max_cand:
            itr += 1
            u_op = heappop(q)
            u = u_op.arr
            dist = u_op.key
            u_arr = self.ix_arr_to_arr(u)
            if u_arr is not None:
                path1 = self.opt_fn(self.arrays, self.matrices, u_arr)
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
                    heappush(q, ot1)

    def ix_arr_to_arr(self, v1):
        v = []
        for idx in range(len(v1)):
            uix = int(v1[idx])
            if len(self.target_cands[idx])-1 < uix:
                break
            v.append(self.target_cands[idx][uix])
        if len(v) == len(self.target_cands):
            return v


def manhattan_dist(arr1, arr2):
    dist = 0
    for ix in range(len(arr1)):
        dist += abs(arr1[ix] - arr2[ix])
    return dist


def tst1():
    op = OptProblm()
    sums1 = [619, 596, 589, 1146, 13, 483, 37, 17, 29, 255, 304]
    for ix in range(len(op.arrays)):
        arr = op.arrays[ix]
        sum1 = sum(arr[op.path1])
        prcnt1 = sum1/sum(arr)
        prcnt2 = sums1[ix]/sum(arr)
        lower = min(prcnt2, 1-prcnt2)
        higher = max(prcnt2, 1-prcnt2)
        assert (lower <= prcnt1 and prcnt1 <= higher)
    return op


def tst2():
    # tst1()
    arrays = [
                [3, 34, 4, 12, 5, 2],
                [0, 25, 4, 12, 5, 2],
                [22, 10, 4, 12, 5, 2],
             ]
    arrs = optimize6(arrays)
    print(arrs)


def tst3():
    arrs = [
        [2, 3],
        [2, 3]
    ]
    path = optimize6(arrs)
    print(path)


if __name__ == "__main__":
    tst3()


#########################
# TODO
# 1. [Done] Full optimization when combined tree across matrices.
# 2. When zeros culled from arrays, update hws_ix and clust_ix.
# 3. [Done] Switch to using pandas file or include CSV in package.
# 4. Try on VMSKU as well and then combined HW VMSKU.

#########################
# 2. Approach
# Save the mask, original arrays and new array.
# Create a mapping between original indices and new indices + vice versa.
