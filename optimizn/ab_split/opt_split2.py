from optimizn.ab_split.opt_split import Node1, Tree, \
    form_arrays, create_matr
import numpy as np
from copy import deepcopy
from optimizn.ab_split.testing.cluster_hw import df1


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
            # self.stop = True
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
    def __init__(self,
                 file_path='/Users/rohitpandey/Docs2/Obsidian/diary/data/Canary_ClusterHW.csv'):
        # This file path isn't used since we import the dataframe
        # from a pandas file.
        self.file_path = file_path
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
        self.matrices = []
        self.targets = []
        for arr in self.arrays:
            sum1 = np.sum(arr)
            matr = create_matr(arr, sum1)
            last_ro = matr[len(matr)-1]
            target = sum1//2
            for x in range(sum1//2-1):
                if last_ro[sum1//2-x]:
                    target = sum1//2-x
                    break
                if last_ro[sum1//2+x]:
                    target = sum1//2+x
                    break
            self.targets.append(target)
            self.matrices.append(matr)

    def optimize(self):
        tr = Tree1(self.arrays, self.matrices, self.targets)
        self.tree = tr


def tst1():
    op = OptProblm()
    op.optimize()
    sums1 = [619, 596, 589, 1146, 13, 483, 37, 17, 29, 255, 304]
    for ix in range(len(op.arrays)):
        arr = op.arrays[ix]
        sum1 = sum(arr[op.tree.path1])
        prcnt1 = sum1/sum(arr)
        prcnt2 = sums1[ix]/sum(arr)
        lower = min(prcnt2, 1-prcnt2)
        higher = max(prcnt2, 1-prcnt2)
        assert (lower <= prcnt1 and prcnt1 <= higher)
    return op


if __name__ == "__main__":
    tst1()


#########################
# TODO
# 1. Full optimization when combined tree across matrices.
# 2. When zeros culled from arrays, update hws_ix and clust_ix.
# 3. [Done] Switch to using pandas file or include CSV in package.
# 4. Try on VMSKU as well and then combined HW VMSKU.

#########################
# 2.
# Save the mask, original arrays and new array.
# Create a mapping between original indices and new indices + vice versa.
