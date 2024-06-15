import numpy as np
from optimizn.ab_split.opt_split6 import read_best_mat, get_arrays
from optimizn.ab_split.opt_split4 import optimize9
from optimizn.ab_split.opt_split5 import optimize11
from optimizn.ab_split.opt_split4 import clean_data, \
    OptProblem3, create_sparse_tree, DataContainer
from optimizn.ab_split.opt_split import form_arrays, unionTrees, create_matr
from optimizn.ab_split.testing.cluster_vmsku import cluster_vmsku


def split_in_3(arrs):
    # n_ros = len(arrs)
    n_cols = len(arrs[0])
    col_split = 20
    arrs_t = np.transpose(arrs)
    arrs1_t = arrs_t[:17]
    arrs2_t = arrs_t[17:34]
    arrs3_t = arrs_t[34:52]
    arrs1 = np.transpose(arrs1_t)
    arrs2 = np.transpose(arrs2_t)
    arrs3 = np.transpose(arrs3_t)
    arrs1, arrs2, arrs3 =\
        clean_data(arrs1), clean_data(arrs2), clean_data(arrs3)
    return arrs1, arrs2, arrs3


class TarCand():
    """
    This is just a demonstration of iterating through
    a jagged array in a way that each arrays index
    is only increased.
    """
    def __init__(self, targ_cands, targets=[]):
        self.targ_cands = targ_cands
        self.n = len(targ_cands)
        if len(targets) == 0:
            self.targets = np.zeros(self.n)
        else:
            self.targets = targets

    def itr_targets(self):
        n = len(self.targ_cands)
        cand = [self.targ_cands[i][0] for i in range(n)]
        ixs = np.zeros(n).astype(int)
        self.m_set = set()
        b_ix = -1
        while len(self.m_set) < n:
            print(cand)
            b_ix = self.get_best(ixs)
            ixs[b_ix] += 1
            if ixs[b_ix] == len(self.targ_cands[b_ix])-1:
                self.m_set.add(b_ix)
            cand[b_ix] = self.targ_cands[b_ix][ixs[b_ix]]
        print(cand)

    def get_best(self, ixs):
        cands = [i for i in range(self.n) if i not in self.m_set]
        # return np.random.choice(cands).astype(int)
        min1 = np.inf
        ix1 = 0
        for cand in cands:
            tmp_num = abs(self.targ_cands[cand][ixs[cand]] -
                          self.targets[cand])
            if tmp_num < min1:
                min1 = tmp_num
                ix1 = cand
        return ix1


class OptProblem4(OptProblem3, TarCand):
    def __init__(self, dc):
        super().__init__(dc)
        self.max_tree_size = np.inf
        self.targ_cands = dc.target_cands
        self.n = len(self.targ_cands)

    def itr_arrays(self):
        n = len(self.targ_cands)
        cand = [self.targ_cands[i][0] for i in range(n)]
        ixs = np.zeros(n).astype(int)
        self.m_set = set()
        b_ix = -1
        while len(self.m_set) < n:
            print("Finding path for:" + str(cand))
            self.path1 = self.find_path()
            if self.path1 is not None and len(self.path1) > 0:
                return
            b_ix = self.get_best(ixs)
            ixs[b_ix] += 1
            if ixs[b_ix] == len(self.targ_cands[b_ix])-1:
                self.m_set.add(b_ix)
            target = self.targ_cands[b_ix][ixs[b_ix]]
            cand[b_ix] = target
            print("Updating tree at index:" + str(b_ix) +
                  " with target: " + str(target))
            self.update_tree(target, b_ix)
        print("Finding path for:" + str(cand))
        self.path1 = self.find_path()

    def update_tree(self, target, ix):
        arr = self.arrays[ix]
        mat1 = self.matrices[ix]
        tree1 = create_sparse_tree(arr, mat1, target)
        tree2 = self.trees[ix]
        if tree1 is None:
            self.trees[ix] = None
        else:
            tree2.root = unionTrees(tree2.root, tree1.root)


def optimize12(arrs):
    dc = DataContainer(arrs)
    op = OptProblem4(dc)
    op.itr_arrays()
    return op.path1


def optimize13(arrs):
    """
    Optimizes an array that has already been split in two.
    Will work without an optimal split as well.
    """
    arrs1, arrs2, arrs3 = split_in_3(arrs)
    split1 = optimize12(arrs1)
    split2 = optimize12(arrs2)


def tst1():
    targ_cands = [
        [1, 2, 3],
        [4, 5, 6, 10],
        [7, 8, 9]
    ]
    tg = TarCand(targ_cands)
    tg.itr_targets()


def tst2():
    arr = [
            [3, 34, 4, 12, 5, 2],
            [0, 25, 4, 12, 5, 2],
            [22, 10, 4, 12, 5, 2],
        ]
    path1 = optimize12(arr)
    print(path1)


def tst3():
    # arrs = read_best_mat()
    arrs, vm_ix, cl_ix =\
        form_arrays(cluster_vmsku, "Usage_VMSize")
    arrs1, arrs2, arrs3 = split_in_3(arrs)
    return optimize12(arrs2)


if __name__ == "__main__":
    aa = tst3()
