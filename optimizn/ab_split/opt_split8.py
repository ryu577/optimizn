import numpy as np
from copy import deepcopy
from optimizn.ab_split.opt_split6 import read_best_mat, get_arrays, obj_fn,\
    BlockMatrix, obj_fn_ix, write2csv
from optimizn.ab_split.opt_split4 import clean_data
from optimizn.ab_split.opt_split7 import optimize12
from optimizn.ab_split.evaluation import calc_sol_delta
from optimizn.ab_split.testing.cluster_vmsku import cluster_vmsku
from optimizn.ab_split.opt_split import form_arrays


class OptProblem5():
    def __init__(self, arrs):
        self.arrs = arrs

    def clean_data2(self, arrays):
        """
        This version of clean_data was needed since we need to return
        the mask as well to keep track of the columns that were dropped.
        """
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
        return arrays, mask

    def split_by_cols(self, arrs, ix=25):
        """
        Make the split configurable.
        """
        arrs_t = np.transpose(arrs)
        arrs1_t = arrs_t[:ix]
        arrs2_t = arrs_t[ix:]
        arrs3_t = arrs_t[ix:]
        arrs1 = np.transpose(arrs1_t)
        arrs2 = np.transpose(arrs2_t)
        arrs3 = np.transpose(arrs3_t)
        res1, res2, res3 =\
            self.clean_data2(arrs1), self.clean_data2(arrs2),\
                self.clean_data2(arrs3)
        self.arrs1, self.msk1 = res1
        self.arrs2, self.msk2 = res2
        self.arrs3, self.msk3 = res3
        return self.arrs1, self.arrs2, self.arrs3
    
    def split_by_cols2(self, arrs, ixs=[0, 17, 34, 52]):
        arrs_t = np.transpose(arrs)
        res_arrs = []
        msks = []
        for ii in range(len(ixs)-1):
            arrs1_t = arrs_t[ixs[ii]:ixs[ii+1]]
            arrs1 = np.transpose(arrs1_t)
            arrs1, msk1 = self.clean_data2(arrs1)
            arrs = deepcopy(arrs1); msk = deepcopy(msk1)
            res_arrs.append(arrs)
            msks.append(msk)
        self.res_arrs = res_arrs
        self.msks = msks

    def split1(self):
        arrs1, arrs2, arrs3 = self.split_by_cols(self.arrs)
        cst, bsti = obj_fn(arrs2, len(arrs2[0])//2)
        print("Cost: " + str(cst) + " Best ix: " + str(bsti))

    def sim_anneal(self, ret_f=0.5):
        print("Now simulated annealing.")
        bm = BlockMatrix(self.arrs2)
        bm.anneal(n_iter=2000, reset_p=0.0, retarding_factor=ret_f)
        arr1 = bm.best_solution
        print(obj_fn_ix(arr1, bm.col_ix, bm.ro_ix))
        write2csv(arr1)
        sol1 = obj_fn(arr1, bm.col_ix)
        print(sol1)
        print(bm.best_cost)
        return bm.best_solution


def tst2():
    arrs = read_best_mat()
    op = OptProblem5(arrs)
    op.split_by_cols(arrs)
    aa0 = optimize12(op.arrs1)
    op1 = OptProblem5(op.arrs2)
    arr1, arr2, arr3 = op1.split_by_cols(op.arrs2, len(op.arrs2[0])//2)
    print("Columns in first: " + str(arr1[0]))
    aa1 = optimize12(arr1)
    print("Done with first half")
    print("Columns in second: " + str(arr2[0]))
    aa2 = optimize12(arr2)
    print("Done with second half")


def best_heuristic():
    aa0 = [20, 0]
    aa1 = [12, 11, 8, 6, 4, 3]
    aa2 = [10, 9, 6, 5, 4, 1, 0]
    aa1_prime = [i for i in np.arange(13) if i not in aa1]
    aa2_prime = [i for i in np.arange(14) if i not in aa2]

    aa1s = [aa1, aa1_prime]
    aa2s = [aa2, aa2_prime]

    min_obj = np.inf
    res1 = []
    for aa1 in aa1s:
        for aa2 in aa2s:
            res = [0, 22]
            for i in aa1:
                res.append(i + 25)

            for i in aa2:
                res.append(i + 25 + 13)

            arrs = read_best_mat()
            obj1 = calc_sol_delta(arrs, res)
            if obj1 < min_obj:
                min_obj = obj1
                res1 = res

    print(min_obj)
    # 11523


def simple_split_heuristic():
    arrs, vm_ix, cl_ix =\
        form_arrays(cluster_vmsku, "Usage_VMSize")
    arrs = clean_data(arrs)
    op = OptProblem5(arrs)
    op.split_by_cols2(arrs)
    aa0 = optimize12(op.res_arrs[0])
    # aa0 = [16, 15, 10, 7, 6]
    aa1 = optimize12(op.res_arrs[1])
    aa2 = optimize12(op.res_arrs[2])
