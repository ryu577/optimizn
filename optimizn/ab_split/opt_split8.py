import numpy as np
from copy import deepcopy
from optimizn.ab_split.opt_split6 import read_best_mat, get_arrays, obj_fn,\
    BlockMatrix, obj_fn_ix, write2csv
from optimizn.ab_split.opt_split4 import clean_data


def split_by_cols(arrs):
    """
    Make the split configurable.
    """
    arrs_t = np.transpose(arrs)
    arrs1_t = arrs_t[:25]
    arrs2_t = arrs_t[25:]
    arrs3_t = arrs_t[25:]
    arrs1 = np.transpose(arrs1_t)
    arrs2 = np.transpose(arrs2_t)
    arrs3 = np.transpose(arrs3_t)
    arrs1, arrs2, arrs3 =\
        clean_data(arrs1), clean_data(arrs2), clean_data(arrs3)
    return arrs1, arrs2, arrs3


def tst1(ret_f=0.5):
    arrs = read_best_mat()
    arrs1, arrs2, arrs3 = split_by_cols(arrs)
    cst, bsti = obj_fn(arrs2, len(arrs2[0])//2)
    print("Cost: " + str(cst) + " Best ix: " + str(bsti))
    print("Now simulated annealing.")
    bm = BlockMatrix(arrs2)
    bm.anneal(n_iter=2000, reset_p=0.0, retarding_factor=ret_f)
    arr1 = bm.best_solution
    print(obj_fn_ix(arr1, bm.col_ix, bm.ro_ix))
    write2csv(arr1)
    sol1 = obj_fn(arr1, bm.col_ix)
    print(sol1)
    print(bm.best_cost)


if __name__ == "__main__":
    tst1()
