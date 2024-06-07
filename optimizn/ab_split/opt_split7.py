import numpy as np
from optimizn.ab_split.opt_split6 import read_best_mat
from optimizn.ab_split.opt_split4 import optimize9
from optimizn.ab_split.opt_split5 import optimize11
from optimizn.ab_split.opt_split4 import clean_data


def optimize12(arrs):
    """
    Optimizes an array that has already been split in two.
    Will work without an optimal split as well.
    """
    arrs1, arrs2, arrs3 = split_in_3(arrs)
    split1 = optimize9(arrs1)
    split2 = optimize9(arrs2)


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


def tst1():
    arrs = read_best_mat()
    return arrs
