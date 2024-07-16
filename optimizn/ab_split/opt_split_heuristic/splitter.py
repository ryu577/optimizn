import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import random
from optimizn.ab_split.opt_split6 import BlockMatrix


def main1():
    files = os.listdir('Data/OptSplit/Reader/Matrix/')
    file1 = sorted(files)[len(files)-1]
    mat_df = pd.read_csv('Data/OptSplit/Reader/Matrix/' + file1)
    m_id = mat_df.matId[0]
    mat1 = df_to_mat(mat_df)
    bm = BlockMatrix1(mat1)
    bm.anneal(n_iter=1000, reset_p=0.0)
    # arr1 = bm.best_solution
    write_result(bm, m_id)
    return bm


def df_to_mat(mat_df):
    ro_mx = max(mat_df.roIx)
    co_mx = max(mat_df.coIx)
    mat1 = np.zeros(shape=(ro_mx+1, co_mx+1))
    for _, ro in mat_df.iterrows():
        row, col = ro.roIx, ro.coIx
        mat1[row, col] = int(ro.matDat)
    return mat1


def write_result(bm, m_id):
    permut_id = int(time.time())
    index = [i for i in range(bm.nros + bm.ncols)]
    map_df = pd.DataFrame(columns=['matId', 'permutId',
                                   'IxType',
                                   'Ix', 'origIx'],
                          index=index)
    index = [i for i in range(3)]
    splt_df = pd.DataFrame(columns=['permutId',
                                    'spltIx'],
                           index=index)
    for i in range(len(bm.row_perm)):
        ro_prm = bm.row_perm[i]
        map_df.loc[i] = [m_id, permut_id, 'row', ro_prm, i]
    for i in range(len(bm.col_perm)):
        co_prm = bm.col_perm[i]
        map_df.loc[i] = [m_id, permut_id, 'col', co_prm, i+bm.nros]
    t1 = datetime.now()
    day = t1.day
    month = t1.month
    year = t1.year
    map_df.to_csv('Data/OptSplit/Splitter/Permutation/' +
                  str(year) + str(month) +
                  str(day) + '_' + str(m_id) +
                  '.csv', index=False)
    splt_df.loc[0] = [permut_id, 0]
    splt_df.loc[1] = [permut_id, bm.col_ix]
    splt_df.loc[2] = [permut_id, bm.ncols]
    splt_df.to_csv('Data/OptSplit/Splitter/Split/' +
                   str(year) + str(month) +
                   str(day) + '_' + str(m_id) +
                   '.csv', index=False)


class BlockMatrix1(BlockMatrix):
    def __init__(self, arr):
        super().__init__(arr)
        self.row_perm = np.arange(self.nros)
        self.col_perm = np.arange(self.ncols)
        self.col_ix = self.ncols//2

    def next_candidate(self, arr):
        uu = np.random.uniform()
        [ro1, ro2] = random.sample([i for i in range(self.nros)], 2)
        [co1, co2] = random.sample([i for i in range(self.ncols)], 2)
        if uu < 0.5:
            arr = self.sort_arrs(arr, co1, True)
        else:
            arr1 = np.transpose(arr)
            arr1 = self.sort_arrs(arr1, ro1, False)
            arr = np.transpose(arr1)
        return arr

    def sort_arrs(self, arr, col_ix, by_row=True):
        col_arr = [ro[col_ix] for ro in arr]
        arr_ixs = np.argsort(col_arr)
        arr = [arr[i] for i in arr_ixs]
        if by_row:
            self.row_perm = [self.row_perm[i] for i in arr_ixs]
        else:
            self.col_perm = [self.col_perm[i] for i in arr_ixs]
        return arr


def obj_fn_ix1(arrs, col_ixs, row_ixs):
    arr_sum = np.sum(arrs)
    sum_terms = 0
    prev_ro_ix = 0
    prev_col_ix = 0
    for i1 in range(1, len(col_ixs)):
        coix = col_ixs[i1]
        roix = row_ixs[i1]
        for j1 in range(prev_ro_ix, roix):
            for k1 in range(prev_col_ix, coix):
                sum_terms += arrs[j1][k1]
    return arr_sum - sum_terms


def obj_fn1(arrs, col_ix):
    n_rows = len(arrs)
    min_cst = np.inf
    cst = 0
    best_i = 0
    for i in range(n_rows):
        cst = obj_fn_ix1(arrs, col_ix, i)
        if cst < min_cst:
            min_cst = cst
            best_i = i
    return min_cst, best_i
