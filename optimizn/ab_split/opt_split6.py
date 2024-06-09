import numpy as np
from optimizn.ab_split.opt_split import form_arrays
from optimizn.ab_split.opt_split4 import clean_data
from optimizn.ab_split.testing.cluster_vmsku import cluster_vmsku
from optimizn.combinatorial.simulated_annealing import SimAnnealProblem
from copy import deepcopy
import random
import csv


def get_arrays():
    arrays, vm_ix, cl_ix =\
        form_arrays(cluster_vmsku, "Usage_VMSize")
    arrays = clean_data(arrays)
    return arrays


def obj_fn_ix(arrs, col_ix, row_ix):
    n_rows = len(arrs)
    n_cols = len(arrs[0])
    sum_terms = 0
    for i1 in range(row_ix):
        for j1 in range(col_ix, n_cols):
            sum_terms += arrs[i1][j1]
    for i1 in range(row_ix, n_rows):
        for j1 in range(col_ix):
            sum_terms += arrs[i1][j1]
    return sum_terms


def obj_fn(arrs, col_ix):
    n_rows = len(arrs)
    min_cst = np.inf
    cst = 0
    best_i = 0
    for i in range(n_rows):
        cst = obj_fn_ix(arrs, col_ix, i)
        if cst < min_cst:
            min_cst = cst
            best_i = i
    return min_cst, best_i


def swap_ros(arrs, ix1, ix2):
    tmp1 = deepcopy(arrs[ix1])
    tmp2 = deepcopy(arrs[ix2])
    arrs[ix1] = tmp2
    arrs[ix2] = tmp1


def swap_cols(matrix, ix1, ix2):
    if ix1 == ix2:
        return  # No need to swap if columns are the same
    num_rows = len(matrix)
    for row in range(num_rows):
        # Swap elements at col1 and col2 using a temporary variable
        temp = matrix[row][ix1]
        matrix[row][ix1] = matrix[row][ix2]
        matrix[row][ix2] = temp


class BlockMatrix(SimAnnealProblem):
    def __init__(self, arrs):
        self.arrs = arrs
        self.candidate = arrs
        self.nros = len(arrs)
        self.ncols = len(arrs[0])
        self.col_ix = len(arrs[0])//2
        self.ro_ix = len(arrs)//2
        # params is only needed if we're persisting.
        self.params = None
        super().__init__()

    def cost(self, arrs):
        return obj_fn(arrs, self.col_ix)[0]
        # return obj_fn_ix(arrs, self.col_ix, self.ro_ix)

    def get_candidate(self):
        return self.get_solution()

    def get_solution(self):
        [ro1, ro2] = random.sample([i for i in range(self.nros)], 2)
        [co1, co2] = random.sample([i for i in range(self.ncols)], 2)
        swap_ros(self.arrs, ro1, ro2)
        swap_cols(self.arrs, co1, co2)
        return self.arrs

    def next_candidate(self, arr):
        uu = np.random.uniform()
        [ro1, ro2] = random.sample([i for i in range(self.nros)], 2)
        [co1, co2] = random.sample([i for i in range(self.ncols)], 2)
        if uu < 0.01:
            swap_ros(arr, ro1, ro2)
        elif uu < 0.02:
            swap_cols(arr, co1, co2)
        elif uu < 0.5:
            arr = sorted(arr, key=lambda x: x[co1])
        else:
            arr1 = np.transpose(arr)
            arr1 = sorted(arr1, key=lambda x: x[ro1])
            arr = np.transpose(arr1)
        return arr


def tst1(ret_f=0.5):
    arrs = get_arrays()
    cst, bsti = obj_fn(arrs, 25)
    print("Cost: " + str(cst) + " Best ix: " + str(bsti))
    arrs1 = deepcopy(arrs)
    arrs2 = read_best_mat(10)
    print("Now simulated annealing.")
    bm = BlockMatrix(arrs1)
    bm.anneal(n_iter=1000, reset_p=0.0, retarding_factor=ret_f)
    arr1 = bm.best_solution
    print(obj_fn_ix(arr1, bm.col_ix, bm.ro_ix))
    write2csv(arr1)
    sol1 = obj_fn(arr1, bm.col_ix)
    print(sol1)
    print(bm.best_cost)


def tst2():
    arrs = get_arrays()
    u, v, w = np.linalg.svd(arrs)
    arrs = arrs[:, w[0].argsort()]
    np.savetxt("arrs_tst.csv", arrs, delimiter=",")
    # https://stackoverflow.com/questions/63803033/rearrange-rows-of-a-given-numpy-2d-array-given-a-list-with-the-permutations


def write2csv(arrs):
    with open("arrs.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(arrs)


def read_best_mat(rand_no=0):
    datafile = open('arrs2.csv', 'r')
    datareader = csv.reader(datafile, delimiter=',')
    data = []
    for row in datareader:
        data.append([int(i) for i in row])
    for i in range(rand_no):
        [ro1, ro2] = random.sample([i for i in range(len(data))], 2)
        [co1, co2] = random.sample([i for i in range(len(data[0]))], 2)
        data = sorted(data, key=lambda x: x[co1])
    return data


if __name__ == "__main__":
    tst1()
