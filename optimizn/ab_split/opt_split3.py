import numpy as np
from optimizn.trees.pprnt import display
from optimizn.ab_split.opt_split_dp import isSubsetSum
from copy import deepcopy
from optimizn.ab_split.opt_split import create_matr, unionTrees, Tree
from optimizn.ab_split.trees.lvlTrees import Tree2, intrsctAllTrees
from optimizn.ab_split.opt_split2 import OptProblm, prepare_data, OptProblem2
from optimizn.ab_split.subsetsum.count_subsets import findCnt


def remove_zeros(arr):
    keys = [i for i in np.arange(len(arr)) if arr[i] > 0]
    keys = np.array(keys)
    arr1 = [i for i in arr if i > 0]
    arr1 = np.array(arr1)
    return arr1, keys


def optimize4(arrs):
    matrices = []
    sums = []
    trees = []
    for arr in arrs:
        sum1 = np.sum(arr)
        arr1, keys1 = remove_zeros(arr)
        matr = create_matr(arr1, sum1)
        tree1 = Tree2(arr1, matr, keys1, sum1//2)
        sums.append(sum1//2)
        matrices.append(matr)
        trees.append(tree1)
    # print("First tree before intersect")
    # display(trees[0].root)
    # deltas = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]
    delix = -1
    while True:
        delix += 1
        for delta in [delix, -delix]:
            for ix in range(len(trees)):
                arr = arrs[ix]
                arr1, keys1 = remove_zeros(arr)
                matr = matrices[ix]
                sum1 = sums[ix]
                if 0 < sum1+delta and sum1+delta < len(matr[len(arr1)]):
                    if matr[len(arr1)][sum1+delta]:
                        tree2 = Tree2(arr1, matr, keys1, sum1+delta)
                        trees[ix].root = unionTrees(trees[ix].root, tree2.root)
                        tree1 = intrsctAllTrees(trees)
                        tree1.find_best_path(tree1.root, arrs)
                        path1 = tree1.path
                        if path1 is not None and len(path1) > 0:
                            return path1


def find_a_path(arrays, matrices, targets):
    trees = []
    for ix in range(len(arrays)):
        arr = arrays[ix]
        matr = matrices[ix]
        target = targets[ix]
        tree1 = create_sparse_tree(arr, matr, target)
        trees.append(tree1)
    tree1 = intrsctAllTrees(trees)
    tree1.find_best_path(tree1.root, arrays)
    return tree1.path


def create_sparse_tree(arr, matr, target, max_paths=np.inf):
    cnt1 = findCnt(arr, len(arr), target)
    if cnt1 > max_paths:
        return None
    arr1, keys1 = remove_zeros(arr)
    a1 = [0]
    a2 = [i+1 for i in keys1]
    a1.extend(a2)
    matr1 = [matr[i] for i in a1]
    tree1 = Tree2(arr1, matr1, keys1, target)
    return tree1


def optimize7(arrays, verbose=False, max_cand=np.inf):
    op = prepare_data(arrays, find_a_path)
    if verbose:
        op.verbose = True
    op.max_cand = max_cand
    op.itr_arrays_heap()
    return op.path1


def optimize5(arrs):
    matrices = []
    sums = []
    trees = []
    op1 = prepare_data(arrs)
    for ix in range(len(arrs)):
        arr = arrs[ix]
        target = op1.target_cands[ix][0]
        sum1 = np.sum(arr)
        arr1, keys1 = remove_zeros(arr)
        matr = create_matr(arr1, sum1)
        tree1 = Tree2(arr1, matr, keys1, target)
        sums.append(sum1//2)
        matrices.append(matr)
        trees.append(tree1)
    delix = -1
    while True:
        delix += 1
        for delta in [delix, -delix]:
            for ix in range(len(trees)):
                arr = arrs[ix]
                target = op1.target_cands[ix][0]
                arr1, keys1 = remove_zeros(arr)
                matr = matrices[ix]
                sum1 = sums[ix]
                if 0 < sum1+delta and sum1+delta < len(matr[len(arr1)]):
                    if matr[len(arr1)][sum1+delta]:
                        tree2 = Tree2(arr1, matr, keys1, target+delta)
                        trees[ix].root = unionTrees(trees[ix].root, tree2.root)
                        tree1 = intrsctAllTrees(trees)
                        tree1.find_best_path(tree1.root, arrs)
                        path1 = tree1.path
                        if path1 is not None and len(path1) > 0:
                            return path1


def tst1():
    op = OptProblm()
    arrs = op.arrays
    matrices = []
    sums = []
    trees = []
    for arr in arrs:
        sum1 = np.sum(arr)
        arr1, keys1 = remove_zeros(arr)
        matr = create_matr(arr1, sum1)
        tree1 = Tree2(arr1, matr, keys1, sum1//2)
        sums.append(sum1//2)
        matrices.append(matr)
        trees.append(tree1)
    delix = -1
    valid_trees = 0
    set1 = set(np.arange(len(arrs)))
    while len(set1) > 0:
        delix += 1
        for delta in [delix, -delix]:
            for ix in range(len(trees)):
                arr = arrs[ix]
                arr1, keys1 = remove_zeros(arr)
                matr = matrices[ix]
                sum1 = sums[ix]
                if 0 < sum1+delta and sum1+delta < len(matr[len(arr1)]):
                    if matr[len(arr1)][sum1+delta]:
                        tree2 = Tree2(arr1, matr, keys1, sum1+delta)
                        trees[ix].root = unionTrees(trees[ix].root, tree2.root)
                        if tree2.root is not None:
                            print("Found tree for: " + str(ix) + " delta: " +
                                  str(delta))
                            valid_trees += 1
    return trees


def tst2():
    arrays = [
        [0, 0, 0, 6, 2, 8, 0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0]
    ]
    path1 = optimize7(arrays)
    print("Optimal path:")
    print(path1)
    return path1


if __name__ == "__main__":
    tst2()


#########################
# TODO
# 1. The optimize5 method is supposed to be optimal since it explores
#    the entire tree for the best solution. Why then is it sub-optimal
#    for problem 5.
