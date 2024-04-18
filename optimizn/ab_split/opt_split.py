import numpy as np
#from ppbtree import print_tree
from copy import deepcopy
from optimizn.ab_split.opt_split_dp import isSubsetSum
from optimizn.trees.pprnt import display
import pandas as pd
from collections import Counter


class Node1():
    def __init__(self, key):
        self.key = key
        self.val = key
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)


class Tree():
    def __init__(self, arr, mat, sum1=-1):
        """
        The mat is a dynamic programming matrix.
        """
        if sum1 < 0:
            sum1 = len(mat[0])-1
        self.arr = arr
        self.mat = mat
        self.path = None
        self.root = self.mk_tree(len(arr)-1, sum1)

    def mk_tree(self, ro, col):
        if col < 0 or ro < -1 or not self.mat[ro+1][col]:
            return
        node1 = Node1(1)
        node1.right = self.mk_tree(ro-1, col)
        node1.left = self.mk_tree(ro-1, col - self.arr[ro])
        return node1

    def find_1path(self, node, depth=0, path=[]):
        # You have to reverse the array and then pick out the indices.
        if depth > len(self.arr):
            self.path = deepcopy(path)
            return
        if node is None:
            return
        # This is only appending depths. The array will need to be reversed
        # and indexed by these depths.
        path.append(depth)
        self.find_1path(node.left, depth+1, path)
        path.pop()
        self.find_1path(node.right, depth+1, path)


def unionTrees(t1, t2):
    if (not t1):
        return t2
    if (not t2):
        return t1
    t1.left = unionTrees(t1.left, t2.left)
    t1.right = unionTrees(t1.right, t2.right)
    return t1


def intrsctTrees(t1, t2):
    if not t1 or not t2:
        return
    t1.left = intrsctTrees(t1.left, t2.left)
    t1.right = intrsctTrees(t1.right, t2.right)
    return t1


def optimize(arrs):
    matrices = []
    sums = []
    trees = []
    for arr in arrs:
        sum1 = np.sum(arr)
        matr = create_matr(arr, sum1+6)
        tree1 = Tree(arr, matr, sum1//2)
        sums.append(sum1//2)
        matrices.append(matr)
        trees.append(tree1)
    tree1 = deepcopy(trees[0])
    # Build a no-compromise tree by taking intersections.
    path1 = intrsctAllTrees2(trees)
    if path1 is not None:
        return path1
    # If no path was found, we'll have to start
    # making compromises.
    deltas = [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
    for delta in deltas:
        for ix in range(len(trees)):
            arr = arrs[ix]
            matr = matrices[ix]
            sum1 = sums[ix]
            if matr[len(arr)][sum1+delta]:
                tree2 = Tree(arr, matr, sum1+delta)
                trees[ix].root = unionTrees(trees[ix].root, tree2.root)
                path1 = intrsctAllTrees2(trees)
                if path1 is not None:
                    return path1


def intrsctAllTrees2(trees):
    tree1 = deepcopy(trees[0])
    for tree in trees:
        tree1.root = intrsctTrees(tree1.root, tree.root)
    tree1.find_1path(tree1.root)
    return tree1.path


def create_matr(arr=[3, 34, 4, 12, 5, 2], sum1=9):
    n = len(arr)
    matr = isSubsetSum(arr, n, sum1)
    return matr


def tst2():
    arrs = [[2, 5, 9, 3, 1],
            [2, 3, 4, 4, 3]]
    path1 = optimize(arrs)
    assert path1[0] == 0
    assert path1[1] == 2
    print(path1)
    arrs = [[2, 4, 7, 9],
            [1, 2, 3, 2],
            [4, 7, 5, 2]]
    path1 = optimize(arrs)
    assert path1[0] == 1
    assert path1[1] == 3
    print(path1)
    arrs = [[7, 0, 7, 0],
            [0, 5, 0, 4]]
    path1 = optimize(arrs)
    assert path1[0] == 0
    assert path1[1] == 3
    print(path1)


def tst1():
    arr = [3, 34, 4, 12, 5, 2]
    sum = 9
    n = len(arr)
    matr = isSubsetSum(arr, n, sum)
    tr = Tree(arr, matr)
    display(tr.root)
    tr.find_1path(tr.root)
    print(tr.path)
    #print_tree(tr.root)
    print("###########")
    arr = [3, 4, 5, 2]
    sum = 6
    n = len(arr)
    matr = isSubsetSum(arr, n, sum)
    tr1 = Tree(arr, matr)
    display(tr1.root)
    tr1.find_1path(tr1.root)
    print(tr1.path)
    print("###########")
    unionTrees(tr.root, tr1.root)
    display(tr.root)


def tst3():
    df = pd.read_csv('Canary_ClusterHW.csv')
    hws = [i for i in Counter(df['Hardware_Model_V2']).keys()]
    clusters = [i for i in Counter(df['Cluster']).keys()]
    arrays = []
    arr = np.zeros(len(clusters))
    for hw in hws:
        entry = df[df.Hardware_Model_V2 == hw]['dcount_NodeId'][0]


# Driver code
if __name__ == '__main__':
    tst2()
