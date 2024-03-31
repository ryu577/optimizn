import numpy as np
from ppbtree import Node, add, print_tree
from copy import deepcopy
from optimizn.ab_split.opt_split_dp import isSubsetSum
from optimizn.trees.pprnt import display


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
    def __init__(self, arr, mat):
        """
        The mat is a dynamic programming matrix.
        Comes from 240329.py isSubsetSum.
        """
        self.arr = arr
        self.mat = mat
        self.root = self.mk_tree(len(arr)-1, len(mat[0])-1)

    def mk_tree(self, ro, col):
        if col < 0 or ro < -1 or not self.mat[ro+1][col]:
            return
        node1 = Node1(1)
        node1.right = self.mk_tree(ro-1, col)
        node1.left = self.mk_tree(ro-1, col - self.arr[ro])
        return node1

    def find_1path(self, node, depth=0, path=[]):
        if depth > len(arr):
            self.path = deepcopy(path)
            return
        if node is None:
            return
        path.append(depth)
        self.find_1path(node.left, depth+1, path)
        path.pop()
        self.find_1path(node.right, depth+1, path)


def combTrees(t1, t2):
    if (not t1):
        return t2
    if (not t2):
        return t1
    t1.left = intrsctTrees(t1.left, t2.left)
    t1.right = intrsctTrees(t1.right, t2.right)
    return t1


def intrsctTrees(t1, t2):
    if not t1 or not t2:
        return
    t1.left = intrsctTrees(t1.left, t2.left)
    t1.right = intrsctTrees(t1.right, t2.right)
    return t1


# Driver code
if __name__ == '__main__':
    arr = [3, 34, 4, 12, 5, 2]
    sum = 9
    n = len(arr)
    matr = isSubsetSum(arr, n, sum)
    tr = Tree(arr, matr)
    display(tr.root)
    tr.find_1path(tr.root)
    print(tr.path)
    print_tree(tr.root)
