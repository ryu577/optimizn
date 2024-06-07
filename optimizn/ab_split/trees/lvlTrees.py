import numpy as np
from optimizn.trees.pprnt import display
from optimizn.ab_split.opt_split_dp import isSubsetSum
from copy import deepcopy
from optimizn.ab_split.evaluation import calc_sol_delta
import pickle


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


def intrsct(n1, n2):
    if not n1 or not n2:
        return None
    if n1.key == n2.key:
        n = Node1(n1.key)
        n.left = intrsct(n1.left, n2.left)
        n.right = intrsct(n1.right, n2.right)
    elif n1.key > n2.key:
        n = Node1(n1.key)
        n.left = intrsct(n1.left, n2)
        n.right = intrsct(n1.right, n2)
    else:
        n = Node1(n2.key)
        n.left = intrsct(n1, n2.left)
        n.right = intrsct(n1, n2.right)
    return n


def toy_tree1():
    n1 = Node1(10-3)
    n2 = Node1(10-4)
    nn2 = Node1(10-4)
    n1.left = n2
    n1.right = nn2
    n5 = Node1(10-5)
    nn5 = Node1(10-5)
    n2.right = n5
    nn2.left = nn5
    n8 = Node1(10-8)
    nn8 = Node1(10-8)
    n5.left = n8
    nn5.right = nn8
    n9 = Node1(10-9)
    nn9 = Node1(10-9)
    n8.right = n9
    nn8.left = nn9
    n9.left = Node1(-1)
    nn9.right = Node1(-1)
    display(n1)
    return n1


def toy_tree2():
    n5 = Node1(10-5)
    n6 = Node1(10-6)
    n5.left = n6
    n7 = Node1(10-7)
    n6.right = n7
    n7.left = Node1(-1)
    display(n5)
    return n5


class Tree2():
    def __init__(self, arr, mat, keys, sum1=-1):
        """
        The mat is a dynamic programming matrix.
        """
        if sum1 < 0:
            sum1 = len(mat[0])-1
        self.arr = arr
        self.mat = mat
        self.path = None
        self.keys = keys
        self.mincost = np.inf
        self.stop = False
        self.root = self.mk_tree(len(mat)-2, sum1)

    def mk_tree(self, ro, col):
        # print(str(ro) + "," + str(col))
        if col < 0 or ro < -1 or not self.mat[ro+1][col]:
            return
        if ro >= 0:
            key1 = self.keys[ro]
        else:
            key1 = -1
        node1 = Node1(key1)
        # if self.mat[ro-1+1][col]:
        node1.right = self.mk_tree(ro-1, col)
        # if col - self.arr[ro] > -1 and \
        #        self.mat[ro-1+1][col - self.arr[ro]]:
        node1.left = self.mk_tree(ro-1, col - self.arr[ro])
        return node1

    def find_best_path(self, node, arrs, path=[]):
        if node is None or self.stop:
            return
        if node.key == -1:
            # self.stop = True
            objfn = calc_sol_delta(arrs, path)
            if objfn < self.mincost:
                self.mincost = objfn
                self.path = deepcopy(path)
            return
        path.append(node.key)
        self.find_best_path(node.left, arrs, path)
        path.pop()
        self.find_best_path(node.right, arrs, path)

    def find_1_path(self, node, path=[]):
        if node is None:
            return
        if node.key == -1:
            self.path = deepcopy(path)
            return
        path.append(node.key)
        self.find_1_path(node.left, path)
        path.pop()
        self.find_1_path(node.right, path)


def tst1():
    n1 = toy_tree1()
    print("########")
    n2 = toy_tree2()
    print("#########")
    nn = intrsct(n1, n2)
    print("#########")
    display(nn)


def mk_tree(arr=[6, 2, 8, 1, 2], keys=[3, 4, 5, 8, 9]):
    sum = 14
    n = len(arr)
    matr = isSubsetSum(arr, n, sum)
    tr = Tree2(arr, matr, keys)
    display(tr.root)
    return tr.root


def intrsctAllTrees(trees):
    # A None for the tree itself means that every path is a valid path.
    trees_clean = [tree for tree in trees if tree is not None]
    if len(trees_clean) == 0:
        return None
    tree1 = deepcopy(trees_clean[0])
    for tree in trees_clean:
        tree1.root = intrsct(tree1.root, tree.root)
    return tree1


def tst2():
    arr = [0, 5, 0, 4]
    arr1 = [5, 4]
    matr = [
            [True, False, False, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False, False, False],
            [True, False, False, False, False, True, False, False, False, False],
            [True, False, False, False, False, True, False, False, False, False],
            [True, False, False, False, True, True, False, False, False, True]
        ]
    matr1 = [
            [True, False, False, False, False, False, False, False, False, False],
            [True, False, False, False, False, True, False, False, False, False],
            [True, False, False, False, True, True, False, False, False, True]
        ]
    keys1 = [1, 3]
    a1 = [0]
    a2 = [i+1 for i in keys1]
    a1.extend(a2)
    matr1 = [matr[i] for i in a1]
    target = 4
    tr1 = Tree2(arr1, matr1, keys1, target)
    display(tr1.root)


def load_pkl(filename='dat1.pickle'):
    with open(filename, 'rb') as file:
        # Deserialize and load the object from the file
        loaded_object = pickle.load(file)
    return loaded_object


def tst3():
    dat1 = load_pkl('/Users/rohitpandey/Documents/github/optimizn/dat1.pickle')
    (arr1, matr1, keys1, target) = dat1
    tr = Tree2(arr1, matr1, keys1, target)
    return tr


if __name__ == "__main__":
    tst3()


# Array with too many ways to split it into half making the tree
# too large.
arr = [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  9,  9,  5,  5, 10, 10, 10,  5,
       5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  2,  5,  5,  5,  5,  2,
       2,  2,  2,  8,  6,  5,  2]
