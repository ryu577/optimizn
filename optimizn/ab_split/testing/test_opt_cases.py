import numpy as np
from optimizn.ab_split.opt_split2 import OptProblm
from optimizn.ab_split.opt_split import optimize


class TstCases():
    def __init__(self, fn, toy_only=False):
        """
        fn has to take as input a two dimensional array. Each
        entry in that array is the input arrays.
        """
        self.fn = fn
        self.toy_only = toy_only
        op = OptProblm()
        self.inputoutput = {
            "problem1": [
                [
                    [2, 5, 9, 3, 1],
                    [2, 3, 4, 4, 3]
                ],
                [
                    [.4, .6],
                    [.4, .6]
                ]
            ],
            "problem2:": [
                [
                    [2, 4, 7, 9],
                    [1, 2, 3, 2],
                    [4, 7, 5, 2]
                ],
                [
                    [.4, .6],
                    [.4, .6],
                    [.4, .6]
                ]
            ],
            "problem3:": [
                [
                    [7, 0, 7, 0],
                    [0, 5, 0, 4]
                ],
                [
                    [.45, .55],
                    [.45, .55]
                ]
            ],
            "problem4:": [
                [
                    [7, 0, 7, 0],
                    [0, 5, 0, 4]
                ],
                [
                    [.45, .55],
                    [.45, .55]
                ]
            ],
            "problem5:": [
                op.arrays,
                [[.45, .55] for i in range(len(op.arrays))]
            ]
        }

    def tst_all(self):
        for k in self.inputoutput.keys():
            print("Trying problem " + k)
            arr = self.inputoutput[k][0]
            rnges = self.inputoutput[k][1]
            split = self.fn(arr)
            for i in range(len(arr)):
                ar = arr[i]
                rng = rnges[i]
                sum1 = sum(ar)
                cnt1 = sum([ar[ix] for ix in split])
                assert rng[0] < cnt1/sum1 and cnt1/sum1 < rng[1]
            print("Passed: " + k)


def tst1():
    tc = TstCases(optimize, True)
    tc.tst_all()
