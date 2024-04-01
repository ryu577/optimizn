from optimizn.ab_split.opt_split import optimize


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
