import numpy as np


def findCnt(a: list, n: int, target: int):
    # Initializing the matrix
    tab = [[0] * (target + 1) for i in range(n + 1)]
    tab[0][0] = 1
    for i in range(1, target + 1):
        tab[0][i] = 0
    for i in range(1, n+1):
        for j in range(target + 1):
            if a[i-1] <= j:
                tab[i][j] = tab[i-1][j] + tab[i-1][j-a[i-1]]
            else:
                tab[i][j] = tab[i-1][j]
    return tab[n][target]


# Driver code
if __name__ == "__main__":
    arr = [3, 3, 3, 3]
    n = len(arr)
    x = 6
    print(findCnt(arr, n, x))
