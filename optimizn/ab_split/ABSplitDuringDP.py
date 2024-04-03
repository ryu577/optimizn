
from typing import List


class CurrSumNode:
    def __init__(self, currSum: int, idx: int, ancestor: 'CurrSumNode'):
        self.currSum = currSum
        self.idx = idx
        self.ancestor = ancestor

def isSubsetSumRecursive(arr: List[int], arrIdx: int, currSumNode: CurrSumNode, targetSum: int, solutions: set[CurrSumNode]):
    """
    Inputs:
    - arr: list of integers
    - arrIdx: integer representing the current index of the array we are at
    - currSumNode: CurrSumNode object
    - targetSum: integer representing the target sum we are shooting for
    - solutions: set of CurrSumNode objects, initially empty
    Output:
    - set of CurrSumNode objects which contain all possible valid paths to reach the target sum
    """
    if currSumNode.currSum == targetSum:
        solutions.add(currSumNode)
    if arrIdx == len(arr) or currSumNode.currSum > targetSum: return False
    if targetSum == 0 or len(arr) == 0: return False

    return isSubsetSumRecursive(arr, arrIdx + 1, CurrSumNode(currSumNode.currSum + arr[arrIdx], arrIdx, currSumNode), targetSum, solutions) or isSubsetSumRecursive(arr, arrIdx + 1, currSumNode, targetSum, solutions)

def printSolution(currSumNode: CurrSumNode):
    """
    Inputs:
    - currSumNode: CurrSumNode object
    Output:
    - None
    """
    sol = ''
    while currSumNode.idx != -1:
        sol = str(currSumNode.idx) + ', ' + sol
        currSumNode = currSumNode.ancestor
    sol = sol[:-2]
    print(f'[{sol}]')

if __name__ == '__main__':
    arr = [3, 34, 4, 12, 5, 2, 5, 5]
    targetSum = 15
    solutions = set()
    isSubsetSumRecursive(arr, 0, CurrSumNode(0, -1, None), targetSum, solutions)

    if len(solutions) > 0:
        print('Found a subset with given sum. Here are the solution Idxs:')
        for solution in solutions:
            printSolution(solution)
    else:
        print('No subset with given sum')
    # Output: 'Found a subset with given sum