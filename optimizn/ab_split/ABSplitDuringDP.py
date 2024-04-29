
from typing import Dict, List


class SumNode:
    def __init__(self, currSum: int, idx: int, ancestor: 'SumNode'):
        self.currSum = currSum
        self.idx = idx
        self.ancestor = ancestor


def isSubsetSumRecursive(arr: List[int], arrIdx: int, currSumNode: SumNode, targetSum: int, solutions: set[SumNode], errTol=0.0):
    """
    Inputs:
    - arr: list of integers
    - arrIdx: integer representing the current index of the array we are at
    - currSumNode: CurrSumNode object
    - targetSum: integer representing the target sum we are shooting for
    - solutions: set of CurrSumNode objects, initially empty
    - errTol: allows us to find a solution that is within errTol of the targetSum. Should be a percentage from 0 to 1
    Output:
    - set of CurrSumNode objects which contain all possible valid paths to reach the target sum
    """
    if currSumNode.currSum == targetSum or abs(currSumNode.currSum - targetSum) / targetSum < errTol:
        solutions.add(currSumNode)
    if arrIdx == len(arr) or currSumNode.currSum > targetSum: return False
    if targetSum == 0 or len(arr) == 0 or errTol > 1 or errTol < 0: return False

    return isSubsetSumRecursive(arr, arrIdx + 1, SumNode(currSumNode.currSum + arr[arrIdx], arrIdx, currSumNode), targetSum, solutions, errTol) or\
        isSubsetSumRecursive(arr, arrIdx + 1, currSumNode, targetSum, solutions, errTol)


def getSolution(currSumNode: SumNode):
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
    # Remove the trailing ', '
    sol = sol[:-2]
    return (f'[{sol}]')


def ABTestSituation(features: List[List[int]], tolerance=0.0):
    """
    Inputs:
    - features: list of lists of integers
    - tolerance: float representing the error tolerance in the sum. Can be varied to allow for some error in targetSum

    This function allows to test various feature and tolerance scenarios to find a common solution for all features
    """
    targetSplit = 0.5
    solutions: Dict[float, Dict[str, int]] = dict()
    while targetSplit > 0:
        solutions[targetSplit] = dict()
        
        for feature in features:
            targetSumSolutions = set()
            targetSum = targetSplit * sum(feature)

            if targetSum % 1 != 0: 
                targetSum1 = int(targetSum)
                targetSum2 = targetSum1 + 1

                isSubsetSumRecursive(feature, 0, SumNode(0, -1, None), targetSum1, targetSumSolutions, tolerance)
                isSubsetSumRecursive(feature, 0, SumNode(0, -1, None), targetSum2, targetSumSolutions, tolerance)
            else:
                isSubsetSumRecursive(feature, 0, SumNode(0, -1, None), targetSum, targetSumSolutions, tolerance)
            
            for solution in targetSumSolutions:
                sol = getSolution(solution)
                if sol not in solutions[targetSplit]:
                    solutions[targetSplit][sol] = 1
                    continue
                
                solutions[targetSplit][sol] += 1
                if solutions[targetSplit][sol] == len(features):
                    print(f'All features have the same solution: {sol} for targetSplit: {targetSplit} +-{tolerance * 100}%')
                    return sol

        targetSplit -= 0.01
        targetSplit = round(targetSplit, 2)

    print('No common solution found')
    print(solutions)
    return sol


if __name__ == '__main__':
    arr = [3, 34, 4, 12, 5, 2, 5, 5]
    targetSum = 15
    solutions = set()
    isSubsetSumRecursive(arr, 0, SumNode(0, -1, None), targetSum, solutions)

    print("-------------------Testing isSubsetSumRecursive-------------------")
    if len(solutions) > 0:
        print('Found a subset with given sum. Here are the solution Idxs:')
        for solution in solutions:
            ans = getSolution(solution)
            print(ans)
    else:
        print('No subset with given sum')

    print("----------------------Testing ABTestSituation----------------------")
    print("\n->Testing perfect split scenario")
    features = [
        [25, 25, 25, 25],
        [15, 35, 25, 25],
        [10, 40, 25, 25],
    ]
    ABTestSituation(features)
    
    print("\n->Testing imperfect split scenario")
    features1 = [
        [33, 33, 15, 15, 3],
        [29, 37, 11, 11, 11],
        [22, 44, 11, 11, 11],
    ]
    ABTestSituation(features1)

    features2 = [
        [3, 34, 4, 12, 5, 2],
        [0, 25, 4, 12, 5, 2],
        [22, 10, 4, 12, 5, 2],
    ]
    ABTestSituation(features2)

    print("\n->Testing imperfect split scenario with tolerance")
    features1 = [
        [33, 33, 15, 15, 3],
        [29, 37, 11, 11, 11],
        [22, 44, 11, 11, 11],
    ]
    ABTestSituation(features1, tolerance=0.03)

    features2 = [
        [3, 34, 4, 12, 5, 2],
        [0, 25, 4, 12, 5, 2],
        [22, 10, 4, 12, 5, 2],
    ]
    ABTestSituation(features2, tolerance=0.15)