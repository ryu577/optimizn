import numpy as np
from branch_and_bound import BnBProblem


class MinPathCoverProblem(BnBProblem):

    def __init__(self, edges1, edges2):
        self.edges1 = edges1
        self.edges2 = edges2
        self.vertices = set(edges1.flatten()).union(set(edges2.flatten()))
        self._get_all_paths()
        print(self.all_paths)
        super().__init__(init_sol=(np.ones(len(self.all_paths)), -1))

    def _get_all_paths(self):
        self.all_paths = []
        for u1, u2 in self.edges1:
            for v1, v2 in self.edges2:
                if u2 == v1:
                    self.all_paths.append([u1, v1, v2])
        self.all_paths = np.array(self.all_paths)

    def _pick_rem_paths(self, sol):
        # determine which vertices can still be covered
        choices = sol[0]
        last_idx = sol[1] + 1
        covered = set()
        for i in range(0, last_idx):
            if choices[i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        to_cover = set(self.all_paths[last_idx:].flatten()).difference(covered)

        # select paths that cover the most of the remaining vertices
        new_path_idxs = []
        new_covered = set()
        while len(to_cover.difference(new_covered)) != 0:
            path_idx = None
            path_covered = set()
            for i in range(last_idx, len(choices)):
                new_verts = set(self.all_paths[i]).intersection(
                    to_cover.difference(new_covered))
                if len(new_verts) > len(path_covered):
                    path_idx = i
                    path_covered = new_verts
            new_path_idxs.append(path_idx)
            new_covered = new_covered.union(path_covered)
        return new_path_idxs

    def lbound(self, sol):
        # sum of existing paths and (number of vertices left to cover / 3) 
        choices = sol[0]
        last_idx = sol[1] + 1
        covered = set()
        for i in range(0, last_idx):
            if choices[i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        return sum(choices[0: sol[1] + 1]) + len(self.vertices.difference(
            covered)) / 3

    def cost(self, sol):
        return sum(sol[0])

    def branch(self, sol):
        new_sols = []
        if sol[1] + 1 < len(sol[0]):
            for val in [0, 1]:
                new_sol = np.zeros(len(sol[0]))
                new_sol[0: sol[1] + 1] = sol[0][0: sol[1] + 1]
                new_sol[sol[1] + 1] = val
                for idx in self._pick_rem_paths((new_sol, sol[1] + 1)):
                    new_sol[idx] = 1
                new_sols.append((new_sol, sol[1] + 1))
        return new_sols

    def is_sol(self, sol):
        covered = set()
        for i in range(len(sol[0])):
            if sol[0][i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        return covered == self.vertices


def test_bnb_minpathcover():
    TEST_CASES = [
        (
            np.array([[1, 4], [2, 4], [2, 5], [3, 5]]),
            np.array([[4, 6], [4, 7], [5, 8]])
        )
    ]
    for i in range(len(TEST_CASES)):
        print('\n=============================')
        print(f'TEST CASE {i}')
        edges1 = TEST_CASES[i][0]
        edges2 = TEST_CASES[i][1]
        mpc = MinPathCoverProblem(edges1, edges2)
        scr, sol = mpc.solve()
        solution = []
        for j in range(len(sol[0].astype(int))):
            if sol[0][j] == 1:
                solution.append(mpc.all_paths[j])
        solution = np.array(solution)
        print(f'\nScore: {scr}\nSolution: {solution}')
        print('=============================\n')


if __name__ == '__main__':
    test_bnb_minpathcover()
