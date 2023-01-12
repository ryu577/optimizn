import numpy as np
from optimizn.combinatorial.branch_and_bound import BnBProblem
from graphing.special_graphs.neural_trigraph.rand_graph import rep_graph
from graphing.special_graphs.neural_trigraph.path_cover import \
    min_cover_trigraph


class MinPathCoverProblem1(BnBProblem):
    '''
    Solution format:
    1. Array of size p (p = number of complete paths in graph), values are 0
    if path is not used in cover, 1 if path is used in cover
    2. Last considered path index

    Branching strategy:
    Each level of the solution space tree corresponds to a path.
    Each solution branches into at most two solutions (the path is either used
    or omitted from the path cover). If the path does not cover any new
    vertices, then the branching only produces the solution where the path
    is omitted from the cover
    '''
    def __init__(self, edges1, edges2, iters_limit, print_iters, time_limit):
        self.edges1 = edges1
        self.edges2 = edges2
        self.vertices = set(edges1.flatten()).union(set(edges2.flatten()))
        self._get_all_paths()
        super().__init__(init_sol=(np.ones(len(self.all_paths)), -1),
                         iters_limit=iters_limit, print_iters=print_iters,
                         time_limit=time_limit)

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
                # do not include path in cover if no new vertices are covered
                if val == 1:
                    covered = set()
                    for i in range(0, sol[1] + 1):
                        for v in self.all_paths[i]:
                            covered.add(v)
                    if len(set(self.all_paths[sol[1] + 1])
                            .difference(covered)) == 0:
                        continue
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


class MinPathCoverProblem2(BnBProblem):
    '''
    Solution format:
    1. Path cover (paths that cover the first vertex to the last covered
    vertex)
    2. Remaining paths (paths that cover the vertices after the last covered
    vertex)
    3. Last covered vertex

    Branching strategy:
    At a level of the solution space tree that corresponds to vertex X,
    the solution nodes correspond to path covers that cover vertex X.
    Branching any of these solutions produces a new set of solutions that
    correspond to path covers that cover vertex X+1. These path covers
    either remain the same (if the path cover already covered vertex X+1)
    or include one extra path (to cover vertex X+1).
    '''
    def __init__(self, edges1, edges2, iters_limit, print_iters, time_limit):
        self.edges1 = edges1
        self.edges2 = edges2
        self.vertices = set(edges1.flatten()).union(set(edges2.flatten()))
        self.all_paths = []
        self.cov_dict = {}
        for u1, u2 in self.edges1:
            for v1, v2 in self.edges2:
                path = (u1, v1, v2)
                if u2 == v1:
                    self.all_paths.append(path)
                    for vert in path:
                        if vert not in self.cov_dict.keys():
                            self.cov_dict[vert] = set()
                        self.cov_dict[vert].add(path)
        super().__init__(init_sol=(np.zeros((0, 3)),
                         np.array(self.all_paths), min(self.vertices) - 1),
                         iters_limit=iters_limit, print_iters=print_iters,
                         time_limit=time_limit)

    def lbound(self, sol):
        # sum of existing paths and (number of vertices left to cover / 3)
        path_cover = sol[0]
        rem_verts = self.vertices.difference(set(path_cover.flatten()))
        return len(path_cover) + (len(rem_verts) / 3)

    def cost(self, sol):
        path_cover = sol[0]
        rem_paths = sol[1]
        return len(path_cover) + len(rem_paths)

    def branch(self, sol):
        # get components of solution
        path_cover = sol[0]
        last_cov_vert = sol[2]

        # if next vertex to cover has already been covered, retain
        # solution and cover the vertex after that
        new_sols = []
        new_last_cov_vert = last_cov_vert + 1
        covered = set(path_cover.flatten())
        if new_last_cov_vert in covered:
            new_sols.append((sol[0], sol[1], new_last_cov_vert))
        # otherwise, branch based on paths that can cover the next vertex,
        # complete solution by picking paths that greedily cover remaining
        # vertices
        else:
            cand_paths = np.array(list(self.cov_dict[last_cov_vert + 1]))
            for cand_path in cand_paths:
                cand_path = np.array([cand_path])
                new_path_cover = np.concatenate(
                    (path_cover, cand_path), axis=0)
                new_last_cov_vert = last_cov_vert + 1
                new_rem_paths = []
                rem_verts = self.vertices.difference(
                    set(new_path_cover.flatten()).union(
                        set(np.array(new_rem_paths).flatten())))
                while len(rem_verts) > 0:
                    opt_path = None
                    opt_cov = {}
                    for path in self.all_paths:
                        cov = rem_verts.intersection(path)
                        if len(cov) > len(opt_cov):
                            opt_path = path
                            opt_cov = rem_verts.intersection(path)
                    rem_verts = rem_verts.difference(set(opt_path))
                    new_rem_paths.append(opt_path)
                new_rem_paths = np.array(new_rem_paths)
                new_sols.append((new_path_cover, new_rem_paths,
                                 new_last_cov_vert))
        return new_sols

    def is_sol(self, sol):
        path_cover = sol[0]
        rem_paths = sol[1]
        return self.vertices == set(path_cover.flatten()).union(
            set(rem_paths.flatten()))


def test_bnb_minpathcover():
    EDGES = [
        (
            np.array([[1, 4], [2, 4], [2, 5], [3, 5]]),
            np.array([[4, 6], [4, 7], [5, 8]])
        ),
        rep_graph(8, 10, 14, reps=4),
        rep_graph(10, 14, 10, reps=4),
        rep_graph(20, 40, 20, reps=4),
        rep_graph(20, 40, 20, reps=8),
        rep_graph(40, 50, 60, reps=10)
    ]
    LENGTHS = [
        3,
        len(min_cover_trigraph(EDGES[1][0], EDGES[1][1])),
        len(min_cover_trigraph(EDGES[2][0], EDGES[2][1])),
        len(min_cover_trigraph(EDGES[3][0], EDGES[3][1])),
        len(min_cover_trigraph(EDGES[4][0], EDGES[4][1])),
        len(min_cover_trigraph(EDGES[5][0], EDGES[5][1])),
    ]
    for i in range(len(EDGES)):
        print('\n=============================')
        print(f'TEST CASE {i}')
        edges1 = EDGES[i][0]
        edges2 = EDGES[i][1]

        # first approach
        mpc1 = MinPathCoverProblem1(
            edges1, edges2, iters_limit=100, print_iters=1000,
            time_limit=120)
        scr1, sol1 = mpc1.solve()

        # second approach
        mpc2 = MinPathCoverProblem2(
            edges1, edges2, iters_limit=1e6, print_iters=1000,
            time_limit=120)
        scr2, sol2 = mpc2.solve()

        print('\nFIRST APPROACH:')
        solution1 = []
        for j in range(len(sol1[0].astype(int))):
            if sol1[0][j] == 1:
                solution1.append(mpc1.all_paths[j])
        solution1 = np.array(solution1)
        print(f'Score: {scr1}\nSolution: {solution1}')
        if len(solution1) != LENGTHS[i]:
            print(f'Paths: {len(solution1)} '
                  + f'Optimal number of paths: {LENGTHS[i]}')
        else:
            print(f'Optimal number of paths reached: {len(solution1)}')
        assert set(solution1.flatten()) == mpc1.vertices, \
            'Not all vertices covered'
        print('All vertices covered')

        print('\nSECOND APPROACH:')
        solution2 = np.concatenate((sol2[0], sol2[1]), axis=0)
        print(f'Score: {scr2}\nSolution: {solution2}')
        if len(solution2) != LENGTHS[i]:
            print(f'Paths: {len(solution2)} '
                  + f'Optimal number of paths: {LENGTHS[i]}')
        else:
            print(f'Optimal number of paths reached: {len(solution2)}')
        assert set(solution2.flatten()) == mpc2.vertices, \
            'Not all vertices covered'
        print('All vertices covered')
        print('=============================\n')


if __name__ == '__main__':
    test_bnb_minpathcover()
