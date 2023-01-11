import time
from queue import Queue


class BnBProblem():
    def __init__(self, init_sol, iters_limit=1e6, print_iters=100,
                 time_limit=3600):
        self.min_cost = float('inf')
        self.best_sol = None
        self.queue = Queue()
        self.iters_limit = iters_limit
        self.print_iters = print_iters
        self.time_limit = time_limit
        self.init_sol = init_sol
        self.iters = 0
        self.time_elapsed = 0
        if not self.is_sol(init_sol):
            raise Exception('Initial solution is infeasible')

    def lbound(self, sol):
        '''
        Computes lower bound for a solution and the feasible solutions 
        that can be obtained from it
        '''
        raise NotImplementedError('Implement a function to compute a lower '
            + 'bound on a feasible solution')

    def cost(self, sol):
        '''
        Computes the cost of a solution
        '''
        raise NotImplementedError('Implement a function to compute a cost '
            + 'for a feasible solution')

    def branch(self, sol):
        '''
        Generates other potential solutions from an existing feasible solution
        '''
        raise NotImplementedError('Implement a function to produce other '
            + 'potential solutions from a single feasible solution')

    def is_sol(self, sol):
        '''
        Checks if potential solution is feasible solution or not
        '''
        raise NotImplementedError('Implement a function to check '
            + 'if a solution is a feasible solution')

    def _print_results(self):
        if self.iters == 1 or self.iters % self.print_iters == 0:
            print(f'\nSolutions explored: {self.iters}')
            print(f'Time elapsed: {self.time_elapsed} seconds')
            print(f'Best solution: {self.best_sol}')
            print(f'Score: {self.min_cost}')

    def solve(self):
        '''
        Executes branch and bound algorithm
        '''
        # initialization
        start = time.time()
        self.queue.put(self.init_sol)

        # explore feasible solutions
        while not self.queue.empty() and self.iters != self.iters_limit:
            # get feasible solution
            curr_sol = self.queue.get()

            # do not explore current solution if lowest possible cost is higher 
            # than minimum cost
            lbound = self.lbound(curr_sol)
            if lbound >= self.min_cost:
                continue

            # score current solution, update minimum cost and best solution
            cost = self.cost(curr_sol)
            if self.min_cost > cost:
                self.min_cost = cost
                self.best_sol = curr_sol

            # if lower bound not yet reached, explore other feasible solutions
            if cost != lbound:
                next_sols = self.branch(curr_sol)
                for next_sol in next_sols:
                    if self.is_sol(next_sol):
                        self.queue.put(next_sol)

            # print best solution and min cost, check if time limit exceeded
            self.iters += 1
            self.time_elapsed = time.time() - start
            if self.iters == 1 or (self.iters > 0 and self.iters % 10 == 0):
                self._print_results()
            if self.time_elapsed > self.time_limit:
                break

        # return minimum cost and best solution
        self._print_results()
        return self.min_cost, self.best_sol
