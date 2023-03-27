import time
from queue import PriorityQueue
from optimizn.combinatorial.opt_problem import OptProblem
from copy import deepcopy

# References:
# https://imada.sdu.dk/~jbj/heuristikker/TSPtext.pdf
class BnBProblem(OptProblem):
    def __init__(self, params):
        self.params = params
        self.queue = PriorityQueue()
        self.total_iters = 0
        self.total_time_elapsed = 0
        super().__init__()
        if not self.is_sol(self.best_solution):
            raise Exception('Initial solution is infeasible')

    def lbound(self, sol):
        '''
        Computes lower bound for a solution and the feasible solutions that
        can be obtained from it
        '''
        raise NotImplementedError(
            'Implement a function to compute a lower bound on a feasible '
            + 'solution')

    def branch(self, sol):
        '''
        Generates other potential solutions from an existing feasible solution
        '''
        raise NotImplementedError(
            'Implement a function to produce other potential solutions from a '
            + 'single feasible solution')

    def is_sol(self, sol):
        '''
        Checks if potential solution is feasible solution or not
        '''
        raise NotImplementedError(
            'Implement a function to check if a solution is a feasible '
            + 'solution')

    def _print_results(self, iters, print_iters, time_elapsed, force=False):
        if force or iters == 1 or iters % print_iters == 0:
            print(f'\nSolutions explored (current run): {iters}')
            print(f'Solutions explored (total): {self.total_iters}')
            queue = list(self.queue.queue)
            print(f'Queue size: {len(queue)}')
            #  print(f'Queue: {queue}')
            print(f'Time elapsed (current run): {time_elapsed} seconds')
            print(f'Time elapsed (total): {self.total_time_elapsed} seconds')
            print(f'Best solution: {self.best_solution}')
            print(f'Score: {self.best_cost}')

    def solve(self, iters_limit=1e6, print_iters=100, time_limit=3600):
        '''
        Executes branch and bound algorithm
        '''
        # initialization
        start = time.time()
        iters = 0
        time_elapsed = 0
        original_total_time_elapsed = deepcopy(self.total_time_elapsed)
        sol_count = 1  # breaks ties between solutions with same lower bound
        # solutions generated earlier are given priority in such cases

        # if problem class instance is loaded, queue is saved as list, so
        # convert back to PriorityQueue
        if type(self.queue) is not PriorityQueue:
            queue = PriorityQueue()
            while not self.queue.empty():
                queue.put(self.queue.get())
        # otherwise, queue is created as PriorityQueue, so put initial solution
        # onto PriorityQueue
        else:
            self.queue.put((self.lbound(self.best_solution),
                            sol_count, self.best_solution))

        # explore feasible solutions
        while not self.queue.empty() and iters != iters_limit and\
                time_elapsed < time_limit:
            # get feasible solution
            lbound, _, curr_sol = self.queue.get()

            # move to next feasible solution if lowest possible cost of
            # current feasible solution and branched solutions is not
            # lower then the lower cost already seen
            if self.cost_delta(self.best_cost, lbound) > 0:
                # score current solution, update minimum cost and best solution
                cost = self.cost(curr_sol)
                if self.cost_delta(self.best_cost, cost) > 0:
                    self.best_cost = cost
                    self.best_solution = curr_sol

                # if lower bound not yet reached, consider other feasible
                # solutions
                if self.cost_delta(cost, lbound) > 0:
                    next_sols = self.branch(curr_sol)
                    for next_sol in next_sols:
                        if self.is_sol(next_sol):
                            lbound = self.lbound(curr_sol)
                            if self.cost_delta(self.best_cost, lbound) > 0:
                                sol_count += 1
                                self.queue.put((lbound, sol_count, next_sol))

            # print best solution and min cost, check if time limit exceeded
            iters += 1
            self.total_iters += 1
            time_elapsed = time.time() - start
            self.total_time_elapsed = original_total_time_elapsed +\
                time_elapsed
            self._print_results(iters, print_iters, time_elapsed)

        # return best solution and cost
        self._print_results(iters, print_iters, time_elapsed, force=True)

        # convert the queue to a list before saving solution
        self.queue = list(self.queue.queue)
        self.persist()
        return self.best_solution, self.best_cost
