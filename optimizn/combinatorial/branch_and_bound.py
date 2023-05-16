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
        if not self.is_feasible(self.best_solution) or not self.is_complete(
                self.best_solution):
            raise Exception('Initial solution is infeasible or incomplete: '
                            + f'{self.best_solution}')
        print(f'Initial solution: {self.best_solution}')
        print(f'Initial solution cost: {self.best_cost}')

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

    def is_complete(self, sol):
        '''
        Checks if a potential solution is a complete solution
        '''
        raise NotImplementedError(
            'Implement a function to check if a solution is a complete '
            + 'solution')

    def is_feasible(self, sol):
        '''
        Checks if potential solution is a feasible solution
        '''
        raise NotImplementedError(
            'Implement a function to check if a solution is a feasible '
            + 'solution')

    def complete_solution(self, sol):
        '''
        Completes an incomplete solution using a heuristic for early detection
        of optimal/close-to-optimal solutions (returns None) by default
        '''
        raise NotImplementedError(
            'Implement a function to complete an incomplete solution using a'\
            + ' heuristic')

    def _print_results(self, iters, print_iters, time_elapsed, force=False):
        if force or iters == 1 or iters % print_iters == 0:
            print(f'\nIterations (current run): {iters}')
            print(f'Iterations (total): {self.total_iters}')
            queue = list(self.queue.queue)
            print(f'Queue size: {len(queue)}')
            print(f'Time elapsed (current run): {time_elapsed} seconds')
            print(f'Time elapsed (total): {self.total_time_elapsed} seconds')
            print(f'Best solution: {self.best_solution}')
            print(f'Score: {self.best_cost}')

    def _update_best_solution(self, sol):
        # get cost of solution and update minimum cost and best solution
        # if needed
        cost = self.cost(sol)
        if self.cost_delta(self.best_cost, cost) > 0:
            self.best_cost = cost
            self.best_solution = sol

    def solve(self, iters_limit=1e6, print_iters=100, time_limit=3600,
              bnb_type=0):
        '''
        Executes either the traditional (bnb_type=0) or modified (bnb_type=1)
        branch and bound algorithm
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
            for item in self.queue:
                queue.put(item)
            self.queue = queue
        # otherwise, queue is created as PriorityQueue, so put initial solution
        # onto PriorityQueue
        else:
            self.queue.put((self.lbound(self.best_solution), sol_count,
                            self.best_solution))

        # explore solutions
        while not self.queue.empty() and iters != iters_limit and\
                time_elapsed < time_limit:
            # get solution
            lbound, _, curr_sol = self.queue.get()

            # get and process branched solutions
            next_sols = self.branch(curr_sol)
            for next_sol in next_sols:
                # skip infeasible solutions
                if not self.is_feasible(next_sol):
                    continue

                # compute lower bound of branched solution
                lbound = self.lbound(next_sol)

                # process branched solution
                if self.is_complete(next_sol):
                    # if solution is complete, update best solution and best
                    # solution cost if needed
                    self._update_best_solution(next_sol)
                else:
                    # if algorithm type is 1 (modified branch and bound), then
                    # complete solution and update best solution and best
                    # solution cost if needed
                    if bnb_type == 1:
                        completed_sol = self.complete_solution(next_sol)
                        self._update_best_solution(completed_sol)

                    # if lower bound is less than best solution cost, put
                    # incomplete, feasible solution into queue
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
        return self.best_solution, self.best_cost

    def persist(self):
        # convert the queue to a list before saving solution
        self.queue = list(self.queue.queue)
        super().persist()
