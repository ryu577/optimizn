from queue import Queue

class BnBProblem():
    def __init__(self, init_sol): 
        self.min_cost = float('inf')
        self.best_sol = None
        self.queue = Queue()
        self.init_sol = init_sol
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

    def solve(self): 
        '''
        Executes branch and bound algorithm
        '''
        # initialization
        self.queue.put(self.init_sol)

        # explore feasible solutions
        while not self.queue.empty():
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

            # print results
            print('\nBest Solution (upper bound): ', self.min_cost, self.best_sol)
            print('Curr Solution: ', curr_sol)
            print('Lower bound:', lbound)
            print('Cost:', cost)
            print('Queue: ', list(self.queue.queue))

        # return minimum cost and best solution
        return self.min_cost, self.best_sol
