from queue import PriorityQueue

class BnBProblem():
    def __init__(self, init_sol): 
        self.best_score = float('inf')
        self.best_sol = None
        self.queue =  PriorityQueue()
        self.init_sol = init_sol

    def lbound(self, sol): 
        raise NotImplementedError('Implement a function to compute a lower '\
            + 'bound on a feasible solution') 

    def score(self, sol): 
        raise NotImplementedError('Implement a function to compute a score '\
            + 'for a feasible solution') 

    def branch(self, sol):
        raise NotImplementedError('Implement a function to produce other '\
            + 'feasible solutions from a single feasible solution') 

    def is_sol(self, sol): 
        raise NotImplementedError('Implement a function to check '\
            + 'if a solution is a feasible solution') 

    def process(self): 
        self.queue.put((self.lbound(self.init_sol), self.init_sol))
        while not self.queue.empty():
            lbound, curr_sol = self.queue.get()
            if lbound >= self.best_score: 
                continue 
            score = self.score(curr_sol)
            if self.best_score > score: 
                self.best_score = score 
                self.best_sol = curr_sol 
            if score != lbound: 
                next_sols = self.branch(curr_sol)
                for next_sol in next_sols: 
                    self.queue.put((self.lbound(next_sol), next_sol))
        return self.best_score, self.best_sol