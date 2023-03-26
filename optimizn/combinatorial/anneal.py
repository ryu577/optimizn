from numpy.random import uniform
# from numpy import e
import numpy as np
from copy import deepcopy
from datetime import datetime
from optimizn.combinatorial.opt_problem import OptProblem


class SimAnnealProblem(OptProblem):
    def __init__(self):
        ''' Initialize the problem '''
        # Instead of always stopping at a random solution, pick
        # the best one sometimes and the best daily one other times.
        self.candidate = self.get_candidate()
        self.current_cost = self.cost(self.candidate)
        self.best_candidate = make_copy(self.candidate)
        self.best_cost = self.current_cost
        self.init_time = datetime.now()
        self.init_secs = int(self.init_time.timestamp())

    def next_candidate(self):
        ''' Switch to the next candidate.'''
        raise Exception("Not implemented")

    def anneal(self, n_iter=100000, reset_p=1/10000):
        """
        See: https://github.com/toddwschneider/shiny-salesman/blob/master/helpers.R
        And: https://toddwschneider.com/posts/traveling-salesman-with-simulated-annealing-r-and-shiny/
        """
        reset = False
        j = -1
        for i in range(n_iter):
            j = j + 1
            temprature = current_temperature(j)
            if i % 10000 == 0:
                print("Iteration: " + str(i) + " Current best solution: "
                      + str(self.best_cost))
            # eps = 0.3 * e**(-i/n_iter)
            if np.random.uniform() < reset_p:
                print("Switching to a completely random solution.")
                self.new_candidate = self.get_candidate()
                self.new_cost = self.cost(self.new_candidate)
                self.update_candidate(self.new_candidate,
                                      self.new_cost)
                print("with cost: " + str(self.current_cost))
                j = 0
                reset = True
            else:
                self.new_candidate = self.next_candidate(self.candidate)
                self.new_cost = self.cost(self.new_candidate)
            cost_del = self.cost_delta(self.new_cost, self.current_cost)
            eps = np.exp(cost_del / temprature)

            if self.new_cost < self.current_cost or eps < uniform() or reset:
                self.update_candidate(self.new_candidate,
                                      self.new_cost)
                if reset:
                    reset = False
            if self.new_cost < self.best_cost:
                self.update_best(self.new_candidate, self.new_cost)
                print("Best cost updated to:" + str(self.new_cost))

    def update_candidate(self, candidate, cost):
        self.candidate = make_copy(candidate)
        self.current_cost = cost

    def update_best(self, candidate, cost):
        self.best_candidate = make_copy(candidate)
        self.best_cost = cost


def make_copy(candidate):
    return deepcopy(candidate)


def s_curve(x, center, width):
    return 1 / (1 + np.exp((x - center) / width))


def current_temperature(iter, s_curve_amplitude=4000, 
                        s_curve_center=0, s_curve_width=3000):
    return s_curve_amplitude * s_curve(iter, s_curve_center, s_curve_width)
