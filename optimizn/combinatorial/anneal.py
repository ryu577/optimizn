from numpy.random import uniform
from numpy import e
import numpy as np
from copy import deepcopy
import pickle
from datetime import datetime
import os


class OptProblem():
    def cost(self, candidate):
        ''' Gets the cost for candidate solution.'''
        raise Exception("Not implemented")

    def get_candidate(self):
        ''' Gets a feasible candidate.'''
        raise Exception("Not implemented")

    def next_candidate(self):
        ''' Switch to the next candidate.'''
        raise Exception("Not implemented")

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

    def cost_delta(self, new_cost, curr_cost):
        return new_cost - curr_cost

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
                print("Iteration: " + str(i) + \
                    " Current best solution: " + str(self.best_cost))
            #eps = 0.3 * e**(-i/n_iter)
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
            eps = np.exp(cost_del/ temprature)

            if self.new_cost < self.current_cost or eps < uniform() or reset:
                self.update_candidate(self.new_candidate,
                                      self.new_cost)
                if reset:
                    reset = False
            if self.new_cost < self.best_cost:
                self.update_best(self.new_candidate, self.new_cost)
                print("Best cost updated to:" + str(self.new_cost))


    def persist(self):
        create_folders(self.name)
        existing_obj = load_latest_pckl("Data//" +\
                        self.name + "//DailyObj")
        self.obj_changed = (existing_obj == self)
        if self.obj_changed or existing_obj is None:
            # Write the latest input object that has changed.
            f_name = "Data//" + self.name + "//DailyObj//" +\
                     str(self.init_secs) + ".obj"
            file1 = open(f_name, 'wb')
            pickle.dump(self.params, file1)
            print("Wrote to DailyObj")
        # Write the optimization object.
        f_name = "Data//" + self.name + "//DailyOpt//" +\
                     str(self.init_secs) + ".obj"
        file1 = open(f_name, 'wb')
        pickle.dump(self, file1)
        print("Wrote to DailyOpt")

        # Now check if the current best is better 
        # than the global best
        existing_best = load_latest_pckl("Data//" +
                        self.name + "//GlobalOpt")
        if existing_best is None or \
            self.best_cost > existing_best.best_cost or \
            self.obj_changed:
            f_name = "Data//" + self.name + "//GlobalOpt//" +\
                     str(self.init_secs) + ".obj"
            file1 = open(f_name, 'wb')
            pickle.dump(self, file1)
            print("Wrote to GlobalOpt")

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


def load_latest_pckl(path1="Data/DailyObj"):
    msh_files = os.listdir(path1)
    msh_files = [i for i in msh_files if not i.startswith('.')]
    msh_files = sorted(msh_files)
    if len(msh_files) > 0:
        latest_file = msh_files[len(msh_files)-1]
        filehandler = open(path1 + "//" + latest_file, 'rb')
        existing_obj = pickle.load(filehandler)
        return existing_obj
    return None


def create_folders(name):
    if not os.path.exists("Data//"):
            os.mkdir("Data//")
    if not os.path.exists("Data//" + name + "//"):
        os.mkdir("Data//" + name + "//")
    if not os.path.exists("Data//" + name + "//DailyObj//"):
        os.mkdir("Data//" + name + "//DailyObj//")
    if not os.path.exists("Data//" + name + "//DailyOpt//"):
        os.mkdir("Data//" + name + "//DailyOpt//")
    if not os.path.exists("Data//" + name + "//GlobalOpt//"):
        os.mkdir("Data//" + name + "//GlobalOpt//")


# Since this class is going to be inherited, 
# trying some experiments with multiple inheritance.
# from: https://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
class First(object):
  def __init__(self):
    print("First(): entering")
    super(First, self).__init__()
    print("First(): exiting")

  def other(self):
      print("first other called")

class Second(object):
  def __init__(self):
    print("Second(): entering")
    super(Second, self).__init__()
    print("Second(): exiting")

  def other2(self):
      print("Another other")

class Third(First, Second):
  def __init__(self):
    print("Third(): entering")
    super(Third, self).__init__()
    print("Third(): exiting")

  def other(self):
      super().other()


def tst_inher():
    th = Third()
    th.other()

