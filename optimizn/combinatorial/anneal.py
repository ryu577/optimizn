from numpy.random import uniform
from numpy import e
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
		self.candidate = self.get_candidate()
		self.current_cost = self.cost(self.candidate)
		self.best_cost = self.current_cost
		self.best_candidate = make_copy(self.candidate)
		self.init_time = datetime.now()
		self.init_secs = int(self.init_time.timestamp())

	def anneal(self, n_iter=100000):
		for i in range(n_iter):
			if i % 10000 == 0:
				print("Iteration: " + str(i) + \
					" Current best solution: " + str(self.best_cost))
			eps = 0.3 * e**(-i/n_iter)
			self.new_candidate = self.next_candidate()
			self.new_cost = self.cost(self.new_candidate)
			if self.new_cost < self.best_cost:
				self.update_best(self.new_candidate, self.new_cost)
				print("Best cost updated to:" + str(self.new_cost))
			if self.new_cost < self.current_cost or eps < uniform():
				self.update_candidate(self.new_candidate,
									  self.new_cost)

	def persist(self):
		existing_obj = load_latest_pckl("Data//" + 
						self.name + "//DailyObj")
		self.obj_changed = (existing_obj == self)
		if self.obj_changed:
			# Write the latest input object that has changed.
			f_name = "Data//" + self.name + "//DailyObj//"
					 str(self.init_secs) + ".obj"
			file1 = open(f_name, 'w')
			pickle.dump(self.ntg, file1)
		# Write the optimization object.
		f_name = "Data//" + self.name + "//DailyOpt//"
					 str(self.init_secs) + ".obj"
		file1 = open(f_name, 'w')
		pickle.dump(self, file1)

		# Now check if the current best is better 
		# than the global best
		existing_best = load_latest_pckl("Data//" + 
						self.name + "//GlobalOpt")
		if existing_best is None or
			self.best_cost > existing_best.best_cost:
			f_name = "Data//" + self.name + "//GlobalOpt//"
					 str(self.init_secs) + ".obj"
			file1 = open(f_name, 'w')
			pickle.dump(self, file1)

	def update_candidate(self, candidate, cost):
		self.candidate = make_copy(candidate)
		self.current_cost = cost

	def update_best(self, candidate, cost):
		self.best_candidate = make_copy(candidate)
		self.best_cost = cost


def make_copy(candidate):
	return deepcopy(candidate)


def load_latest_pckl(path1="Data/DailyObj"):
	msh_files = os.listdir(path1)
	msh_files = sorted(msh_files)
	if len(msh_files) > 0:
		latest_file = msh_files[len(msh_files)-1]
		filehandler = open(filename, 'r')
		existing_obj = pickle.load(filehandler)
		return existing_obj
	return None


# Since this class is going to be inherited, 
# try some experiments with multiple inheritance.
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

