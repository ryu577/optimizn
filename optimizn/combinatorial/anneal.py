from numpy.random import uniform
from numpy import e
from copy import deepcopy


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

	def update_candidate(self, candidate, cost):
		self.candidate = make_copy(candidate)
		self.current_cost = cost

	def update_best(self, candidate, cost):
		self.best_candidate = make_copy(candidate)
		self.best_cost = cost


def make_copy(candidate):
	return deepcopy(candidate)

