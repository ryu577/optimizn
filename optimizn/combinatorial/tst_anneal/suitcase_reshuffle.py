import numpy as np
from optimizn.combinatorial.anneal import OptProblem
from copy import deepcopy


class SuitCases():
	def __init__(self, config):
		"""
		The configuration of the suitcases
		is an array of arrays. The last element
		of each array must be the amount of empty space.
		This means that the sum of each array is the 
		capacity of that suitcase.
		"""
		self.config = config
		self.capacities = []
		for ar in config:
			self.capacities.append(sum(ar))


class SuitCaseReshuffle(OptProblem):
	def __init__(self, params):
		self.params = params
		self.name = "SuitcaseReshuffling"
		super().__init__()

	def get_candidate(self):
		self.candidate = self.params.config
		return self.candidate

	def cost(self, candidate):
		maxx = 0
		for ar in candidate:
			maxx = max(maxx, ar[len(ar)-1])
		return -maxx

	def next_candidate(self, candidate):
		# Find two items to swap from two different
		# suitcases.
		keep_going = True
		while keep_going:
			candidate1 = deepcopy(candidate)
			l = np.arange(len(candidate))
			cases = np.random.choice(l, size=2, replace=False)
			ix1 = np.random.choice(len(candidate[cases[0]]))
			ix2 = np.random.choice(len(candidate[cases[1]]))
			size1 = candidate[cases[0]][ix1]
			size2 = candidate[cases[1]][ix2]
			candidate1[cases[0]][ix1] = size2
			candidate1[cases[1]][ix2] = size1
			arr1 = candidate1[cases[0]]
			arr2 = candidate1[cases[1]]
			caps = self.params.capacities
			if caps[cases[0]] < sum(arr1[:len(arr1)-1])\
				or caps[cases[1]] < sum(arr2[:len(arr2)-1]):
				continue
			else:
				keep_going = False
				arr1[len(arr1)-1] = caps[cases[0]]\
									- sum(arr1[:len(arr1)-1])
				arr2[len(arr2)-1] = caps[cases[1]]\
									- sum(arr2[:len(arr2)-1])
		return candidate1


def tst1():
	config = [[7,5,1],[4,6,1]]
	sc = SuitCases(config)
	scr = SuitCaseReshuffle(params=sc)

def tst2():
	from optimizn.combinatorial.tst_anneal.suitcase_reshuffle import *

	config = [[7,5,1],[4,6,1]]
	sc = SuitCases(config)
	scr = SuitCaseReshuffle(params=sc)
	candidate = scr.get_candidate()
	scr.anneal()


