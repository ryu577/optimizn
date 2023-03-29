import numpy as np
from copy import deepcopy
from optimizn.combinatorial.anneal import SimAnnealProblem
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp

# pip install python-tsp
# https://github.com/fillipe-gsm/python-tsp
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search


class CityGraph():
	def __init__(self, num_cities=50):
		# Generate x-y coordinates of some cities.
		# Here, we just draw them from a normal dist.
		self.xs = np.random.normal(loc=0,scale=5,size=(num_cities,2))
		self.num_cities = len(self.xs)
		self.dists = np.zeros((len(self.xs), len(self.xs)))
		# Populate the matrix of euclidean distances.
		for i in range(len(self.xs)):
			for j in range(i+1, len(self.xs)):
				dist = (self.xs[i][0]-self.xs[j][0])**2
				dist += (self.xs[i][1]-self.xs[j][1])**2
				dist = np.sqrt(dist)
				self.dists[i,j] = dist
		for i in range(len(self.xs)):
			for j in range(i):
				self.dists[i,j] = self.dists[j,i]


class TravSalsmn(SimAnnealProblem):
	"""
	Finding the min path cover of a neural trigraph.
	"""
	def __init__(self, params):
		self.params = params
		super().__init__()

	def get_candidate(self):
		"""
		A candidate is going to be an array
		representing the order of cities
		visited.
		"""
		self.candidate = np.random.permutation(
			np.arange(self.params.num_cities))
		return self.candidate

	def cost(self, candidate):
		tour_d = 0
		for i in range(1, len(candidate)):
			tour_d += self.params.dists[candidate[i], candidate[i-1]]
		return tour_d

	def next_candidate(self, candidate):
		nu_candidate = deepcopy(candidate)
		swaps = np.random.choice(
			np.arange(len(candidate)), size=2, replace=False)
		to_swap = nu_candidate[swaps]
		nu_candidate[swaps[0]] = to_swap[1]
		nu_candidate[swaps[1]] = to_swap[0]
		return nu_candidate


def dist_from_lat_long(lat1, long1, lat2, long2):
	"""
	Taken from: https://www.omnicalculator.com/other/latitude-longitude-distance
	Doesn't currently work. Need to debug (230108)
	"""
	theta1 = lat1
	theta2 = lat2
	phi1 = long1
	phi2 = long2
	r = 6400
	dtheta1 = (theta2-theta1)/2
	dtheta1 = np.sin(dtheta1)**2
	dtheta2 = np.cos(theta1)*np.cos(theta1)
	dtheta2 *= np.sin((phi2-phi1)/2)**2
	d = np.sqrt(dtheta1+dtheta2)
	d = np.arcsin(d)
	dist = 2*r*d
	return d


def tst1():
	# import optimizn.combinatorial.tst_anneal.trav_salsmn as ts
	tt = CityGraph()
	ts1 = TravSalsmn(tt)
	print("Best solution with external library: ")
	#permutation, distance = solve_tsp_dynamic_programming(tt.dists)
	permutation, distance = solve_tsp_local_search(tt.dists)
	print(distance)
	ts1.anneal()
	print("Best solution: " + str(ts1.best_cost))
	return ts1


## https://developers.google.com/optimization/routing/tsp
# Their solution didn't work, some cpp error.


