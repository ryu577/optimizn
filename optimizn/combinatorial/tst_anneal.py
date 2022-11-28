from optimizn.combinatorial.anneal import OptProblem
from graphing.special_graphs.neural_trigraph.rand_graph import *
from graphing.graph import Graph
from graphing.traversal.clr_traversal import Graph1
from graphing.special_graphs.neural_trigraph.path_cover import\
	complete_paths
from graphing.special_graphs.neural_trigraph.neural_trigraph import\
	NeuralTriGraph


# For demonstration purposes. 
# We pick the min path cover problem 
# where we have an algorithm for computing
# the optimal solution.
class MinPathCover_NTG(OptProblem):
	"""
	Finding the min path cover of a neural trigraph.
	"""
	def __init__(self, ntg):
		self.ntg = ntg
		self.adj = ntg.g1.adj

	def get_candidate(self):
		"""
		A candidate is going to be an array of
		arrays, where each array is a full path
		from the left-most layer of the graph
		to the right-most layer.
		"""
		paths = []
		self.covered = {}
		for i in range(1, self.ntg.max_ix+1):
			if i not in self.covered:
				self.add_path(i)
				paths.append(path[0])
		self.candidate = paths
		return paths

	def next_candidate(self):
		ix = np.random.choice(range(len(paths)))
		del_path = paths[ix]
		paths = np.delete(paths, ix, 0)
		for ixx in del_path:
			if self.covered[ixx] == 1:
				path = complete_paths([[ixx]], 
					self.ntg.left_edges, self.ntg.right_edges)
				self.add_path(ixx)

	def add_path(self, i):
		path = complete_paths([[i]], 
			self.ntg.left_edges, self.ntg.right_edges)
		for j in path[0]:
			if j in self.covered:
				self.covered[j] += 1
			else:
				self.covered[j] = 1

	def cost(self, candidate):
		''' 
		Gets the cost for candidate solution.
		'''
		return(len(candidate))


# Scipy simulated annealing can't be used 
# because it expects a 1-d array
# and probably permutes the 1-d array.
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html


edges1, edges2 = neur_trig_edges(5, 7, 9)
ntg = NeuralTriGraph(edges1, edges2)
print(ntg.g1.adj)
mpc = MinPathCover_NTG(ntg)
paths = mpc.get_candidate()
#Now, can we get the min path cover for this?

