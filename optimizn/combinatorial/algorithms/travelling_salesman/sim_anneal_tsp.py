import numpy as np
from copy import deepcopy
from optimizn.combinatorial.simulated_annealing import SimAnnealProblem
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp


class TravSalsmn(SimAnnealProblem):
    def __init__(self, params):
        self.params = params
        super().__init__()
    
    def _get_closest_city(self, city, visited):
        # get the unvisited city closest to the one provided
        min_city = None
        min_dist = float('inf')
        dists = self.params.dists[city]
        for i in range(len(dists)):
            if i != city and i not in visited and dists[i] < min_dist:
                min_city = i
                min_dist = dists[i]
        return min_city

    def _complete_path(self, path):
        # complete the path greedily, iteratively adding the unvisited city
        # closest to the last city in the accmulated path
        visited = set(path)
        complete_path = deepcopy(path)
        while len(complete_path) != self.params.dists.shape[0]:
            if len(complete_path) == 0:
                next_city = 0
            else:
                last_city_idx = 0 if len(complete_path) == 0 else\
                    complete_path[-1]
                next_city = self._get_closest_city(last_city_idx, visited)
            visited.add(next_city)
            complete_path.append(next_city)
        return complete_path

    def get_candidate(self):
        """
        A candidate is going to be an array
        representing the order of cities
        visited.
        """
        # greedily assembled path of cities
        return np.array(self._complete_path([]))

    def reset_candidate(self):
        # random path of cities
        return np.random.permutation(np.arange(self.params.num_cities))

    def cost(self, candidate):
        tour_d = 0
        for i in range(1, len(candidate)):
            tour_d += self.params.dists[candidate[i], candidate[i-1]]
        tour_d += self.params.dists[
            candidate[0], candidate[len(candidate) - 1]]
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
    This was taken from the following source. Doesn't currently work. Need to
    debug (230108)

    Source:
    
    [1]
    Title: Latitude Longitude Distance Calculator
    Author: Luciano Miño
    URL: https://www.omnicalculator.com/other/latitude-longitude-distance
    Date accessed: January 8, 2023
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
