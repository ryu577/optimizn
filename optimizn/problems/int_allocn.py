import numpy as np
import cvxpy as cp
import pandas as pd

def integer_probs(p,m,n):
    """
    We want to allocate an integer vector in a way that each element
    get's atleast m, total items is n and the probabilities are as 
    close to p as possible.
    """
    x = cp.Variable(len(p),integer=True)
    constraints = [m <= x, x <= n, sum(x) == n]
    objective = cp.Minimize(cp.sum_squares(x/n - p))
    problm = cp.Problem(objective, constraints)
    _ = problm.solve()
    return x.value


def integer_probs_v2(p,m,n):
    """
    We want to allocate an integer vector in a way that each element
    get's atleast m, total items is n and the probabilities are as 
    close to p as possible.
    """
    x = cp.Variable(len(p))
    constraints = [m <= x, x <= n, sum(x) == n]
    objective = cp.Minimize(cp.sum_squares(x/n - p))
    problm = cp.Problem(objective, constraints)
    _ = problm.solve()
    h = redistribute(x.value)
    return h


def redistribute(x_value):
    """
    Given an array of floats, converts them into ints. Does
    this by taking the excess fractional part and re-distributing
    it in the same proportion.
    """
    vals = x_value // 1
    excess = int(sum(x_value % 1))
    excess_vals_unif = np.ones(len(x_value))* excess//len(x_value)
    excess_vals_nonunif = np.concatenate((np.ones(excess % len(x_value)),\
                        np.zeros(len(x_value)-excess%len(x_value)))\
                        ,axis=0)
    h = vals + excess_vals_unif + excess_vals_nonunif
    return h

