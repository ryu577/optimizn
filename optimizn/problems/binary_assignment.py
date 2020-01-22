## Binary assignment for equal spoils.

import numpy as np
import cvxpy as cp
import optimizn.problems.sample_ci as sci


## https://math.stackexchange.com/questions/3515223/binary-assignment-distributing-candies/3515391#3515391

def formultn1(ci = sci.candy_preferences):
    #ci = np.array([10,7,6,3])
    x = cp.Variable(len(ci),boolean=True)
    objective = cp.Minimize(cp.sum_squares(ci@(2*x-1)))
    problm = cp.Problem(objective)
    #res = problm.solve(solver='GLPK')
    #_ = problm.solve(solver=cp.GLPK_MI)
    _ = problm.solve()
    return x.value


def formulatn2(ci = sci.candy_preferences):    
    #ci = np.array([10,7,6,3])
    z = cp.Variable()
    x = cp.Variable(len(ci),boolean=True)
    constraints = [ci@x<=z, ci@(1-x)<=z]
    #constraints = [sum(ci*x)<=z,sum(ci*(1-x))<=z]
    objective = cp.Minimize(z+0*sum(x))
    problm = cp.Problem(objective,constraints)
    _ = problm.solve(solver=cp.GLPK_MI)
    return x.value

