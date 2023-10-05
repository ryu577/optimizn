def check_bnb_sol(bnb_instance, init_cost, bnb_type, params):
    # determine BnB type
    if bnb_type == 0:
        bnb_alg = 'traditional'
    else:
        bnb_alg = 'modified'

    # check that final solution is complete and feasible
    assert bnb_instance.is_complete(bnb_instance.best_solution), 'Final '\
        + f'solution ({bnb_instance.best_solution}) is not complete. '\
        + f'Algorithm: {bnb_alg} branch and bound. Params: {params}'
    assert bnb_instance.is_feasible(bnb_instance.best_solution), 'Final '\
        + f'solution ({bnb_instance.best_solution}) is not feasible. '\
        + f'Algorithm: {bnb_alg} branch and bound. Params: {params}'

    # check that final solution is not worse than initial solution
    assert bnb_instance.best_cost <= init_cost, 'Final solution is less '\
        + f'optimal than initial solution. Cost of initial solution: '\
        + f'{init_cost}. Cost of final solution: {bnb_instance.best_cost}. '\
        + f'Algorithm: {bnb_alg} branch and bound. Params: {params}'


def check_sol(sol, opt_sol):
    assert sol == opt_sol, 'Final solution is not optimal solution. Final '\
        + f'solution: {sol}. Optimal solution: {opt_sol}'


def check_sol_optimality(sol_cost, opt_sol_cost, ratio=1.0):
    assert sol_cost <= opt_sol_cost * ratio, 'Final solution cost '\
        + f'({sol_cost}) is greater than {ratio} * '\
        + f'optimal solution cost, where optimal solution cost '\
        + f'= {opt_sol_cost}'
