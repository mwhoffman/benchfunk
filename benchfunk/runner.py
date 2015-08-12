from jug import TaskGenerator
from jug.compound import CompoundTaskGenerator

import numpy as np
import pybo


__all__ = ['run_instance', 'run_experiment', 'run_stack']


@TaskGenerator
def run_instance(problem, policy, niter, seed):
    func = problem[0](rng=seed, **problem[1])           # instantiate problem
    bounds = func.bounds
    policy, policy_kwargs = policy                      # unpack policy

    model = pybo.init_model(func, bounds)               # initialize model
    ninit = model.ndata


    R = [pybo.recommenders.best_latent(model, bounds)]  # get recommendation

    for _ in xrange(ninit, niter):
        index = policy(model, bounds, **policy_kwargs)
        xnext, _ = pybo.solvers.solve_lbfgs(index, bounds)

        model.add_data(xnext, func(xnext))
        R.append(pybo.recommenders.best_latent(model, bounds))

    data = np.zeros(niter - ninit + 1,
                    [('R', np.float, (len(bounds),)),
                     ('F', np.float)])

    data['R'] = R
    data['F'] = func.get_f(data['R'])

    return data


@CompoundTaskGenerator
def run_experiment(problem, policies, niter, nreps, script=None, name=''):
    data = dict()
    script = run_instance if script is None else script

    for key, policy in policies.items():
        namekey = '.'.join([name, key])

        data[key] = [script(problem, policy, niter, seed)
                     for seed in xrange(nreps)]

        for t in data[key]:
            t.name = namekey

    return data


@CompoundTaskGenerator
def run_stack(problems, policies, niter, nreps, script=None, name=''):
    data = dict()
    script = run_instance if script is None else script

    for key, problem in problems.items():
        namekey = '.'.join([name, key])

        data[key] = run_experiment(
            problem,
            policies,
            niter,
            nreps,
            script=script,
            name=namekey)

        data[key].name = namekey

    return data
