"""
Methods used to construct a jugfile for running benchmarks.
"""

from jug import TaskGenerator
from jug.compound import CompoundTaskGenerator

from pybo import solve_bayesopt

__all__ = ['run_instance', 'run_experiment', 'run_stack']


@TaskGenerator
def run_instance(problem, policy, niter, seed):
    """
    Default runner for a single instance (ie problem/policy pair).
    """
    # instantiate a problem and its bounds
    func = problem[0](rng=seed, **problem[1])
    bounds = func.bounds

    _, _, info = solve_bayesopt(func, bounds,
                                niter=niter, policy=policy,
                                recommender='incumbent')

    xbest = info.xbest
    fbest = func.get_f(xbest)

    return xbest, fbest


@CompoundTaskGenerator
def run_experiment(problem, policies, niter, nreps, script=None, name=''):
    """
    Default runner for a single experiment.
    """
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
    """
    Run a stack of problem instances.
    """
    data = dict()
    for key, problem in problems.items():
        namekey = '.'.join([name, key])
        data[key] = run_experiment(problem, policies, niter, nreps, script,
                                   namekey)
        data[key].name = namekey

    return data
