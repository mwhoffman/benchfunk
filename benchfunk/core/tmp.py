"""
Example script which runs a number of policies on a single problem with varying
levels of observation noise.
"""

import numpy as np

from collections import OrderedDict
from jug import TaskGenerator
from jug.compound import CompoundTaskGenerator

from pybo import solve_bayesopt
from pybo.domains import Discrete, Box
from benchfunk import functions


@TaskGenerator
def run_instance(problem, kwargs, strategy, niter, seed):
    """
    Default runner for a single instance (ie problem/policy pair).
    """
    # instantiate a problem and grab its bounds
    func = problem(rng=seed, **kwargs)

    if hasattr(func, 'bounds'):
        domain = Box(func.bounds)
    else:
        domain = Discrete(func.X)

    # solve the problem
    _, _, info = solve_bayesopt(func, domain, niter=niter, **strategy)

    # obtain the results
    xbest = info.xbest
    fbest = func.get_f(xbest)

    return fbest


def run_experiment(name, problem, kwargs, strategies, niter, nreps):
    """
    Default runner for a single experiment.
    """
    results = dict()

    for key, strategy in strategies.items():
        # create the subtasks
        results[key] = [run_instance(problem, kwargs, strategy, niter, seed)
                        for seed in xrange(nreps)]

        # rename the subtasks so they are easily viewable
        for t in results[key]:
            t.name = ':'.join([name, key])

    return results


def run_stack(stack):
    results = dict()
    for name, (problem, kwargs, strategies, niter, nreps) in stack.items():
        results[name] = run_experiment(name, problem, kwargs, strategies,
                                       niter, nreps)
    return results

