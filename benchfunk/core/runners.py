"""
Example script which runs a number of policies on a single problem with varying
levels of observation noise.
"""

from collections import OrderedDict
from jug import TaskGenerator

from pybo import solve_bayesopt
from pybo.domains import Discrete, Box

__all__ = ['run_instance', 'run_experiment', 'run_stack']


@TaskGenerator
def run_instance(problem, model, policy, niter, seed):
    """
    Default runner for a single instance (ie problem/policy pair). With the
    exception of the `seed` keyword, all other arguments passed to
    run_instance() can be user-defined as long as the same interface (keywords)
    is used at runtime. In other words, each element of the dictionary that
    is passed to run_experiment() should itself be a dictionary with all the
    keywords in run_instance().

    """
    # instantiate the problem
    func, kwargs = problem
    func = func(rng=seed, **kwargs)

    # get the function's bounds/domain
    if hasattr(func, 'bounds'):
        domain = Box(func.bounds)
    else:
        domain = Discrete(func.X)

    # solve the problem using the prescribed model and policy
    _, _, info = solve_bayesopt(func,
                                domain,
                                niter=niter,
                                model=model,
                                policy=policy,
                                recommender='incumbent')

    # obtain the results
    xbest = info.xbest
    fbest = func.get_f(xbest)

    return fbest


def run_experiment(name, experiment, eparams, nreps=1):
    """
    Run `experiment()`, repeated `nreps` times, on each set of kwargs in
    `eparams`.

    Parameters:
        name: str, label associated with this set of experiments.
        experiment: function, function to run on each value of eparams.
        eparams: dict, named list (dict) of experimental parameters (dicts) to
            be passed to experiment() function.
        nrep: int, number of repetitions of each instance to run, default 1.

    Returns:
        results: dict, dictionary of jug tasks.
    """
    results = OrderedDict()

    for task, kwargs in eparams.items():
        # create the subtasks
        results[task] = [experiment(seed=seed, **kwargs)
                         for seed in xrange(nreps)]

        # rename the subtasks so they are easily viewable
        for t in results[task]:
            t.name = ':'.join([name, task])

    return results


def run_stack(experiment, stack, nreps=1):
    """
    Runs `experiment()` on a stack of named experiments passed via the
    dictionary stack. Each key should correspond to a meaningful grouping of
    experiments.

    Parameters:
        experiment: function, function to run on each value of stack.
        stack: dict, stack of parameters to pass to experiment(), grouped and
            named in a dictionary.
        nreps: int, number of repetitions of each instance to run, default 1.

    Returns:
        results: dict, dictionary of dictionary of jug tasks, corresponding to
            a prescribed semantic grouping of experiments.
    """
    results = OrderedDict()

    for name, eparams in stack.items():
        results[name] = run_experiment(name, experiment, eparams, nreps)

    return results
