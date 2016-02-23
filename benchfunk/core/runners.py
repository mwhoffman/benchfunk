"""
Functions to facilitate running repeated experiments with random seeds for
easily reproducible results.
"""

from collections import OrderedDict

__all__ = ['run_experiment', 'run_stack']


def run_experiment(name, experiment, eparams, nreps=1):
    """
    Run `experiment()`, repeated `nreps` times, on each set of kwargs in
    `eparams`.

    Parameters:
        name: str, label associated with this set of experiments.
        experiment: function, function to run on each value of eparams, must
            have a 'seed' keyword argument.
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
