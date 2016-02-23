from benchfunk import functions
from benchfunk.core import run_stack
from collections import OrderedDict
import itertools
import jug
import numpy as np

nreps = 5

@jug.TaskGenerator
def experiment(func, sn2, niter, seed):
    """Example experiment. Testing the uniform random sampling strategy."""

    # setup objective
    objective = func(sn2, rng=seed)
    bounds = objective.bounds

    ndim = len(bounds)
    xmin, xmax = bounds.T

    # uniform random sampling strategy
    rng = np.random.RandomState(seed)
    X = xmin + (xmax - xmin) * rng.rand(niter, ndim)
    y = np.full(niter, -np.inf)
    xbest = np.empty_like(X)

    for i in xrange(niter):
        y[i] = objective(X[i])
        xbest[i] = X[np.argmax(y)]

    return xbest


# prescribe functions and noise levels to test
funcs = [functions.Gramacy,
        functions.Branin]
sn2s = [0.01, 0.1]
niters = [20, 40, 80]

# list all experiments
experiments = itertools.product(funcs, sn2s, niters)

# build stack of experiments
stack = OrderedDict()
for func, sn2, niter in experiments:
    name = format(func(sn2))
    if name not in stack:
        stack[name] = OrderedDict()

    key = str(niter)    # name the particular run

    # the following keys must correspond to the kwargs of experiment()
    stack[name][key] = dict(func=func, sn2=sn2, niter=niter)

# run stack of tasks
results = run_stack(experiment, stack, nreps)
