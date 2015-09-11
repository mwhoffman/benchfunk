"""
Plotting functions for a stack of experiments.
"""

import matplotlib
matplotlib.use('pdf')

import numpy as np

from ezplot import figure
from jug import TaskGenerator

__all__ = ['plot_stack']


@TaskGenerator
def plot_stack(stack_results, problems=None, policies=None, name=''):
    """
    Plot a single stack of experiments.
    """
    problems = problems if problems is not None else stack_results.key()
    nfigs = len(problems)

    fig = figure(figsize=(5*nfigs, 4))

    for i, expt in enumerate(problems):
        results = stack_results[expt]
        policies = policies if policies is not None else results.keys()

        ax = fig.add_subplot(1, nfigs, i+1)

        for policy in policies:
            _, fbest = zip(*results[policy])
            iters = np.arange(np.shape(fbest)[1])

            mu = np.mean(fbest, axis=0)
            std = np.std(fbest, axis=0) / np.sqrt(len(fbest))
            ax.plot_banded(iters, mu, std, label=policy)

        ax.set_title(expt, fontsize=16)

    ax.legend(loc=0, fontsize=16)
    fig.savefig(name)
