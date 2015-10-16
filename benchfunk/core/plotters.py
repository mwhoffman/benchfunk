"""
Plotting functions for a stack of experiments.
"""

import numpy as np
import os

from ezplot import figure
from .io import dump

__all__ = ['plot_stack']


def plot_stack(stack):
    """
    Plot a stack of experiments.

    Parameters:
        stack: dict or str, either the dictionary of results obtained from
            dumping or a string representing the path to the execution script.
    """

    if isinstance(stack, str):
        stack = dump(stack)

    for name, experiment in stack.items():

        fig = figure(figsize=(5, 4))
        ax = fig.add_subplot(1, 1, 1)

        for key, results in experiment.items():
            nreps, niter = np.shape(results)

            mu = np.mean(results, axis=0)
            std = np.std(results, axis=0) / np.sqrt(nreps)

            ax.plot_banded(range(1, niter+1), mu, std, label=key)

        ax.set_title(name, fontsize=14)
        ax.legend(loc=0, fontsize=14)
