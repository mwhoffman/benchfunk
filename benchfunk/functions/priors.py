"""
Benchmark problems which are sampled from a given prior distribution over
potential functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..utils import rstate


__all__ = ['PriorFunction']


class PriorFunction(object):
    """
    Benchmark function which corresponds to a sample from a given generative
    model. The model should define methods `sample`, `predict`, and `add_data`,
    which correspond to sampling, predicting, and posterior updating. The
    prediction step should return marginal mean and a variance estimates.
    Finally, the model should also define a likelihood via `model.like` which
    can be sampled from.
    """
    def __init__(self, model, bounds, n, rng=None):
        rng = rstate(rng)
        bounds = np.array(bounds, ndmin=2, dtype=float)
        d = len(bounds)

        # get new data
        X = np.meshgrid(*(np.linspace(a, b, n) for a, b in bounds))
        X = np.reshape(X, (d, -1)).T
        Y = model.sample(X, rng=rng)

        # store everything
        self.bounds = bounds
        self._model = model.copy()

        # add the new data
        self._model.add_data(X, Y)

    def __call__(self, x):
        return self.get(x)[0]

    def get(self, X):
        """Vectorized access to the benchmark function."""
        # FIXME: I don't like this. It requires you to know the internal
        # workings of the model.
        return self._model.like.sample(self.get_f(X))

    def get_f(self, X):
        """Vectorized access to the noise-free benchmark function."""
        X = np.array(X, ndmin=2, dtype=float, copy=False)
        return self._model.predict(X)[0]
