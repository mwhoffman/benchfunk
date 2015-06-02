"""
Benchmark problems which are sampled from a given prior distribution over
potential functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mwhutils import grid
from mwhutils import random


class PriorFunction(object):
    """
    Benchmark function which corresponds to a sample from the given prior.
    """
    def __init__(self, model, bounds, n, rng=None):
        rng = random.rstate(rng)
        bounds = np.array(bounds, ndmin=2, dtype=float)

        # get new data
        X = grid.regular(bounds, n)
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
        return self._model._post.like.sample(self.get_f(X))

    def get_f(self, X):
        """Vectorized access to the noise-free benchmark function."""
        X = np.array(X, ndmin=2, dtype=float, copy=False)
        return self._model.predict(X)[0]
