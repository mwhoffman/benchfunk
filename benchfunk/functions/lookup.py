"""
Benchmark function from "Practical Bayesian optimization of machine learning
algorithms." The LDA example tunes the online latent Dirichlet allocation, of
Hoffman et al., applied to Wikipedia articles, while the SVM example tunes a
support vector machine. In order to save computation, the algorithms'
performances were precomputed on a regular grid and saved in the attached file
lda.csv and svm.csv, respectively.

Data available thanks to:
    Katharina Eggensperger,
    Matthias Feurer,
    Jasper Snoek,
    Ryan P. Adams, and
    Hugo Larochelle.

Important note:
In the LDA dataset reported by HPOlib, the entry for (0.5, 1024, 64) was
duplicated so the following row was removed:

    0.5, 1024, 64, 1520.064450, 10877.800000
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os.path

__all__ = ['LookupTable', 'LDA', 'SVM']


class LookupTable(object):
    def __init__(self, filename, delimiter=','):
        # load dataset
        data = np.loadtxt(filename, delimiter=delimiter)

        # last two columns are kept for targets and runtimes
        self.ndim = data.shape[1] - 2
        self._lookup = dict()

        # populate values using convention that first ndim columns are the
        # inputs and ndim'th column is the target
        for row in data:
            key = tuple(row[:self.ndim])
            # FIXME: turn this into a maximization problem. I don't like this,
            # but it simplifies things for the time being.
            self._lookup[key] = row[self.ndim]

        # expose the keys. I'm not necessarily sold on the name though...
        self.X = np.array(self._lookup.keys())

    def __call__(self, x):
        x = tuple(x)
        if x in self._lookup:
            return -self._lookup[x]
        else:
            raise KeyError('no table entry for input x = {}'.format(x))


class LDA(LookupTable):
    def __init__(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'lda.csv')
        super(LDA, self).__init__(fname)


class SVM(LookupTable):
    def __init__(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'svm.csv')
        super(SVM, self).__init__(fname)
