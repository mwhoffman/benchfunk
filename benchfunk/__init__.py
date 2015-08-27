"""
Benchmark functions for global optimization. Each of the functions can be
instantiated with a call of the form `f = Function(sn2)` where sn2 is the noise
variance (which defaults to zero). If the optional integer parameter rng is
given to the function this specifies a random seed that the noise will be
generated from, if not given it will use the global numpy random state.
Finally, if the optional minimize flag is True, then the function will be
negated so that it can be used as a minimization problem.

NOTE: all problems in this set of benchmarks correspond to maximization
problems.

Once instantiated each problem instance has the following attributes:

    - ndim: the number of dimensions;
    - bounds: the problem bounds, as a (ndim, 2)-array;
    - xopt: the location of a global optimizer.

And the function can be queried either by calling `f(x)` or `f.get(X)` for
vectorized access. Also available is `f.get_f(X)` for access to the noise-free
function.
"""

from .classics import *
from .priors import *

from . import classics
from . import priors

__all__ = []
__all__ += classics.__all__
