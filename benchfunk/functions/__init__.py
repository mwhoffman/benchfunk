"""
Black-box functions to optimize.
"""

from .classics import *
from .priors import *

from . import classics
from . import priors

__all__ = []
__all__ += classics.__all__
__all__ += priors.__all__
