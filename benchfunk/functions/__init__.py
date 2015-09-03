"""
Black-box functions to optimize.
"""

from .classics import *
from .custom import *
from .priors import *

from . import classics
from . import custom
from . import priors

__all__ = []
__all__ = classics.__all__
__all__ = custom.__all__
__all__ = priors.__all__
