"""
Black-box functions to optimize.
"""

from .classics import *
from .custom import *

from . import classics
from . import custom

__all__ = []
__all__ = classics.__all__
__all__ = custom.__all__
