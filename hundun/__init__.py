"""
% Using any of these subpackages requires an explicit import.
equations
utils
"""

from .__version__ import __version__

from .systems import *  # noqa:
from .exploration import *  # noqa:
from .exploration._utils import embedding
from .lyapunov import calc_les, calc_lyapunov_dimension
from .utils._draw import Drawing

__all__ = ['__version__', 'embedding', 'calc_les', 'calc_lyapunov_dimension',
           'Drawing']
