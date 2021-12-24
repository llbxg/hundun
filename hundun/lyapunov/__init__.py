from ._les_differential import (calc_les_differential,
                                calc_les_differential_w_qr,
                                calc_max_le_differential,
                                calc_les_differential_w_orth)
from ._les_difference import calc_les_difference

from ..systems._systems import DynamicalSystems as _DynamicalSystems
from ..systems._differential import Differential as _Differential
from ..systems._difference import Difference as _Difference


import numpy as _np

def calc_les(system, **options):

    if issubclass(system, _Differential):
        return calc_les_differential(system, **options)

    elif issubclass(system, _Difference):
        return calc_les_difference(system, **options)

    if not issubclass(system, _DynamicalSystems):
        raise TypeError(f"{system} must be a DynamicalSystems")

    return _np.zeros((3, 3)), (None, None, None)
