"""
Added for symmetry with the core API
"""

from functools import partial
from numba.core.extending import intrinsic as _intrinsic
from numba.core.extending import overload as _overload

intrinsic = _intrinsic(target='cuda')
overload = partial(_overload, target='cuda')
