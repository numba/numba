"""
Added for symmetry with the core API
"""

from functools import partial
from numba.core.extending import intrinsic as _intrinsic
from numba.core.extending import overload as _overload
from numba.core.extending import overload_method as _overload_method
from numba.core.extending import overload_attribute as _overload_attribute

intrinsic = _intrinsic(target='cuda')
overload = partial(_overload, target='cuda')
overload_method = partial(_overload_method, target='cuda')
overload_attribute = partial(_overload_attribute, target='cuda')
