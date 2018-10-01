import math
from numba import types, utils
from numba.extending import typeof_impl

# FIXME: Python 3
@typeof_impl.register(types.fake_str)
def typeof_unicode_type(val, c):
    return types.unicode_type
