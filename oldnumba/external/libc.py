# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from .external import ExternalFunction
from numba import *

c_string_type = char.pointer()

class printf(ExternalFunction):
    arg_types = [void.pointer()]
    return_type = int32
    is_vararg = True

class puts(ExternalFunction):
    arg_types = [c_string_type]
    return_type = int32

class labs(ExternalFunction):
    arg_types = [long_]
    return_type = long_

class llabs(ExternalFunction):
    arg_types = [longlong]
    return_type = longlong

class atoi(ExternalFunction):
    arg_types = [c_string_type]
    return_type = int_

class atol(ExternalFunction):
    arg_types = [c_string_type]
    return_type = long_

class atoll(ExternalFunction):
    arg_types = [c_string_type]
    return_type = longlong

class atof(ExternalFunction):
    arg_types = [c_string_type]
    return_type = double

class strlen(ExternalFunction):
    arg_types = [c_string_type]
    return_type = size_t

__all__ = [k for k, v in globals().items()
           if (v != ExternalFunction
               and isinstance(v, type)
               and issubclass(v, ExternalFunction))]

