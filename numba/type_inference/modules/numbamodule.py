import numpy as np

import numba
from numba.minivect import minitypes
from numba import typesystem
from numba.type_inference.module_type_inference import register, register_inferer


@register(numba)
def typeof(context, expr):
    from numba import nodes

    obj = expr.variable.type
    type = typesystem.CastType(obj)
    return nodes.const(obj, type)
