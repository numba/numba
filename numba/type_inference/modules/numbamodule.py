# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np

import numba
from numba.minivect import minitypes
from numba import typesystem
from numba.type_inference.module_type_inference import register, register_inferer


@register(numba)
def typeof(expr_type):
    from numba import nodes

    type = typesystem.CastType(expr_type)
    return nodes.const(expr_type, type)
