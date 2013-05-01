# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba.minivect.minitypes import *
from numba.minivect.minitypes import (FunctionType)

from .types import *
from .typesystem import *
from .closuretypes import *
from .ssatypes import *
from .templatetypes import *
from .containertypes import *
from .typeutils import *

from .shorthands import *

# from typeset import *
from .typematch import *

from .universe import *
from .defaults import *

# TODO: Remove
from numba.minivect.minitypes import *

__all__ = minitypes.__all__ + [
    'O', 'b1', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8',
    'f4', 'f8', 'f16', 'c8', 'c16', 'c32', 'template',
]
