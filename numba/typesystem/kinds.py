# -*- coding: utf-8 -*-

"""
Kinds for numba types.
"""

from __future__ import print_function, division, absolute_import

#------------------------------------------------------------------------
# Type Kinds
#------------------------------------------------------------------------

# Low level kinds

KIND_VOID       = "void"
KIND_INT        = "int"
KIND_FLOAT      = "float"
KIND_COMPLEX    = "complex"
KIND_FUNCTION   = "function"
KIND_ARRAY      = "array"
KIND_POINTER    = "pointer"
KIND_NULL       = "null"
KIND_CARRAY     = "carray"
KIND_STRUCT     = "struct"

# High-level Numba kinds

KIND_BOOL       = "bool"
KIND_OBJECT     = "object"
KIND_EXTTYPE    = "exttype"
KIND_NONE       = "none"
