# -*- coding: utf-8 -*-
"""
Defines the typeset class and a number of builtin type sets.
"""
from __future__ import print_function, division, absolute_import

import collections
from functools import reduce
from itertools import starmap

from itertools import izip

from numba import typesystem
from numba.typesystem import types

__all__ = [ 'typeset', 'numeric', 'integral', 'floating', 'complextypes' ]

#----------------------------------------------------------------------------
# Signature matching
#----------------------------------------------------------------------------

def _match_argtype(type1, type2):
    return (type1.is_typeset and type2 in type1.types) or type1 == type2

def _build_position_table(signature):
    table = collections.defaultdict(list)
    for i, argtype in enumerate(signature.args):
        if argtype.is_typeset:
            table[argtype].append(i)

    return table

def get_effective_argtypes(promote, signature, argtypes):
    """
    Get promoted argtypes for typeset arguments, e.g.

        signature = floating(floating, floating)
        argtypes = [float, double]

    =>

        [double, double]
    """
    args = list(argtypes)
    position_table = _build_position_table(signature)
    promotion_table = {}

    for poslist in position_table.values():
        if len(poslist) > 1:
            # Find all argument types corresponding to a type set
            types = [args[i] for i in poslist]

            # Promote corresponding argument types
            result_type = reduce(promote, types)

            # Update promotion table
            type_set = signature.args[poslist[-1]]
            promotion_table[type_set] = result_type

            # Build coherent argument type list
            for i in poslist:
                args[i] = result_type

    return promotion_table, args

def match(promote, signature, argtypes):
    """
    See whether a specialization matches the given function signature.
    """
    if len(signature.args) == len(argtypes):
        promotion_table, args = get_effective_argtypes(
            promote, signature, argtypes)

        if all(starmap(_match_argtype, izip(signature.args, args))):
            restype = signature.return_type
            restype = promotion_table.get(restype, restype)
            return restype(*args)

    return None


#----------------------------------------------------------------------------
# Type sets
#----------------------------------------------------------------------------

class typeset(types.NumbaType):
    """
    Holds a set of types that can be used to specify signatures for
    type inference.
    """

    typename = "typeset"
    argnames = ["types", "name"]
    defaults = {"name": None}
    flags = ["object"]

    def __init__(self, types, name):
        super(typeset, self).__init__(frozenset(types), name)
        self.first_type = types[0]

        self._from_argtypes = {}
        for type in types:
            if type.is_function:
                self._from_argtypes[type.args] = type

    def find_match(self, promote, argtypes):
        argtypes = tuple(argtypes)
        if argtypes in self._from_argtypes:
            return self._from_argtypes[argtypes]

        for type in self.types:
            signature = match(promote, type, argtypes)
            if signature:
                return signature

        return None

    def __iter__(self):
        return iter(self.types)

    def __repr__(self):
        return "typeset(%s, ...)" % (self.first_type,)

    def __hash__(self):
        return hash(id(self))


numeric = typeset(typesystem.numeric)
integral = typeset(typesystem.integral)
floating = typeset(typesystem.floating)
complextypes = typeset(typesystem.complextypes)
