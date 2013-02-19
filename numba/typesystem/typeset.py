"""
Defines the typeset class and a number of builtin type sets.
"""

import collections
from functools import reduce
from itertools import izip, starmap

from numba.typesystem import basetypes
from numba.minivect import minitypes

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

def get_effective_argtypes(context, signature, argtypes):
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
            result_type = reduce(context.promote_types, types)

            # Update promotion table
            type_set = signature.args[poslist[-1]]
            promotion_table[type_set] = result_type

            # Build coherent argument type list
            for i in poslist:
                args[i] = result_type

    return promotion_table, args

def match(context, signature, argtypes):
    """
    See whether a specialization matches the given function signature.
    """
    if len(signature.args) == len(argtypes):
        promotion_table, args = get_effective_argtypes(
                            context, signature, argtypes)

        if all(starmap(_match_argtype, izip(signature.args, args))):
            restype = signature.return_type
            restype = promotion_table.get(restype, restype)
            return restype(*args)

    return None


#----------------------------------------------------------------------------
# Type sets
#----------------------------------------------------------------------------

class typeset(minitypes.Type):
    """
    Holds a set of types that can be used to specify signatures for
    type inference.
    """

    is_typeset = True

    def __init__(self, types, name=None):
        super(typeset, self).__init__()

        self.types = frozenset(types)
        self.name = name
        self.first_type = types[0]

        self._from_argtypes = {}
        for type in types:
            if type.is_function:
                self._from_argtypes[type.args] = type

    def find_match(self, context, argtypes):
        argtypes = tuple(argtypes)
        if argtypes in self._from_argtypes:
            return self._from_argtypes[argtypes]

        for type in self.types:
            signature = match(context, type, argtypes)
            if signature:
                return signature

        return None

    def __repr__(self):
        return "typeset(%s, ...)" % (self.first_type,)

    def __hash__(self):
        return hash(id(self))


numeric = typeset(minitypes.numeric)
integral = typeset(minitypes.integral)
floating = typeset(minitypes.floating)
complextypes = typeset(minitypes.complextypes)
