# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function, division, absolute_import

import ctypes
import struct
import weakref
from functools import partial

from numba.traits import traits, Delegate

native_pointer_size = struct.calcsize('@P')

if struct.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian


@traits
class TypeSystem(object):

    def __init__(self, universe, atom_unifier, constant_typer, converters):
        self.universe = universe

        # Find the least general type that subsumes both given types
        # t1 -> t2 -> t3
        self.unify_atoms = atom_unifier

        # Assign types to Python constants (arbitrary python values)
        self.constant_typer = constant_typer

        # Convert between type domains
        self.converters = converters

        # --- initialize
        self.universe.init(self)
        self.constant_typer.init(self)
        for converter in converters.itervalues():
            converter.init(self)

    def typeof(self, value):
        return self.constant_typer.typeof(self, value)

    from_python = typeof # TODO: Remove

    def convert(self, codomain_name, type):
        converter = self.converters[codomain_name]
        return converter.convert(type)

    def __getattr__(self, attr):
        return getattr(self.universe, attr)

#------------------------------------------------------------------------
# Type Universe
#------------------------------------------------------------------------

class Universe(object):

    kind_sorting = {
        # KIND -> rank
    }

    polytypes = {
        # TYPE_KIND -> TypeConstructor
    }

    def init(self, ts):
        "Initialize to the given typesystem"
        # { type_name -> type }
        monotypes = {}
        self.construct_monotypes(monotypes)

        for name, type in monotypes.iteritems():
            setattr(self, name, type)

        # Determine total type ordering
        # self.total_type_order = {}
        # sorted(monotypes.values(), key=lambda t: self.kind_sorting[self.kind(t)])

    def construct_monotypes(self):
        raise NotImplementedError

    def kind(self, type):
        """
        Return the kind of the type. Default available kinds are:

            int, float, complex, function, array, pointer, carray,
            struct, object
        """
        return type.kind

    def byteorder(self, type):
        return nbo

    def rank(self, type):
        "Determine the rank of the type (for sorting)"

    def itemsize(self, type):
        "Determine the size of the tye in bytes"

    def function(self, restype, argtypes, name=None, is_vararg=False):
        "Construct a function type"

    def array(self, dtype, ndim,
              is_c_contig=False,
              is_f_contig=False,
              inner_contig=False):
        "Construct an array type"
        return self.polytypes[KIND_ARRAY](self, )

    def pointer(self, basetype):
        "Construct a pointer type of the given base type"

    def carray(self, base_type, size): # TODO: Remove this type
        "Construct a 1D C array type"

    def struct(self, fields):
        "Construct a *mutable* struct type (to allow recursive types)"

    def strtype(self, type):
        "Return a type string representation"
        if isinstance(type, MonoType):
            return type.name
        else:
            return "%s(%s)" % (type.kind, ", ".join(map(str, type.params)))

#------------------------------------------------------------------------
# Typing of Constants
#------------------------------------------------------------------------

class ConstantTyper(object):

    def init(self, ts):
        "Initialize to the given typesystem"

    def typeof(self, ts, value):
        "Get a concrete type given a typesystem and a python value"
        raise NotImplementedError

#------------------------------------------------------------------------
# Type Conversion between type domains
#------------------------------------------------------------------------

class TypeConverter(object):
    """
    Map types between type universes.
    """

    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

    def convert(self, type):
        "Return an LLVM type for the given type."
        assert type.ts is self.domain

        if isinstance(type, MonoType):
            # cotypes = self.codomain.monotypes[type.kind]
            # return cotypes[type.name]
            return getattr(self.codomain, type.name)
        else:
            # Deconstruct type from domain
            params = type.params
            # Construct type in codomain
            constructor = getattr(self.codomain, type.kind)
            return constructor(*params)

#------------------------------------------------------------------------
# Type Classes
#------------------------------------------------------------------------

class Type(object):
    """
    Base type. Specialized to the typesystem it belongs to.
    """

    __slots__ = ("__weakref__", "ts", "kind")

    def __init__(self, ts, kind):
        self.ts = ts    # TypeSystem
        self.kind = kind

    def __repr__(self):
        return self.ts.universe.strtype(self)

    # Hash by identity
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            return self.kind == attr[3:]
        raise AttributeError(attr)

class MonoType(Type):
    """
    Nullary type constructor creating the most elementary of types.
    Does not compose any other type.
    """

    __slots__ = Type.__slots__ +  ("name",)

    def __init__(self, ts, kind, name):
        super(MonoType, self).__init__(ts, kind)
        self.name = name    # Unique name

class PolyType(Type):
    """
    A type that composes other types.
    """

    __slots__ = Type.__slots__ +  ("args",)

    def __init__(self, ts, kind, args):
        super(PolyType, self).__init__(ts, kind)
        # don't call this 'args' since we already use that in FunctionType
        self.params = args

#------------------------------------------------------------------------
# Type Memoization
#------------------------------------------------------------------------

class Conser(object):

    __slots__ = ("type_constructor", "_entries")

    def __init__(self, constructor):
        self._entries = weakref.WeakKeyDictionary()
        self.constructor = constructor

    def get(self, args):
        result = self._entries.get(args)
        if result is None:
            result = self.constructor(*args)
            self._entries[args] = result

        return result

class TypeConser(Conser):
    """
    Type conser: constructs new types only when not already available.
    Types are weakreffed to sanitize memory consumption.

    This allows types to be compared by and hashed on identity.
    """

    def __init__(self, ts, kind, type_constructor):
        super(TypeConser, self).__init__(partial(type_constructor, ts, kind))

#------------------------------------------------------------------------
# Type Kinds
#------------------------------------------------------------------------

KIND_VOID       = "void"
KIND_INT        = "int"
KIND_FLOAT      = "float"
KIND_COMPLEX    = "complex"
KIND_FUNCTION   = "function"
KIND_ARRAY      = "array"
KIND_POINTER    = "pointer"
KIND_CARRAY     = "carray"
KIND_STRUCT     = "struct"
KIND_OBJECT     = "object"

#------------------------------------------------------------------------
# Default type sizes
#------------------------------------------------------------------------

# Type sizes in bytes
default_type_sizes = {
    # Int
    "char":         1,
    "int8":         1,
    "int16":        2,
    "int32":        4,
    "int64":        8,
    # Unsigned int
    "uchar":        1,
    "uint8":        1,
    "uint16":       2,
    "uint32":       4,
    "uint64":       8,
    # Float
    "float16":      1,
    "float32":      2,
    "float64":      4,
    "float":        4,
    "double":       8,
    # Complex
    "complex64":    8,
    "complex128":   16,
    "complex256":   32,
}

native_sizes = {
    # Int
    "short":        struct.calsize("h"),
    "int":          struct.calsize("i"),
    "long":         struct.calsize("l"),
    "longlong":     struct.calsize("Q"),
    # Unsigned int
    "short":        struct.calsize("H"),
    "int":          struct.calsize("I"),
    "long":         struct.calsize("L"),
    "longlong":     struct.calsize("Q"),
    # Float
    "longdouble":   ctypes.sizeof(ctypes.c_longdouble),
    # Pointer
    "pointer":      ctypes.sizeof(ctypes.c_void_p),
}

#------------------------------------------------------------------------
# Default type sizes
#------------------------------------------------------------------------

class DefaultUniverse(Universe):
    pass