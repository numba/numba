# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function, division, absolute_import

import struct as struct_
import weakref

from numba.traits import traits, Delegate

native_pointer_size = struct_.calcsize('@P')

if struct_.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian


@traits
class TypeSystem(object):

    def __init__(self, universe, atom_unifier, constant_typer, llvm_typer):
        self.universe = universe

        # Find the least general type that subsumes both given types
        # t1 -> t2 -> t3
        self.unify_atoms = atom_unifier

        # Assign types to Python constants (arbitrary python values)
        self.constant_typer = constant_typer

        # Map types from the universe to a low-level LLVM representation
        self.llvm_typer = llvm_typer

    def typeof(self, value):
        return self.constant_typer.typeof(self, value)

    from_python = typeof # TODO: Remove

    def llvm_type(self, type):
        return self.llvm_typer.llvm_type(self, type)

    def __getattr__(self, attr):
        return getattr(self.universe, attr)

#------------------------------------------------------------------------
# Type Universe
#------------------------------------------------------------------------

class Universe(object):

    def init(self, ts):
        "Initialize to the given typesystem"

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

    def function(self, restype, argtypes):
        "Construct a function type"

    def array(self, dtype, ndim,
              is_c_contig=False,
              is_f_contig=False,
              inner_contig=False):
        "Construct an array type"

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
            return ""

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
# LLVM Typing
#------------------------------------------------------------------------

class LLVMTyper(object):

    def init(self, ts):
        "Initialize to the given typesystem"

    def llvm_type(self, ts, type):
        "Return an LLVM type for the given type."
        raise NotImplementedError

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

    def to_llvm(self, context):
        return self.ts.llvm_typer.llvm_type(self)

    # Hash by identity
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            return self.kind == attr[3:]
        raise AttributeError(attr)

class MonoType(Type):
    """
    The most elementary of types. Does not compose any other type.
    """

    __slots__ = Type.__slots__ +  ("name",)

    def __init__(self, ts, kind, name):
        super(MonoType, self).__init__(ts, kind)
        self.name = name    # Unique name

class ProductType(Type):
    """
    A type that composes other types.
    """

    __slots__ = Type.__slots__ +  ("args",)

    def __init__(self, ts, kind, args):
        super(ProductType, self).__init__(ts, kind)
        self.args = args

#------------------------------------------------------------------------
# Type Memoization
#------------------------------------------------------------------------

class Conser(object):
    """
    Type conser: constructs new types only when not already available.
    Types are weakreffed to sanitize memory consumption.

    This allows types to be compared by and hashed on identity.
    """

    __slots__ = ("constructor", "_entries")

    def __init__(self, constructor):
        self._entries = weakref.WeakKeyDictionary()
        self.constructor = constructor

    def get(self, args):
        result = self._entries.get(args)
        if result is None:
            result = self.constructor(*args)
            self._entries[args] = result

        return result

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
