# -*- coding: utf-8 -*-

"""
Inferface for our typesystems.

Some requirements:

  * The typesystem must allow us to switch between low-level representations.
    For instance, we may want to represent an array as a NumPy ndarray, a
    Py_buffer, or some other representation.

  * The sizes of atom types (e.g. int) must be easily customizable. This allows
    an interpreter to switch sizes to simulate different platforms.

  * Type representations and sizes, must be overridable without
    reimplementing the type. E.g. an atom type can be sliced to create
    an array type, which should be separate from its low-level representation.

  * Types should map easily between type domains of the same level, e.g.
    between the low-level numba and llvm types.
    Ideally this forms an isomorphism, which is precluded by ctypes and
    numpy type systems::

       >>> ctypes.c_longlong
       <class 'ctypes.c_long'>

  * No type system but our own should be entrenched in any part of the
    codebase, including the code generator.

  * Sets of type constructors (type universes) must be composable.
    For instance, we may want to extend a low-level type systems of ints
    and pointers with objects to yield a type universe supporting
    both constructs.

  * Universes must be re-usable across type-systems. Types of universes
    represent abstract type concepts, and the type-systems give meaning
    and representation to values of those types.

  * Types must be immutable and consed, i.e.
    ts.pointer(base) is ts.pointer(base) must always be True

  * The use of a certain typesystem must suggest at which level the
    corresponding terms operate.

  * Conversion code should be written with minimal effort:

        - monotypes should map immediately between domains of the same level
        - polytypes should naturally re-construct in domains of the same level

  * Converting a type to a lower-level domain constitutes a one-way
    conversion. This should, where possible, constitute a lowering in the
    same domain followed by a conversion. E.g.:

        def numba_complex_to_llvm(type):
            return to_llvm(lower_complex(type))

  * Type constructors must be substitutable. E.g. an external user may
    want to construct a universe where type construction is logged, or
    where special type checking is performed, disallowing certain compositions.
"""

from __future__ import print_function, division, absolute_import

import ctypes
import struct as struct_
import weakref
from functools import partial

native_pointer_size = struct_.calcsize('@P')

if struct_.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian


class TypeSystem(object):

    def __init__(self, universe, promote=None, typeof=None, converters=None):
        self.universe = universe

        # Find the least general type that subsumes both given types
        # t1 -> t2 -> t3
        self.promote = promote

        # Assign types to Python constants (arbitrary python values)
        self.typeof = typeof
        self.from_python = typeof # TODO: Remove

        # Convert between type domains
        self.converters = converters or {}

    def convert(self, codomain_name, type):
        convert = self.converters[codomain_name]
        return convert(type)

    def __getattr__(self, attr):
        return getattr(self.universe, attr)

#------------------------------------------------------------------------
# Typing of Constants
#------------------------------------------------------------------------

class ConstantTyper(object):

    def __init__(self, universe, typetable, handler_table):
        "Initialize to the given type universe"
        self.universe = universe
        self.typetable = typetable          # type(constant) -> type
        self.handler_table = handler_table  # type(constant) -> handler

    def typeof(self, value):
        """
        Get a concrete type given a python value.
        Return None f this ConstantTyper cannot type the constant
        """
        if type(value) in self.typetable:
            return self.typetable[type(value)]
        elif type(value) in self.handler_table:
            return self.handler_table[type(value)](self.universe, value)
        else:
            return None

#------------------------------------------------------------------------
# Type Conversion between type domains
#------------------------------------------------------------------------

def convert_mono(domain, codomain, type):
    return getattr(codomain, type.typename)

def convert_poly(domain, codomain, type, coparams):
    # Get codomain constructor
    constructor = getattr(codomain, type.kind)
    # Construct type in codomain
    return constructor(*coparams)

class TypeConverter(object):
    """
    Map types between type universes.
    """

    def __init__(self, domain, codomain,
                 convert_mono=convert_mono, convert_poly=convert_poly):
        self.convert_mono = partial(convert_mono, domain, codomain)
        self.convert_poly = partial(convert_poly, domain, codomain)
        self.polytypes = weakref.WeakKeyDictionary()

    def convert(self, type):
        "Return an LLVM type for the given type."
        if isinstance(type, (tuple, list)):
            return tuple(map(self.convert, type))
        elif not isinstance(type, Type):
            return type
        elif type.is_mono:
            return self.convert_mono(type)
        else:
            return self.convert_polytype(type)

    def convert_polytype(self, type):
        if type in self.polytypes:
            return self.polytypes[type]

        # Construct polytype in codomain
        result = self.convert_poly(type, map(self.convert, type.params))

        self.polytypes[type] = result
        return result

#------------------------------------------------------------------------
# Type Classes
#------------------------------------------------------------------------

class Type(object):
    """
    Base of all types.
    """

    slots = ("kind", "params", "is_mono", "metadata", "_metadata")
    __slots__ = slots + ("__weakref__",)

    def __init__(self, kind, params, is_mono=False, metadata=frozenset()):
        self.kind = kind    # Type kind

        # don't call this 'args' since we already use that in FunctionType
        self.params = params
        self.is_mono = is_mono

        # Immutable metadata
        self.metadata = metadata
        self._metadata = metadata and dict(metadata)

    # __________________________________________________________________
    # Type instantiation

    @classmethod
    def mono(cls, kind, name, **kwds):
        """
        Nullary type constructor creating the most elementary of types.
        Does not compose any other type (in this domain).
        """
        return cls(kind, (name,), is_mono=True,
                   metadata=frozenset(kwds.iteritems()))

    @classmethod
    def poly(cls, kind, *args):
        """
        A type that composes other types.
        """
        return cls(kind, args)

    @classmethod
    def default_args(cls, args, kwargs):
        "Add defaults to a given args tuple for type construction"
        return args

    # __________________________________________________________________

    def __repr__(self):
        if self.is_mono:
            return self.params[0]
        else:
            return "%s(%s)" % (self.kind, ", ".join(map(str, self.params)))

    # Hash by identity
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            return self.kind == attr[3:]
        elif self.metadata and attr in self._metadata:
            return self._metadata[attr]
        raise AttributeError(self, attr)

#------------------------------------------------------------------------
# Type Memoization
#------------------------------------------------------------------------

class Conser(object):
    """
    Conser: constructs new objects only when not already available.
    Objects are weakreffed to sanitize memory consumption.

    This allows the objects to be compared by and hashed on identity.
    """

    __slots__ = ("constructor", "_entries")

    def __init__(self, constructor):
        self._entries = weakref.WeakValueDictionary()
        self.constructor = constructor

    def get(self, *args):
        args = tuple(tuple(arg) if isinstance(arg, list) else arg
                         for arg in args)
        # wargs = WeakrefTuple(args)
        result = self._entries.get(args)
        if result is None:
            result = self.constructor(*args)
            self._entries[args] = result

        return result

class TypeConser(object):

    def __init__(self, polytype):
        assert isinstance(polytype, type), polytype
        assert issubclass(polytype, Type), polytype.__mro__
        self.polytype = polytype
        self.conser = Conser(polytype)

    def get(self, *args, **kwargs):
        # Add defaults to the arguments to ensure consing correctness
        args = self.polytype.default_args(args, kwargs)
        return self.conser.get(*args)

def get_conser(ctor):
    if isinstance(ctor, type) and issubclass(ctor, Type):
        return TypeConser(ctor) # Use a conser that tracks defaults
    else:
        return Conser(ctor)

def consing(ctor):
    return get_conser(ctor).get