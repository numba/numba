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
import struct
import weakref
from functools import partial

from numba.traits import traits, Delegate

native_pointer_size = struct.calcsize('@P')

if struct.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian


class TypeSystem(object):

    def __init__(self, universe, promoter=None,
                 constant_typer=None, converters=None):
        self.universe = universe

        # Find the least general type that subsumes both given types
        # t1 -> t2 -> t3
        self.promote = promoter

        # Assign types to Python constants (arbitrary python values)
        self.constant_typer = constant_typer

        # Convert between type domains
        self.converters = converters or {}

    def typeof(self, value):
        assert self.constant_typer, self
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

    polytypes = {
        # KIND -> TypeConstructor
    }

    def __init__(self, kind_sorting=None, itemsizes=None):
        self.kind_sorting = kind_sorting    # KIND -> rank
        self.itemsizes = itemsizes          # KIND -> itemsize (bytes)

        self.monotypes = {}                 # { type_name -> type }
        self.make_monotypes(self.monotypes)

        for name, type in self.monotypes.iteritems():
            setattr(self, name, type)

        self.make_polyctors()

        # Determine total type ordering
        # self.total_type_order = {}
        # sorted(monotypes.values(), key=lambda t: self.kind_sorting[self.kind(t)])

    def make_monotypes(self, ts, monotypes):
        pass

    def make_polyctors(self):
        # Create polytype constructors as methods, e.g. 'pointer(basetype)'
        # Each constructor caches live types
        for kind in self.polytypes:
            ctor = self.get_polyconstructor(kind)
            conser = Conser(ctor)
            setattr(self, kind, conser.get)

    def construct_monotypes(self, monotypes):
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
        "Determine the size of the type in bytes"
        if type.is_mono:
            return self.itemsizes[type]
        elif self.kind(type) in self.itemsizes:
            pass
        else:
            raise NotImplementedError(type)

    def get_polyconstructor(self, kind):
        type_constructor = self.polytypes[kind]
        return type_constructor

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

class TypeConverter(object):
    """
    Map types between type universes.
    """

    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

        self.polytypes = weakref.WeakKeyDictionary()

    def convert(self, type):
        "Return an LLVM type for the given type."
        if type.is_mono:
            return getattr(self.codomain, type.name)
        else:
            return self.convert_polytype(type)

    def convert_polytype(self, type):
        if type in self.polytypes:
            return self.polytypes[type]

        # Deconstruct type from domain
        params = type.params

        # Get codomain constructor
        constructor = getattr(self.codomain, type.kind)

        # Map parameter into codomain
        c = self.convert_polytype
        coparams = [c(t) if isinstance(t, Type) else t for t in params]

        # Construct type in codomain
        result = constructor(*coparams)

        self.polytypes[type] = result
        return result

#------------------------------------------------------------------------
# Type Classes
#------------------------------------------------------------------------

class Type(object):
    """
    Base of all types.
    """

    def __init__(self, kind, params, is_mono=False):
        self.kind = kind    # Type kind

        # don't call this 'args' since we already use that in FunctionType
        self.params = params
        self.is_mono = is_mono

    # __________________________________________________________________
    # Type instantiation

    @classmethod
    def mono(cls, kind, name, ty=None):
        """
        Nullary type constructor creating the most elementary of types.
        Does not compose any other type (in this domain).
        """
        return Type(kind, (name, ty), is_mono=True)

    @classmethod
    def poly(cls, kind, *args):
        """
        A type that composes other types.
        """
        return Type(kind, args)

    # __________________________________________________________________

    @property
    def name(self):
        assert self.is_mono
        return self.params[0]

    @property
    def ty(self):
        assert self.is_mono
        return self.params[1]

    def __repr__(self):
        if self.is_mono:
            return self.name
        else:
            return "%s(%s)" % (self.kind(self),
                               ", ".join(map(str, self.params)))

    # Hash by identity
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            return self.kind == attr[3:]
        raise AttributeError(attr)

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
        self._entries = weakref.WeakKeyDictionary()
        self.constructor = constructor

    def get(self, args):
        result = self._entries.get(args)
        if result is None:
            result = self.constructor(*args)
            self._entries[args] = result

        return result
