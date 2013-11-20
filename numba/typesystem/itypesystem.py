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

        - unit types should map immediately between domains of the same level
        - parametric types should naturally re-construct in domains of the same level

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
import keyword
from functools import partial

import numba
from numba.utils import is_builtin

reserved = set(['bool', 'int', 'long', 'float', 'complex',
                'string', 'struct', 'array']).__contains__

def tyname(name):
    return name + "_" if reserved(name) else name

__all__ = [
    "TypeSystem", "Type", "ConstantTyper", "Conser", "TypeConser",
    "get_conser", "consing",
]

native_pointer_size = struct_.calcsize('@P')

if struct_.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian

if numba.PY3:
    map = lambda f, xs, map=map: list(map(f, xs))

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

    def __repr__(self):
        return "TypeSystem(%s, %s, %s, %s)" % (self.universe.domain_name,
                                               self.promote, self.typeof,
                                               self.converters)

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
            for cls in self.handler_table:
                if isinstance(value, cls):
                    return self.handler_table[cls](self.universe, value)

            return None

#------------------------------------------------------------------------
# Type Conversion between type domains
#------------------------------------------------------------------------

def get_ctor(codomain, kind):
    name = tyname(kind)
    if not hasattr(codomain, name):
        raise AttributeError(
            "Codomain '%s' has no attribute '%s'" % (codomain, name))
    return getattr(codomain, name)

def convert_unit(domain, codomain, type):
    return get_ctor(codomain, type.typename)

def convert_para(domain, codomain, type, coparams):
    return get_ctor(codomain, type.kind)(*coparams) # Construct type in codomain

# ______________________________________________________________________

class TypeConverter(object):
    """
    Map types between type universes.
    """

    def __init__(self, domain, codomain,
                 convert_unit=convert_unit, convert_para=convert_para):
        self.domain, self.codomain = domain, codomain
        self.convert_unit = partial(convert_unit, domain, codomain)
        self.convert_para = partial(convert_para, domain, codomain)
        self.partypes = weakref.WeakKeyDictionary()

    def convert(self, type):
        "Return an LLVM type for the given type."
        if isinstance(type, (tuple, list)):
            return tuple(map(self.convert, type))
        elif not isinstance(type, Type):
            return type
        elif type.is_unit:
            return self.convert_unit(type)
        else:
            return self.convert_parametrized(type)

    def convert_parametrized(self, type):
        # if type in self.partypes: # TODO: Check for type mutability
        #     return self.partypes[type]

        # Construct parametrized type in codomain
        result = self.convert_para(type, map(self.convert, type.params))

        # self.partypes[type] = result
        return result

    def __repr__(self):
        return "TypeConverter(%s -> %s)" % (self.domain.domain_name,
                                            self.codomain.domain_name)

#------------------------------------------------------------------------
# Type Classes
#------------------------------------------------------------------------

def add_flags(obj, flags):
    for flag in flags:
        setattr(obj, "is_" + flag, True)

class Type(object):
    """
    Base of all types.
    """
    metadata = None

    def __init__(self, kind, *params, **kwds):
        self.kind = kind    # Type kind

        # don't call this 'args' since we already use that in function
        self.params = list(params)
        self.is_unit = kwds.get("is_unit", False)
        if self.is_unit:
            self.typename = params[0]
        else:
            self.typename = kind

        # Immutable metadata
        self.metadata = kwds.get("metadata", frozenset())
        self._metadata = self.metadata and dict(self.metadata)

    @classmethod
    def unit(cls, kind, name, flags=(), **kwds):
        """
        Nullary type constructor creating the most elementary of types.
        Does not compose any other type.
        """
        type = cls(kind, name, is_unit=True,
                   metadata=frozenset(kwds.items()))
        add_flags(type, flags)
        type.flags = flags
        return type

    @classmethod
    def default_args(cls, args, kwargs):
        "Add defaults to a given args tuple for type construction"
        return args

    def __repr__(self):
        if self.is_unit:
            return self.params[0].rstrip("_")
        else:
            return "%s(%s)" % (self.kind, ", ".join(map(str, self.params)))

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            return self.kind == attr[3:]
        elif self.metadata and attr in self._metadata:
            return self._metadata[attr]
        raise AttributeError( attr)

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
        try:
            result = self._entries.get(args)
            if result is None:
                result = self.constructor(*args)
                self._entries[args] = result
        except:
            result = self.constructor(*args)

        return result

class TypeConser(object):

    def __init__(self, type):
        assert isinstance(type, type), type
        assert issubclass(type, Type), type.__mro__
        self.type = type
        self.conser = Conser(type)

    def get(self, *args, **kwargs):
        # Add defaults to the arguments to ensure consing correctness
        args = self.type.default_args(args, kwargs)
        return self.conser.get(*args)

def get_conser(ctor):
    if isinstance(ctor, type) and issubclass(ctor, Type):
        return TypeConser(ctor) # Use a conser that tracks defaults
    else:
        return Conser(ctor)

def consing(ctor):
    return get_conser(ctor).get
