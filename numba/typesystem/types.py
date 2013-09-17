# -*- coding: utf-8 -*-

"""
User-facing numba types.
"""

from __future__ import print_function, division, absolute_import

import numba

import ctypes
from itertools import imap
from functools import partial

from numba import odict
from numba.typesystem.itypesystem import Type, Conser, add_flags, tyname

import numpy as np

#------------------------------------------------------------------------
# Type metaclass
#------------------------------------------------------------------------

numba_type_registry = odict.OrderedDict()
register = numba_type_registry.__setitem__

class Accessor(object):
    def __init__(self, idx):
        self.idx = idx

    def __get__(self, obj, type=None):
        return obj.params[self.idx]

    def __set__(self, obj, value):
        if not obj.mutable:
            raise AttributeError("Cannot set attribute '%s' of type '%s'" %
                                            (obj.argnames[self.idx], type(obj)))
        obj.params[self.idx] = value

class TypeMetaClass(type):
    "Metaclass for numba types, conses immutable types."

    def __init__(self, name, bases, dict):
        if dict.get('typename') is None and name[0].islower():
            self.typename = name.rstrip("_")
        if self.typename is not None:
            register(self.typename, self)
        _update_class(self)
        self.conser = Conser(partial(type.__call__, self))

    def __call__(self, *args, **kwds):
        args = self.default_args(args, kwds)
        if not self.mutable:
            return self.conser.get(*args)
        return type.__call__(self, *args)

#------------------------------------------------------------------------
# Type Decorators
#------------------------------------------------------------------------

def _update_class(cls):
    # Build defaults dict { argname : default_value }
    if 'defaults' not in vars(cls):
        cls.defaults = {}
        for i, argname in enumerate(cls.argnames):
            if isinstance(argname, (list, tuple)):
                name, default = argname
                cls.argnames[i] = name
                cls.defaults[name] = default

    # Create accessors
    for i, arg in enumerate(vars(cls).get("argnames", ())):
        assert not getattr(cls, arg, False), (cls, arg)
        setattr(cls, arg, Accessor(i))

    # Process flags
    flags = list(cls.flags)
    if cls.typename:
        flags.append(cls.typename.strip("_"))

    add_flags(cls, flags)

def consing(cls):
    """
    Cons calls to the constructor.
    """
    cls.mutable = False
    return cls

def notconsing(cls):
    cls.mutable = True
    return cls

#------------------------------------------------------------------------
# Type Implementations
#------------------------------------------------------------------------

class _NumbaType(Type):
    """
    MonoType with user-facing methods:

        call: create a function type
        slice: create an array type
        conversion: to_llvm/to_ctypes/get_dtype
    """

    argnames = []
    flags = []
    defaults = {}

    qualifiers = frozenset()

    # ______________________________________________________________________
    # Internal

    def add(self, attr, value):
        "Construct new type with attr=value (e.g. functype.add('args', []))"
        assert not self.mutable
        params = list(self.params)
        params[self.argnames.index(attr)] = value
        return type(self)(*params)

    @property
    def subtypes(self):
        subtypes = []
        for p in self.params:
            if isinstance(p, (Type, list, tuple)):
                subtypes.append(p)

        return subtypes

    # ______________________________________________________________________
    # User functionality

    def pointer(self):
        return pointer(self)

    def ref(self):
        return reference(self)

    def qualify(self, *qualifiers):
        return self # TODO: implement

    def unqualify(self, *qualifiers):
        return self # TODO: implement

    # TODO: Remove context argument in favour of typesystem argument
    def to_llvm(self, context=None):
        from . import defaults
        return defaults.numba_typesystem.convert("llvm", self)

    def to_ctypes(self):
        from . import defaults
        return defaults.numba_typesystem.convert("ctypes", self)

    def to_numpy(self):
        from numba.typesystem import numpy_support
        return numpy_support.to_dtype(self)

    get_dtype = to_numpy

    # ______________________________________________________________________
    # Special methods (user functionality)

    def __getitem__(self, item):
        """
        Support array type creation by slicing, e.g. double[:, :] specifies
        a 2D strided array of doubles. The syntax is the same as for
        Cython memoryviews.
        """
        assert isinstance(item, (tuple, slice)), item

        def verify_slice(s):
            if s.start or s.stop or s.step not in (None, 1):
                raise ValueError(
                    "Only a step of 1 may be provided to indicate C or "
                    "Fortran contiguity")

        if isinstance(item, tuple):
            step_idx = None
            for idx, s in enumerate(item):
                verify_slice(s)
                if s.step and (step_idx or idx not in (0, len(item) - 1)):
                    raise ValueError(
                        "Step may only be provided once, and only in the "
                        "first or last dimension.")

                if s.step == 1:
                    step_idx = idx

            return array_(self, len(item),
                             is_c_contig=step_idx == len(item) - 1,
                             is_f_contig=step_idx == 0)
        else:
            verify_slice(item)
            return array_(self, 1, is_c_contig=bool(item.step))

    def __call__(self, *args):
        """
        Return a new function type when called with type arguments.
        """
        if len(args) == 1 and not isinstance(args[0], Type):
            # Cast in Python space
            # TODO: Create proxy object
            # TODO: Fully customizable type system (do this in Numba, not
            #       minivect)
            return args[0]

        return function(self, args)

@notconsing
class NumbaType(_NumbaType):
    """
    Base for numba types.
    """

    __metaclass__ = TypeMetaClass
    # __slots__ = Type.slots

    typename = None

    def __init__(self, *args, **kwds):
        super(NumbaType, self).__init__(self.typename, *args, **kwds)
        assert len(args) == len(self.argnames), (self.typename, args)
        for name in kwds:
            assert name in self.argnames, (self.typename, kwds, self.argnames)

    @classmethod
    def default_args(cls, args, kwargs):
        names = cls.argnames
        if len(args) == len(names):
            return args

        # Insert defaults in args tuple
        args = list(args)
        for name in names[len(args):]:
            if name in kwargs:
                args.append(kwargs[name])
            elif name in cls.defaults:
                args.append(cls.defaults[name])
            else:
                raise TypeError("Constructor '%s' requires %d arguments "
                                "(got %d)" % (cls.typename, len(names), len(args)))

        return tuple(args)

#------------------------------------------------------------------------
# Low-level parametrized types
#------------------------------------------------------------------------

def pass_by_ref(type): # TODO: Get rid of this
    return type.is_struct or type.is_complex or type.is_datetime or type.is_timedelta

class Function(object):
    """
    Function types may be called with Python functions to create a Function
    object. This may be used to minivect users for their own purposes. e.g.

    @double(double, double)
    def myfunc(...):
       ...
    """

    def __init__(self, signature, py_func):
        self.signature = signature
        self.py_func = py_func

    def __call__(self, *args, **kwargs):
        """
        Implement this to pass the callable test for classmethod/staticmethod.
        E.g.

            @classmethod
            @void()
            def m(self):
                ...
        """
        raise TypeError("Not a callable function")

@consing
class function(NumbaType):
    typename = "function"
    argnames = ['return_type', 'args', ('name', None), ('is_vararg', False)]

    def add_arg(self, i, arg):
        args = list(self.args)
        args.insert(i, arg)
        return self.add('args', args)

    # ______________________________________________________________________

    @property
    def struct_by_reference(self):
        rt = self.return_type
        byref = lambda t: t.is_struct or t.is_complex or t.is_datetime or t.is_timedelta
        return rt and byref(rt) or any(imap(byref, self.args))

    @property
    def actual_signature(self):
        """
        Passing structs by value is not properly supported for different
        calling conventions in LLVM, so we take an extra argument
        pointing to a caller-allocated struct value.
        """
        from numba import typesystem as ts

        if self.struct_by_reference:
            args = []
            for arg in self.args:
                if pass_by_ref(arg):
                    arg = arg.pointer()
                args.append(arg)

            return_type = self.return_type
            if pass_by_ref(self.return_type):
                return_type = ts.void
                args.append(self.return_type.pointer())

            self = function(return_type, args)

        return self

    @property
    def struct_return_type(self):
        # Function returns a struct.
        return self.return_type.pointer()

    # ______________________________________________________________________

    def __repr__(self):
        args = [str(arg) for arg in self.args]
        if self.is_vararg:
            args.append("...")
        if self.name:
            namestr = self.name
        else:
            namestr = ''

        return "%s (*%s)(%s)" % (self.return_type, namestr, ", ".join(args))

    def __call__(self, *args):
        if len(args) != 1 or isinstance(args[0], Type):
            return super(function, self).__call__(*args)

        assert self.return_type is not None
        assert self.argnames is not None
        func, = args
        return Function(self, func)

# ______________________________________________________________________
# Pointers

@consing
class pointer(NumbaType):
    argnames = ['base_type']

    @property
    def is_string(self): # HACK
        import numba
        return self.base_type == numba.char

    def __repr__(self):
        space = " " * (not self.base_type.is_pointer)
        return "%s%s*" % (self.base_type, space)

@consing
class sized_pointer(NumbaType):
    """
    A pointer with knowledge of its range.

    E.g. an array's 'shape' or 'strides' attribute.
    This also allow tuple unpacking.
    """
    typename = "sized_pointer"
    argnames = ["base_type", "size"]
    flags = ["pointer"]

    # def __eq__(self, other):
    #     if other.is_sized_pointer:
    #         return (self.base_type == other.base_type and
    #                 self.size == other.size)
    #     return other.is_pointer and self.base_type == other.base_type
    #
    # def __hash__(self):
    #     return hash(self.base_type.pointer())

@consing
class carray(NumbaType):
    argnames = ["base_type", "size"]

# ______________________________________________________________________
# Structs

@consing
class istruct(NumbaType):

    argnames = ["fields", ("name", None), ("readonly", False), ("packed", False)]

    @property
    def subtypes(self):
        return [f[1] for f in self.fields]

    @property
    def fielddict(self):
        return dict(self.fields)

    def __repr__(self):
        if self.name:
            name = self.name + ' '
        else:
            name = ''
        return 'struct %s{ %s }' % (
                name, ", ".join(["%s %s" % (field_type, field_name)
                                    for field_name, field_type in self.fields]))

    def is_prefix(self, other_struct):
        other_fields = other_struct.fields[:len(self.fields)]
        return self.fields == other_fields

    def offsetof(self, field_name):
        """
        Compute the offset of a field. Must be used only after mutation has
        finished.
        """
        ctype = self.to_ctypes()
        return getattr(ctype, field_name).offset

@notconsing
class struct_(istruct):
    """
    Create a struct type. Fields may be ordered or unordered. Unordered fields
    will be ordered from big types to small types (for better alignment).
    """
    mutable = True

    def __eq__(self, other):
        return other.is_struct and self.fields == other.fields

    def __hash__(self):
        return hash(tuple(self.fields))

    def copy(self):
        return type(self)(self.fields, self.name, self.readonly, self.packed)

    def add_field(self, name, type):
        assert name not in self.fielddict
        self.fields.append((name, type))
        self.mutated = True

    def update_mutated(self):
        self.rank = sum([sort_key(field) for field in self.fields])
        self.mutated = False

#------------------------------------------------------------------------
# High-level types
#------------------------------------------------------------------------

@consing
class array_(NumbaType):
    """
    An array type. array_ may be sliced to obtain a subtype:

    >>> double[:, :, ::1][1:]
    double[:, ::1]
    >>> double[:, :, ::1][:-1]
    double[:, :]

    >>> double[::1, :, :][:-1]
    double[::1, :]
    >>> double[::1, :, :][1:]
    double[:, :]
    """

    argnames = ["dtype", "ndim", "is_c_contig", "is_f_contig", "inner_contig"]
    defaults = dict.fromkeys(argnames[-3:], False)
    flags = ["object"]

    def pointer(self):
        raise Exception("You probably want a pointer type to the dtype")

    @property
    def strided(self):
        return array_(self.dtype, self.ndim)

    def __repr__(self):
        axes = [":"] * self.ndim
        if self.is_c_contig and self.ndim > 0:
            axes[-1] = "::1"
        elif self.is_f_contig and self.ndim > 0:
            axes[0] = "::1"

        return "%s[%s]" % (self.dtype, ", ".join(axes))

    def __getitem__(self, index):
        "Slicing an array slices the dimensions"
        assert isinstance(index, slice)
        assert index.step is None
        assert index.start is not None or index.stop is not None

        start = 0
        stop = self.ndim
        if index.start is not None:
            start = index.start
        if index.stop is not None:
            stop = index.stop

        ndim = len(range(self.ndim)[start:stop])

        if ndim == 0:
            return self.dtype
        elif ndim > 0:
            return type(self)(self.dtype, ndim)
        else:
            raise IndexError(index, ndim)

@consing
class autojit_function(NumbaType):
    "Type for autojit functions"
    argnames = ["autojit_func"]
    flags = ["object"]

@consing
class jit_function(NumbaType):
    "Type for jit functions"
    argnames = ["jit_func"]
    flags = ["object"]

@consing
class numpy_dtype(NumbaType):
    "Type of numpy dtypes"
    argnames = ["dtype"]
    flags = ["object"]

@consing
class complex_(NumbaType):
    argnames = ["base_type"]
    flags = ["numeric"]

    @property
    def itemsize(self):
        return self.base_type.itemsize * 2

    def __repr__(self):
        return "complex%d" % (self.itemsize * 8,)

@consing
class datetime_(NumbaType):
    argnames = ["timestamp", "units", "units_char"]
    flags = ["numeric"]
    is_numpy_datetime = True

    @property
    def itemsize(self):
        return self.timestamp.itemsize + self.units.itemsize

    def __repr__(self):
        if self.units_char:
            return "datetime_" + self.units_char
        else:
            return "datetime"

@consing
class timedelta_(NumbaType):
    argnames = ["diff", "units", "units_char"]
    flags = ["numeric"]
    is_numpy_timedelta = True

    @property
    def itemsize(self):
        return self.diff.itemsize + self.units.itemsize

    def __repr__(self):
        if self.units_char:
            return "timedelta_" + self.units_char
        else:
            return "timedelta"

@consing
class meta(NumbaType):
    """
    A type instance in user code. e.g. double(value). The Name node will have
    a cast-type with dst_type 'double'.
    """
    argnames = ["dst_type"]
    flags = [
        "object",
        "cast",     # backwards compat
    ]

#------------------------------------------------------------------------
# Container Types
#------------------------------------------------------------------------

@consing
class ContainerListType(NumbaType):
    """
    :param base_type: the element type of the tuple
    :param size: set to a value >= 0 is the size is known
    :return: a tuple type representation
    """

    argnames = ["base_type", "size"]
    flags = ["object", "container"]

    def is_sized(self):
        return self.size >= 0

@consing
class tuple_(ContainerListType):
    "tuple(base_type, size)"

@consing
class list_(ContainerListType):
    "list(base_type, size)"

@consing
class MapContainerType(NumbaType):
    argnames = ["key_type", "value_type", "size"]
    flags = ["object"]

@consing
class dict_(MapContainerType):
    "dict(key, value, size)"

#------------------------------------------------------------------------
# Types to be removed
#------------------------------------------------------------------------

class numpy_attribute(NumbaType): # TODO: remove
    argnames = ["module", "attr"]
    flags = ["object", "known_value"]

    @property
    def value(self):
        return getattr(self.module, self.attr)

class module_attribute(NumbaType): # TODO: remove
    argnames = ["module", "attr"]
    flags = ["object", "known_value"]

    @property
    def value(self):
        return getattr(self.module, self.attr)

@consing
class reference(NumbaType): # TODO: remove ?
    """
    A reference to an (primitive or Python) object. This is passed as a
    pointer and dereferences automatically.

    Currently only supported for structs.
    """
    argnames = ["referenced_type"]

@consing
class method(NumbaType): # TODO: remove
    """
    Method of something.

        base_type: the object type the attribute was accessed on
    """
    argnames = ["base_type", "attr_name"]
    flags = ["object"]

class pointer_to_function(NumbaType): # TODO: remove
    """
    Pointer to a function at a known address represented by some Python
        object (e.g. a ctypes or CFFI function).
    """
    typename = "pointer_to_function"
    argnames = ["obj", "ptr", "signature"]
    flags = ["object"]

@consing
class known_value(NumbaType): # TODO: remove
    """
    Type which is associated with a known value or well-defined symbolic
    expression:

        np.add          => np.add
        np.add.reduce   => (np.add, "reduce")

    (Remember that unbound methods like np.add.reduce are transient, i.e.
     np.add.reduce is not np.add.reduce).
    """
    argnames = ["value"]

@consing
class known_pointer(pointer): # TODO: remove
    argnames = ["base_type", "address"]

@notconsing
class global_(known_value): # TODO: Remove
    "Global type"

@consing
class builtin_(known_value): # TODO: remove
    argnames = ["name", "value"]
    flags = ["object"]

    @property
    def func(self):
        return self.value

@consing
class module(known_value): # TODO: remove
    """
    Represents a type for modules.

    Attributes:
        is_numpy_module: whether the module is the numpy module
        module: in case of numpy, the numpy module or a submodule
    """
    flags = ["object"]

    # TODO: Get rid of these
    is_numpy_module = property(lambda self: self.module is np)
    is_numba_module = property(lambda self: self.module is np)
    is_math_module = property(lambda self: self.module is np)

    @property
    def module(self):
        return self.value

#------------------------------------------------------------------------
# Convenience functions...
#------------------------------------------------------------------------

unit = _NumbaType.unit
_array = array_
_struct = struct_

def from_numpy_dtype(np_dtype):
    """
    :param np_dtype: the NumPy dtype (e.g. np.dtype(np.double))
    :return: a dtype type representation
    """
    from numba.typesystem import numpy_support
    return numpy_dtype(numpy_support.map_dtype(np_dtype))

def array(dtype, ndim, is_c_contig=False, is_f_contig=False, inner_contig=False):
    """
    :param dtype: the Numba dtype type (e.g. double)
    :param ndim: the array dimensionality (int)
    :return: an array type representation
    """
    if ndim == 0:
        return dtype
    return _array(dtype, ndim, is_c_contig, is_f_contig, inner_contig)

# ______________________________________________________________________

def sort_key(t):
    n, ty = t
    return ctypes.sizeof(ty.to_ctypes())

def struct_(fields=(), name=None, readonly=False, packed=False, **kwargs):
    "Create a mutable struct type"
    if fields and kwargs:
        raise TypeError("The struct must be either ordered or unordered")
    elif kwargs:
        import ctypes
        fields = sorted(kwargs.iteritems(), key=sort_key, reverse=True)
        # fields = sort_types(kwargs)
        # fields = list(kwargs.iteritems())

    return _struct(fields, name, readonly, packed)