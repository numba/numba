# -*- coding: utf-8 -*-

"""
User-facing numba types.
"""

from __future__ import print_function, division, absolute_import

from functools import partial
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from numba.typesystem.typesystem import Type, Conser, TypeConser

#------------------------------------------------------------------------
# Type metaclass
#------------------------------------------------------------------------

class Registry(object):
    def __init__(self):
        self.registry = {}

    def register(self, name, obj):
        assert name not in self.registry, name
        self.registry[name] = obj

    def items(self):
        return self.registry.items()

numba_type_registry = Registry()

class Accessor(object):
    def __init__(self, idx, mutable):
        self.idx = idx
        self.mutable = mutable

    def __get__(self, obj, type=None):
        return obj.params[self.idx]

    def __set__(self, obj, value):
        if not self.mutable:
            raise AttributeError("Cannot set attribute '%s' of type '%s'" %
                                            (obj.attrs[self.idx], type(obj)))
        obj.params[self.idx] = value

class TypeMetaClass(type):
    "Metaclass for numba types, conses immutable types."

    registry = numba_type_registry

    def __new__(cls, name, bases, dict):
        if not any(getattr(b, "mutable", 0) for b in bases):
            base_attrs = getattr(bases[0], "attrs", ())
            dict["__slots__"] = Type.slots + tuple(dict.get("atts", base_attrs))
        return super(TypeMetaClass, cls).__new__(cls, name, bases, dict)

    def __init__(self, name, bases, dict):
        if self.typename is not None:
            self.registry.register(self.typename, self)

        # Create accessors
        for i, arg in enumerate(dict.get("args", ())):
            setattr(self, arg, Accessor(i, self.mutable))

        # Process flags
        for flag in self.flags:
            setattr(self, "is_" + flag, True)

        if not self.mutable:
            self.conser = Conser(partial(type.__call__, self))

    def __call__(self, *args, **kwds):
        args = self.default_args(args, kwds)
        if not self.mutable:
            return self.conser.get(*args)
        return super(TypeMetaClass, self).__call__(*args, **kwds)

#------------------------------------------------------------------------
# Type Decorators
#------------------------------------------------------------------------

def consing(cls):
    """
    Decorator that conses calls when the class is called. A potential problem
    is that we need default_args() to get the arguments for the consing, and
    subsequently for user constructors (or we need to decorate all types, even
    when not consing...)
    """

#------------------------------------------------------------------------
# Type Implementations
#------------------------------------------------------------------------

class UserType(Type):
    """
    MonoType with user-facing methods:

        call: create a function type
        slice: create an array type
        conversion: to_llvm/to_ctypes/get_dtype
    """

    __slots__ = Type.slots

    def __getitem__(self, item):
        """
        Support array type creation by slicing, e.g. double[:, :] specifies
        a 2D strided array of doubles. The syntax is the same as for
        Cython memoryviews.
        """
        assert isinstance(item, (tuple, slice))

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

            return ArrayType(self.dtype, len(item),
                             is_c_contig=step_idx == len(item) - 1,
                             is_f_contig=step_idx == 0)
        else:
            verify_slice(item)
            return ArrayType(self, 1, is_c_contig=bool(item.step))

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

        return FunctionType(self, args)

    def pointer(self):
        return PointerType(self)

    def qualify(self, *qualifiers):
        return self # TODO: implement

    def to_llvm(self, context):
        raise NotImplementedError("use typesystem.llvm(type) instead")
        return self.typesystem.convert("llvm", self)


class NumbaType(UserType):

    __metaclass__ = TypeMetaClass
    __slots__ = UserType.__slots__

    typename = None
    args = []
    flags = []
    defaults = {}
    mutable = True # Whether to cons type instances

    def __init__(self, *params, **kwds):
        super(NumbaType, self).__init__(self.typename, params, **kwds)

    @classmethod
    def default_args(cls, args, kwargs):
        names = cls.args

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
                raise TypeError(
                    "Constructor '%s' requires %d arguments (got %d)" % (
                                        cls.typename, len(names), len(args)))

        return tuple(args)

mono, poly = UserType.mono, UserType.poly

#------------------------------------------------------------------------
# Low-level polytypes
#------------------------------------------------------------------------

def pass_by_ref(type):
    return type.is_struct or type.is_complex

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

conser = object()

class FunctionType(NumbaType):
    typename = "function"
    args = ['return_type', 'args', 'name', 'is_vararg']
    defaults = {"name": None, "is_vararg": False}
    __new__ = conser

    struct_by_reference = False

    def __repr__(self):
        args = [str(arg) for arg in self.args]
        if self.is_vararg:
            args.append("...")
        if self.name:
            namestr = self.name
        else:
            namestr = ''

        return "%s (*%s)(%s)" % (self.return_type, namestr, ", ".join(args))

    # @property
    # def actual_signature(self):
    #     """
    #     Passing structs by value is not properly supported for different
    #     calling conventions in LLVM, so we take an extra argument
    #     pointing to a caller-allocated struct value.
    #     """
    #     if self.struct_by_reference:
    #         args = []
    #         for arg in self.args:
    #             if pass_by_ref(arg):
    #                 arg = arg.pointer()
    #             args.append(arg)
    #
    #         return_type = self.return_type
    #         if pass_by_ref(self.return_type):
    #             return_type = void
    #             args.append(self.return_type.pointer())
    #
    #         self = FunctionType(return_type, args)
    #
    #     return self

    @property
    def struct_return_type(self):
        # Function returns a struct.
        return self.return_type.pointer()

    def __call__(self, *args):
        if len(args) != 1 or isinstance(args[0], Type):
            return super(FunctionType, self).__call__(*args)

        assert self.return_type is not None
        assert self.args is not None
        func, = args
        return Function(self, func)

# ______________________________________________________________________
# Pointers

class PointerType(NumbaType):
    typename = "pointer"
    args = ['base_type']

    def __repr__(self):
        space = " " * (not self.base_type.is_pointer)
        return "%s%s*" % (self.base_type, space)

class SizedPointerType(NumbaType):
    """
    A pointer with knowledge of its range.

    E.g. an array's 'shape' or 'strides' attribute.
    This also allow tuple unpacking.
    """
    typename = "sized_pointer"
    args = ["base_type", "size"]
    flags = ["pointer"]

    def __eq__(self, other):
        if other.is_sized_pointer:
            return (self.base_type == other.base_type and
                    self.size == other.size)
        return other.is_pointer and self.base_type == other.base_type

    def __hash__(self):
        return hash(self.base_type.pointer())

class CArrayType(NumbaType):
    typename = "carray"
    args = ["base_type", "size"]

# ______________________________________________________________________
# Structs

class StructType(NumbaType):

    typename = "istruct"
    args = ["fields", "name", "readonly", "packed"]
    defaults = dict.fromkeys(args[1:])

    @property
    def fielddict(self):
        return dict(self.fields)

    def __repr__(self):
        if self.name:
            name = self.name + ' '
        else:
            name = ''
        return 'struct %s{ %s }' % (
                name, ", ".join("%s %s" % (field_type, field_name)
                                    for field_name, field_type in self.fields))

    def __eq__(self, other):
        return other.is_struct and self.fields == other.fields

    def __hash__(self):
        return hash(tuple(self.fields))

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

class MutableStructType(StructType):
    """
    Create a struct type. Fields may be ordered or unordered. Unordered fields
    will be ordered from big types to small types (for better alignment).
    """
    typename = "struct"

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

class ArrayType(NumbaType):
    """
    An array type. ArrayType may be sliced to obtain a subtype:

    >>> double[:, :, ::1][1:]
    double[:, ::1]
    >>> double[:, :, ::1][:-1]
    double[:, :]

    >>> double[::1, :, :][:-1]
    double[::1, :]
    >>> double[::1, :, :][1:]
    double[:, :]
    """

    typename = "array"
    args = ["dtype", "ndim", "is_c_contig", "is_f_contig", "inner_contig"]
    defaults = dict.fromkeys(args[-3:], False)
    flags = ["object"]

    def pointer(self):
        raise Exception("You probably want a pointer type to the dtype")

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

class KnownValueType(NumbaType):
    """
    Type which is associated with a known value or well-defined symbolic
    expression:

        np.add          => np.add
        np.add.reduce   => (np.add, "reduce")

    (Remember that unbound methods like np.add.reduce are transient, i.e.
     np.add.reduce is not np.add.reduce).
    """
    typename = "known_value"
    attrs = ["value"]

class ModuleType(KnownValueType):
    """
    Represents a type for modules.

    Attributes:
        is_numpy_module: whether the module is the numpy module
        module: in case of numpy, the numpy module or a submodule
    """
    typename = "module"
    flags = ["object"]

    # TODO: Get rid of these
    is_numpy_module = property(lambda self: self.module is np)
    is_numba_module = property(lambda self: self.module is np)
    is_math_module = property(lambda self: self.module is np)

    @property
    def module(self):
        return self.value

class AutojitFunctionType(NumbaType):
    "Type for autojit functions"
    typename = "autojit_func"
    args = ["autojit_func"]
    flags = ["object"]

class JitFunctionType(NumbaType):
    "Type for jit functions"
    typename = "jit_function"
    args = ["jit_func"]
    flags = ["object"]

class NumpyDtypeType(NumbaType):
    "Type of numpy dtypes"
    typename = "numpy_dtype"
    args = ["dtype"]

class ComplexType(NumbaType):
    typename = "complex"
    args = ["base_type"]

class ReferenceType(NumbaType):
    """
    A reference to an (primitive or Python) object. This is passed as a
    pointer and dereferences automatically.

    Currently only supported for structs.
    """
    # TODO: remove this type?
    typename = "reference"
    args = ["referenced_type"]

class MetaType(NumbaType):
    """
    A type instance in user code. e.g. double(value). The Name node will have
    a cast-type with dst_type 'double'.
    """
    typename = "meta"
    args = ["dst_type"]
    flags = ["object"]


class BuiltinType(KnownValueType):
    typename = "builtin"

    def __init__(self, name, **kwds):
        value = getattr(builtins, name)
        super(BuiltinType, self).__init__(value, **kwds)

        self.name = name
        self.func = self.value

#------------------------------------------------------------------------
# Container Types
#------------------------------------------------------------------------

class ContainerListType(NumbaType):
    """
    :param base_type: the element type of the tuple
    :param size: set to a value >= 0 is the size is known
    :return: a tuple type representation
    """

    args = ["base_type", "size"]
    flags = ["object"]

    def is_sized(self):
        return self.size >= 0

class TupleType(ContainerListType):
    typename = "tuple"

class ListType(ContainerListType):
    typename = "list"

class MapContainerType(NumbaType):
    args = ["key_type", "value_type", "size"]
    flags = ["object"]

class DictType(MapContainerType):
    typename = "dict"

#------------------------------------------------------------------------
# Type Ordering
#------------------------------------------------------------------------

def sort_types_key(field_type):
    if field_type.is_complex:
        return field_type.base_type.rank * 2
    elif field_type.is_numeric or field_type.is_struct:
        return field_type.rank
    elif field_type.is_vector:
        return sort_types_key(field_type.element_type) * field_type.vector_size
    elif field_type.is_carray:
        return sort_types_key(field_type.base_type) * field_type.size
    elif field_type.is_pointer or field_type.is_object or field_type.is_array:
        return 8
    else:
        return 1

def sort_key(keyvalue):
    field_name, field_type = keyvalue
    return sort_types_key(field_type)

def sort_types(types_dict):
    # reverse sort on rank, forward sort on name
    d = {}
    for field in types_dict.iteritems():
        key = sort_key(field)
        d.setdefault(key, []).append(field)

    def key(keyvalue):
        field_name, field_type = keyvalue
        return field_name

    fields = []
    for rank in sorted(d, reverse=True):
        fields.extend(sorted(d[rank], key=key))

    return fields

integral = []
native_integral = []
floating = []
complextypes = []

# integral.sort(key=types.sort_types_key)
# native_integral = [ty for ty in integral if ty.typename in universe.native_sizes]
# floating.sort(key=types.sort_types_key)
# complextypes.sort(key=types.sort_types_key)
