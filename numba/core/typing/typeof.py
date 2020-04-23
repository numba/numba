from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import typing as py_typing

import numpy as np

from numba.core import types, utils, errors
from numba.np import numpy_support

# terminal color markup
_termcolor = errors.termcolor()


class Purpose(enum.Enum):
    # Value being typed is used as an argument
    argument = 1
    # Value being typed is used as a constant
    constant = 2


_TypeofContext = namedtuple("_TypeofContext", ("purpose",))


def typeof(val, purpose=Purpose.argument):
    """
    Get the Numba type of a Python value for the given purpose.
    """
    # Note the behaviour for Purpose.argument must match _typeof.c.
    c = _TypeofContext(purpose)
    ty = typeof_impl(val, c)
    if ty is None:
        msg = _termcolor.errmsg(
            f"Cannot determine Numba type of {type(val)}")
        raise ValueError(msg)
    return ty


@singledispatch
def typeof_impl(val, c):
    """
    Generic typeof() implementation.
    """
    tp = _typeof_buffer(val, c)
    if tp is not None:
        return tp

    # cffi is handled here as it does not expose a public base class
    # for exported functions or CompiledFFI instances.
    from numba.core.typing import cffi_utils
    if cffi_utils.SUPPORTED:
        if cffi_utils.is_cffi_func(val):
            return cffi_utils.make_function_type(val)
        if cffi_utils.is_ffi_instance(val):
            return types.ffi

    return getattr(val, "_numba_type_", None)


def _typeof_buffer(val, c):
    from numba.core.typing import bufproto
    try:
        m = memoryview(val)
    except TypeError:
        return
    # Object has the buffer protocol
    try:
        dtype = bufproto.decode_pep3118_format(m.format, m.itemsize)
    except ValueError:
        return
    type_class = bufproto.get_type_class(type(val))
    layout = bufproto.infer_layout(m)
    return type_class(dtype, m.ndim, layout=layout,
                      readonly=m.readonly)


@typeof_impl.register(ctypes._CFuncPtr)
def _typeof_ctypes_function(val, c):
    from .ctypes_utils import is_ctypes_funcptr, make_function_type
    if is_ctypes_funcptr(val):
        return make_function_type(val)


@typeof_impl.register(type)
def _typeof_type(val, c):
    """
    Type various specific Python types.
    """
    if issubclass(val, BaseException):
        return types.ExceptionClass(val)
    if issubclass(val, tuple) and hasattr(val, "_asdict"):
        return types.NamedTupleClass(val)

    if issubclass(val, np.generic):
        return types.NumberClass(numpy_support.from_dtype(val))

    from numba.typed import Dict
    if issubclass(val, Dict):
        return types.TypeRef(types.DictType)

    from numba.typed import List
    if issubclass(val, List):
        return types.TypeRef(types.ListType)

    if c.purpose == Purpose.argument:
        # We only return for argument contexts.
        # For situations like x = int(y), we want typeof(int) to return None,
        # so that Context.resolve_value_type calls Context._get_global_type.

        if val in (int, float, complex):
            return typeof_impl(val(0), c)

        if val is str:
            return typeof_impl(str("numba"), c)


# type(py_typing.List) is different in Python 3.6 vs. 3.7+.
@typeof_impl.register(type(py_typing.List))
def _typeof_typing(val, c):
    # The type hierarchy of python typing library changes in 3.7.
    if utils.PYVERSION < (3, 7):
        list_check = lambda x: issubclass(x, py_typing.List)
        dict_check = lambda x: issubclass(x, py_typing.Dict)
        set_check = lambda x: issubclass(x, py_typing.Set)
        tuple_check = lambda x: issubclass(x, py_typing.Tuple)
        union_check = lambda x: x.__origin__ == py_typing.Union
    else:
        list_check = lambda x: x.__origin__ is list
        dict_check = lambda x: x.__origin__ is dict
        set_check = lambda x: x.__origin__ is set
        tuple_check = lambda x: x.__origin__ is tuple
        union_check = lambda x: x.__origin__ is py_typing.Union

    if union_check(val):
        (arg_1_py, arg_2_py) = val.__args__
        if arg_2_py is not type(None): # noqa: E721
            raise ValueError(
                "Cannot type Union that is not an Optional "
                f"(second type {arg_2_py} is not NoneType")
        arg_1_nb = typeof_impl(arg_1_py, c)
        if arg_1_nb is None:
            raise ValueError(f"Cannot type optional inner type {arg_1_py}")
        return types.Optional(arg_1_nb)

    if list_check(val):
        (element_py,) = val.__args__
        element_nb = typeof_impl(element_py, c)
        if element_nb is None:
            raise ValueError(
                f"Cannot type list element type {element_py}")
        return types.List(element_nb)

    if dict_check(val):
        key_py, value_py = val.__args__
        key_nb = typeof_impl(key_py, c)
        value_nb = typeof_impl(value_py, c)
        if key_nb is None:
            raise ValueError(f"Cannot type dict key type {key_py}")
        if value_nb is None:
            raise ValueError(f"Cannot type dict value type {value_py}")
        return types.DictType(key_nb, value_nb)

    if set_check(val):
        (element_py,) = val.__args__
        element_nb = typeof_impl(element_py, c)
        if element_nb is None:
            raise ValueError(
                f"Cannot type set element type {element_py}")
        return types.Set(element_nb)

    if tuple_check(val):
        tys = tuple(typeof_impl(elem, c) for elem in val.__args__)
        if any(ty is None for ty in tys):
            return
        return types.BaseTuple.from_types(tys)


# Before Python 3.7, there is not a common shared metaclass for typing.
if utils.PYVERSION < (3, 7):
    typeof_impl.register(type(py_typing.Union), _typeof_typing)


@typeof_impl.register(bool)
def _typeof_bool(val, c):
    return types.boolean


@typeof_impl.register(float)
def _typeof_float(val, c):
    return types.float64


@typeof_impl.register(complex)
def _typeof_complex(val, c):
    return types.complex128


def _typeof_int(val, c):
    # As in _typeof.c
    nbits = utils.bit_length(val)
    if nbits < 32:
        typ = types.intp
    elif nbits < 64:
        typ = types.int64
    elif nbits == 64 and val >= 0:
        typ = types.uint64
    else:
        raise ValueError("Int value is too large: %s" % val)
    return typ


for cls in utils.INT_TYPES:
    typeof_impl.register(cls, _typeof_int)


@typeof_impl.register(np.generic)
def _typeof_numpy_scalar(val, c):
    try:
        return numpy_support.map_arrayscalar_type(val)
    except NotImplementedError:
        pass


@typeof_impl.register(str)
def _typeof_str(val, c):
    return types.string


@typeof_impl.register(type((lambda a: a).__code__))
def _typeof_code(val, c):
    return types.code_type


@typeof_impl.register(type(None))
def _typeof_none(val, c):
    return types.none


@typeof_impl.register(type(Ellipsis))
def _typeof_ellipsis(val, c):
    return types.ellipsis


@typeof_impl.register(tuple)
def _typeof_tuple(val, c):
    tys = [typeof_impl(v, c) for v in val]
    if any(ty is None for ty in tys):
        return
    return types.BaseTuple.from_types(tys, type(val))


@typeof_impl.register(list)
def _typeof_list(val, c):
    if len(val) == 0:
        raise ValueError("Cannot type empty list")
    ty = typeof_impl(val[0], c)
    if ty is None:
        raise ValueError(
            f"Cannot type list element type {type(val[0])}")
    return types.List(ty, reflected=True)


@typeof_impl.register(set)
def _typeof_set(val, c):
    if len(val) == 0:
        raise ValueError("Cannot type empty set")
    item = next(iter(val))
    ty = typeof_impl(item, c)
    if ty is None:
        raise ValueError(
            f"Cannot type set element type {type(item)}")
    return types.Set(ty, reflected=True)


@typeof_impl.register(slice)
def _typeof_slice(val, c):
    return types.slice2_type if val.step in (None, 1) else types.slice3_type


@typeof_impl.register(enum.Enum)
@typeof_impl.register(enum.IntEnum)
def _typeof_enum(val, c):
    clsty = typeof_impl(type(val), c)
    return clsty.member_type


@typeof_impl.register(enum.EnumMeta)
def _typeof_enum_class(val, c):
    cls = val
    members = list(cls.__members__.values())
    if len(members) == 0:
        raise ValueError("Cannot type enum with no members")
    dtypes = {typeof_impl(mem.value, c) for mem in members}
    if len(dtypes) > 1:
        raise ValueError("Cannot type heterogeneous enum: "
                         "got value types %s"
                         % ", ".join(sorted(str(ty) for ty in dtypes)))
    if issubclass(val, enum.IntEnum):
        typecls = types.IntEnumClass
    else:
        typecls = types.EnumClass
    return typecls(cls, dtypes.pop())


@typeof_impl.register(np.dtype)
def _typeof_dtype(val, c):
    tp = numpy_support.from_dtype(val)
    return types.DType(tp)


@typeof_impl.register(np.ndarray)
def _typeof_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return types.Array(dtype, val.ndim, layout, readonly=readonly)


@typeof_impl.register(types.NumberClass)
def _typeof_number_class(val, c):
    return val


@typeof_impl.register(types.Literal)
def _typeof_literal(val, c):
    return val


@typeof_impl.register(types.TypeRef)
def _typeof_typeref(val, c):
    return val


@typeof_impl.register(types.Type)
def _typeof_nb_type(val, c):
    if isinstance(val, types.BaseFunction):
        return val
    elif isinstance(val, (types.Number, types.Boolean)):
        return types.NumberClass(val)
    else:
        return types.TypeRef(val)
