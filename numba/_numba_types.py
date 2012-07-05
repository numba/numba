import math
import types
import ctypes

import numpy as np
from numpy import ctypeslib
# from numpy.ctypeslib import _typecodes

from numba import llvm_types, _ext
from numba.minivect.minitypes import *
from numba.minivect import miniast, minitypes

__all__ = minitypes.__all__ + [
    'O', 'b', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8',
    'f4', 'f8', 'f16', 'c8', 'c16', 'c32',
]

class NumbaType(minitypes.Type):
    is_numba_type = True

class TupleType(NumbaType, minitypes.ObjectType):
    is_tuple = True
    name = "tuple"
    size = 0

class ListType(NumbaType, minitypes.ObjectType):
    is_list = True
    name = "list"
    size = 0

class IteratorType(NumbaType, minitypes.ObjectType):
    is_iterator = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(IteratorType, self).__init__(**kwds)
        self.base_type = base_type

    def __repr__(self):
        return "iterator<%s>" % (self.base_type,)

class PHIType(NumbaType):
    """
    Type for phi() values.
    """
    is_phi = True

class ModuleType(NumbaType, minitypes.ObjectType):
    """
    Represents a type for modules.

    Attributes:
        is_numpy_module: whether the module is the numpy module
        module: in case of numpy, the numpy module or a submodule
    """
    is_module = True
    is_numpy_module = False
    module = None

    def __repr__(self):
        if self.is_numpy_module:
            return 'numpy'
        else:
            return 'ModuleType'

class NumpyAttributeType(NumbaType, minitypes.ObjectType):
    """
    Type for attributes of a numpy (sub)module.

    Attributes:
        module: the numpy (sub)module
        attr: the attribute name (str)
    """
    is_numpy_attribute = True
    module = None
    attr = None

    def __repr__(self):
        return "%s.%s" % (self.module.__name__, self.attr)

class NumpyDtypeType(NumbaType, minitypes.ObjectType):
    is_numpy_dtype = True
    dtype = None

    def resolve(self):
        return _map_dtype(self.dtype)

class EllipsisType(NumbaType, minitypes.ObjectType):
    is_ellipsis = True

    def __eq__(self, other):
        return other.is_ellipsis

    def __repr__(self):
        return "..."

class SliceType(NumbaType, minitypes.ObjectType):
    is_slice = True

    def __eq__(self, other):
        return other.is_slice

    def __repr__(self):
        return ":"

class NewAxisType(NumbaType, minitypes.ObjectType):
    is_newaxis = True

    def __eq__(self, other):
        return other.is_newaxis

    def __repr__(self):
        return "newaxis"

class GlobalType(NumbaType):
    is_global = True

class BuiltinType(NumbaType):
    is_builtin = True

class ModuleAttributeType(NumbaType):
    is_module_attr = True

class RangeType(NumbaType):
    is_range = True

tuple_ = TupleType()
phi = PHIType()
module_type = ModuleType()

# Convert llvm Type object to kind-bits string
def llvmtype_to_strtype(type):
    """
    Map LLVM type back to our type system.
    """
    if typ.kind == lc.TYPE_FLOAT:
        return 'f32'
    elif typ.kind == lc.TYPE_DOUBLE:
        return 'f64'
    elif typ.kind == lc.TYPE_INTEGER:
        return 'i%d' % typ.width
    elif typ.kind == lc.TYPE_POINTER and\
         typ.pointee.kind == lc.TYPE_FUNCTION:
        return ['func'] + typ.pointee.args
    elif typ.kind == lc.TYPE_POINTER and\
         typ.pointee.kind == lc.TYPE_STRUCT:
        return ['arr[]']


#
### Type shorthands
#

O = object_
b = bool_
i1 = int8
i2 = int16
i4 = int32
i8 = int64
u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64

f4 = float_
f8 = double
f16 = float128

c8 = complex64
c16 = complex128
c32 = complex256

class NumbaTypeMapper(minitypes.TypeMapper):
    def to_llvm(self, type):
        if type.is_array:
            return _numpy_array
        elif type.is_complex:
            return lc.Type.struct([type.base_type, type.base_type])
        elif type.is_py_ssize_t:
            return llvm_types._llvm_py_ssize_t
        elif type.is_object:
            return llvm_types._pyobject_head_struct_p

        return super(NumbaTypeMapper, self).to_llvm(type)

    def from_python(self, value):
        if isinstance(value, np.ndarray):
            dtype = _map_dtype(value.dtype)
            return minitypes.ArrayType(dtype, value.ndim,
                                       is_c_contig=value.flags['C_CONTIGUOUS'],
                                       is_f_contig=value.flags['F_CONTIGUOUS'])
        elif isinstance(value, tuple):
            return tuple_
        elif isinstance(value, types.ModuleType):
            return module_type
        else:
            return super(NumbaTypeMapper, self).from_python(value)

def _map_dtype(dtype):
    """
    >>> _map_dtype(np.dtype(np.int32))
    int32
    >>> _map_dtype(np.dtype(np.int64))
    int64
    >>> _map_dtype(np.dtype(np.object))
    PyObject *
    >>> _map_dtype(np.dtype(np.float64))
    double
    >>> _map_dtype(np.dtype(np.complex128))
    complex128
    """
    item_idx = int(math.log(dtype.itemsize, 2))
    if dtype.kind == 'i':
        return [i1, i2, i4, i8][item_idx]
    elif dtype.kind == 'u':
        return [u1, u2, u4, u8][item_idx]
    elif dtype.kind == 'f':
        if dtype.itemsize == 2:
            pass # half floats not supported yet
        elif dtype.itemsize == 4:
            return f4
        elif dtype.itemsize == 8:
            return f8
        elif dtype.itemsize == 16:
            return f16
    elif dtype.kind == 'b':
        return i1
    elif dtype.kind == 'c':
        if dtype.itemsize == 8:
            return c8
        elif dtype.itemsize == 16:
            return c16
        elif dtype.itemsize == 32:
            return c32
    elif dtype.kind == 'O':
        return O

    raise NotImplementedError("dtype %s not supported" % (dtype,))

# We don't support all types....
def pythontype_to_strtype(typ):
    if issubclass(typ, float):
        return 'f64'
    elif issubclass(typ, int):
        return 'i%d' % _plat_bits
    elif issubclass(typ, (types.BuiltinFunctionType, types.FunctionType)):
        return ["func"]
    elif issubclass(typ, tuple):
        return 'tuple'

# Add complex, unsigned, and bool
def str_to_llvmtype(str):
    # STR -> LLVM
    if __debug__:
        print("str_to_llvmtype(): str = %r" % (str,))
    n_pointer = 0
    if str.endswith('*'):
        n_pointer = str.count('*')
        str = str[:-n_pointer]
    if str[0] == 'f':
        if str[1:] == '32':
            ret_val = lc.Type.float()
        elif str[1:] == '64':
            ret_val = lc.Type.double()
    elif str[0] == 'i':
        num = int(str[1:])
        ret_val = lc.Type.int(num)
    elif str.startswith('arr['):
        ret_val = _numpy_array
    else:
        raise TypeError, "Invalid Type: %s" % str
    for _ in xrange(n_pointer):
        ret_val = lc.Type.pointer(ret_val)
    return ret_val

def _handle_list_conversion(typ):
    assert isinstance(typ, list)
    crnt_elem = typ[0]
    dimcount = 1
    while isinstance(crnt_elem, list):
        crnt_elem = crnt_elem[0]
        dimcount += 1
    return dimcount, crnt_elem

def _dtypeish_to_str(typ):
    n_pointer = 0
    if typ.endswith('*'):
        n_pointer = typ.count('*')
        typ = typ[:-n_pointer]
    dt = np.dtype(typ)
    return ("%s%s%s" % (dt.kind, 8*dt.itemsize, "*" * n_pointer))

def convert_to_llvmtype(typ):
    n_pointer = 0
    if isinstance(typ, list):
        return _numpy_array
    return str_to_llvmtype(_dtypeish_to_str(typ))

def convert_to_ctypes(type):
    # FIXME: At some point we should add a type check to the
    # wrapper code s.t. it ensures the given argument conforms to
    # the following:
    #     np.ctypeslib.ndpointer(dtype = np.dtype(crnt_elem),
    #                            ndim = dimcount,
    #                            flags = 'C_CONTIGUOUS')
    # For now, we'll just allow any Python objects, and hope for the best.

    if type.is_pointer:
        return ctypes.POINTER(convert_to_ctypes(type.base_type))
    elif type.is_object or type.is_array:
        return ctypes.py_object
    elif type.is_float:
        if type.itemsize == 4:
            return ctypes.c_float
        elif type.itemsize == 8:
            return ctypes.c_double
        else:
            return ctypes.c_longdouble
    elif type.is_int:
        item_idx = int(math.log(type.itemsize))
        if type.is_signed:
            values = [ctypes.c_int8, ctypes.c_int16, ctypes.c_int32,
                      ctypes.c_int64]
        else:
            values = [ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32,
                      ctypes.c_uint64]
        return values[item_idx]
    elif type.is_complex:
        if type.itemsize == 8:
            return Complex64
        elif type.itemsize == 16:
            return Complex128
        else:
            return Complex256
    elif type.is_c_string:
        return ctypes.c_char_p
    elif type.is_function:
        return_type = convert_to_ctypes(type.return_type)
        arg_types = [covert_to_ctypes(arg_type) for arg_type in type.args]
        return ctypes.CFUNCTYPE(return_type, arg_types)
    elif type.is_py_ssize_t:
        return getattr(ctypes, 'c_uint%d' % (_ext.sizeof_py_ssize_t() * 8))
    else:
        raise NotImplementedError(type)

# NOTE: The following ctypes structures were inspired by Joseph
# Heller's response to python-list question about ctypes complex
# support.  In that response, he said these were only suitable for
# Linux.  Might our milage vary?

class ComplexMixin (object):
    def _get (self):
        # FIXME: Ensure there will not be a loss of precision here!
        return self._numpy_ty_(self.real + (self.imag * 1j))

    def _set (self, value):
        self.real = value.real
        self.imag = value.imag

    value = property(_get, _set)

    @classmethod
    def from_param(cls, param):
        ret_val = cls()
        ret_val.value = param
        return ret_val

    @classmethod
    def make_ctypes_prototype_wrapper(cls, ctypes_prototype):
        '''This is a hack so that functions that return a complex type
        will construct a new Python value from the result, making the
        Numba compiled function a drop-in replacement for a Python
        function.'''
        # FIXME: See if there is some way of avoiding this additional
        # wrapper layer.
        def _make_complex_result_wrapper(in_func):
            ctypes_function = ctypes_prototype(in_func)
            def _complex_result_wrapper(*args, **kws):
                # Return the value property, not the ComplexMixin
                # instance built by ctypes.
                return ctypes_function(*args, **kws).value
            return _complex_result_wrapper
        return _make_complex_result_wrapper

class Complex64 (ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_float), ('imag', ctypes.c_float)]
    _numpy_ty_ = complex64

class Complex128 (ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]
    _numpy_ty_ = complex128

# TODO: What if sizeof(long double) != 16?
class Complex256 (ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_longdouble), ('imag', ctypes.c_longdouble)]
    _numpy_ty_ = complex128

def convert_to_strtype(typ):
    # LLVM -> STR
    # FIXME: This current mapping preserves dtype information, but
    # loses ndims.
    arr = 0
    if isinstance(typ, list):
        arr = 1
        _, typ = _handle_list_conversion(typ)
    return '%s%s%s' % ('arr[' * arr, _dtypeish_to_str(typ), ']' * arr)

# Add complex, unsigned, and bool
def typcmp(type1, type2):
    if type1==type2:
        return 0
    kind1 = type1[0]
    kind2 = type2[0]
    if kind1 == kind2:
        return cmp(int(type1[1:]),int(type2[1:]))
    if kind1 == 'f':
        return 1
    else:
        return -1

def typ_isa_number(typ):
    ret_val = ((not typ.endswith('*')) and
               (typ.startswith(('i', 'f'))))
    return ret_val

# Both inputs are Variable objects
#  Resolves types on one of them.
#  Won't work if both need resolving
# Currently delegates casting to Variable.llvm(), but only in the
# presence of a builder instance.
def resolve_type(arg1, arg2, builder = None):
    if __debug__:
        print("resolve_type(): arg1 = %r, arg2 = %r" % (arg1, arg2))
    typ = None
    typ1 = None
    typ2 = None
    if arg1._llvm is not None:
        typ1 = arg1.typ
    else:
        try:
            str_to_llvmtype(arg1.typ)
            typ1 = arg1.typ
        except TypeError:
            pass
    if arg2._llvm is not None:
        typ2 = arg2.typ
    else:
        try:
            str_to_llvmtype(arg2.typ)
            typ2 = arg2.typ
        except TypeError:
            pass
    if typ1 is None and typ2 is None:
        raise TypeError, "Both types not valid"
        # The following is trying to enforce C-like upcasting rules where
    # we try to do the following:
    # * Use higher precision if the types are identical in kind.
    # * Use floats instead of integers.
    if typ1 is None:
        typ = typ2
    elif typ2 is None:
        typ = typ1
    else:
        # FIXME: What about arithmetic operations on arrays?  (Though
        # we'd need to support code generation for them first...)
        if typ_isa_number(typ1) and typ_isa_number(typ2):
            if typ1[0] == typ2[0]:
                if int(typ1[1:]) > int(typ2[1:]):
                    typ = typ1
                else:
                    typ = typ2
            elif typ1[0] == 'f':
                typ = typ1
            elif typ2[0] == 'f':
                typ = typ2
        else:
            # Fall-through case: just use the left hand operand's type...
            typ = typ1
    if __debug__:
        print("resolve_type() ==> %r" % (typ,))
    return (typ,
            arg1.llvm(typ, builder = builder),
            arg2.llvm(typ, builder = builder))

# This won't convert any llvm types.  It assumes
#  the llvm types in args are either fixed or not-yet specified.
def func_resolve_type(mod, func, args):
    # already an llvm function
    if func.val and func.val is func._llvm:
        typs = [llvmtype_to_strtype(x) for x in func._llvm.type.pointee.args]
        lfunc = func._llvm
    else:
        # we need to generate the function including the types
        typs = [arg.typ if arg._llvm is not None else '' for arg in args]
        # pick first one as choice
        choicetype = None
        for typ in typs:
            if typ is not None:
                choicetype = typ
                break
        if choicetype is None:
            raise TypeError, "All types are unspecified"
        typs = [choicetype if x is None else x for x in typs]
        lfunc = map_to_function(func.val, typs, mod)

    llvm_args = [arg.llvm(typ) for typ, arg in zip(typs, args)]
    return lfunc, llvm_args

if __name__ == '__main__':
    import doctest
    doctest.testmod()