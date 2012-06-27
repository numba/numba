import math

import numpy as np

from numba.minivect.minitypes import *
from numba.minivect import miniast, minitypes

__all__ = minitypes.__all__ + [
    'O', 'b', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8',
    'f4', 'f8', 'f16', 'c8', 'c16', 'c32',
]

minitypes.Type.is_numba_type = False
class NumbaType(minitypes.Type):
    is_numba_type = True
    is_iterator = False

class TupleType(NumbaType, minitypes.ObjectType):
    name = "tuple"

tuple_ = TupleType()

class IteratorType(NumbaType, minitypes.ObjectType):
    is_iterator = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(IteratorType, self).__init__(**kwds)
        self.base_type = base_type

    def __repr__(self):
        return "iterator<%s>" % (self.base_type,)

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
            raise NotImplementedError("Complex types not implemented yet")

        return super(NumbaTypeMapper, self).to_llvm(type)

    def from_python(self, value):
        if isinstance(value, np.ndarray):
            dtype = _map_dtype(value.dtype)
            return minitypes.ArrayType(dtype, value.ndim,
                                       is_c_contig=value.flags['C_CONTIGUOUS'],
                                       is_f_contig=value.flags['F_CONTIGUOUS'])
        elif isinstance(value, tuple):
            return tuple_
        else:
            return super(NumbaTypeMapper, self).from_python(value)

def _map_dtype(dtype):
    """
    >>> _map_dtype(np.dtype(np.int32))
    int16
    >>> _map_dtype(np.dtype(np.object))
    PyObject *
    >>> _map_dtype(np.dtype(np.float64))
    double
    >>> _map_dtype(np.dtype(np.complex128))
    complex128
    """
    item_idx = int(math.log(dtype.itemsize))
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

def convert_to_ctypes(typ):
    # STR -> CTYPES
    import ctypes
    from numpy.ctypeslib import _typecodes
    if isinstance(typ, list):
        # FIXME: At some point we should add a type check to the
        # wrapper code s.t. it ensures the given argument conforms to
        # the following:
        #     np.ctypeslib.ndpointer(dtype = np.dtype(crnt_elem),
        #                            ndim = dimcount,
        #                            flags = 'C_CONTIGUOUS')
        # For now, we'll just allow any Python objects, and hope for the best.
        return ctypes.py_object
    n_pointer = 0
    if typ.endswith('*'):
        n_pointer = typ.count('*')
        typ = typ[:-n_pointer]
        if __debug__:
            print("convert_to_ctypes(): n_pointer = %d, typ' = %r" %
                  (n_pointer, typ))
    ret_val = _typecodes[np.dtype(typ).str]
    for _ in xrange(n_pointer):
        ret_val = ctypes.POINTER(ret_val)
    return ret_val

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