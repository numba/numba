import __builtin__
import math
import copy
import types

import llvm.core
import numpy as np
# from numpy.ctypeslib import _typecodes

import numba
from numba import llvm_types, extension_types
from numba.minivect.minitypes import *
from numba.minivect.minitypes import map_dtype
from numba.minivect import minitypes
from numba.minivect.ctypes_conversion import (convert_from_ctypes,
                                              convert_to_ctypes)

__all__ = minitypes.__all__ + [
    'O', 'b1', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8',
    'f4', 'f8', 'f16', 'c8', 'c16', 'c32' 
]

def is_obj(type):
    return type.is_object or type.is_array

def promote_closest(context, int_type, candidates):
    """
    promote_closest(Py_ssize_t, [int_, long_, longlong]) -> longlong
    """
    for candidate in candidates:
        promoted = context.promote_types(int_type, candidate)
        if promoted == candidate:
            return candidate

    return candidates[-1]

# Patch repr of objects to print "object_" instead of "PyObject *"
minitypes.ObjectType.__repr__ = lambda self: "object_"

class NumbaType(minitypes.Type):
    is_numba_type = True

class TupleType(NumbaType, minitypes.ObjectType):
    is_tuple = True
    name = "tuple"
    size = 0

    def __str__(self):
        return "tuple(%s)" % ", ".join(["..."] * self.size)

class ListType(NumbaType, minitypes.ObjectType):
    is_list = True
    name = "list"
    size = 0

    def __str__(self):
        return "list(%s)" % ", ".join(["..."] * self.size)

class DictType(NumbaType, minitypes.ObjectType):
    is_dict = True
    name = "dict"
    size = 0

    def __str__(self):
        return "dict(%s)" % ", ".join(["..."] * self.size)

class IteratorType(NumbaType, minitypes.ObjectType):
    is_iterator = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(IteratorType, self).__init__(**kwds)
        self.base_type = base_type

    def __repr__(self):
        return "iterator<%s>" % (self.base_type,)

class UninitializedType(NumbaType):
    is_uninitialized = True

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

    def __init__(self, module, **kwds):
        super(ModuleType, self).__init__(**kwds)
        self.module = module
        self.is_numpy_module = module is np
        self.is_numba_module = module is numba

    def __repr__(self):
        if self.is_numpy_module:
            return 'numpy'
        else:
            return 'ModuleType'

class ModuleAttributeType(NumbaType, minitypes.ObjectType):
    is_module_attribute = True

    module = None
    attr = None

    def __repr__(self):
        return "%s.%s" % (self.module.__name__, self.attr)

    @property
    def value(self):
        return getattr(self.module, self.attr)

class NumpyAttributeType(ModuleAttributeType):
    """
    Type for attributes of a numpy (sub)module.

    Attributes:
        module: the numpy (sub)module
        attr: the attribute name (str)
    """

    is_numpy_attribute = True

class MethodType(NumbaType, minitypes.ObjectType):
    """
    Method of something.

        base_type: the object type the attribute was accessed on
    """

    is_method = True

    def __init__(self, base_type, attr_name, **kwds):
        super(MethodType, self).__init__(**kwds)
        self.base_type = base_type
        self.attr_name = attr_name

class ExtMethodType(NumbaType, minitypes.FunctionType):
    """
    Extension method type used for vtab purposes.

    is_class: is classmethod?
    is_static: is staticmethod?
    """

    def __init__(self, return_type, args, name=None,
                 is_class=False, is_static=False, **kwds):
        super(ExtMethodType, self).__init__(return_type, args, name, **kwds)
        self.is_class = is_class
        self.is_static = is_static

class NumpyDtypeType(NumbaType, minitypes.ObjectType):
    is_numpy_dtype = True
    dtype = None

    def resolve(self):
        return map_dtype(self.dtype)

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

class GlobalType(NumbaType, minitypes.ObjectType):
    is_global = True

class BuiltinType(NumbaType, minitypes.ObjectType):
    is_builtin = True

    def __init__(self, name, **kwds):
        super(BuiltinType, self).__init__(**kwds)
        self.name = name
        self.func = getattr(__builtin__, name)

class RangeType(NumbaType, minitypes.ObjectType):
    is_range = True

class NoneType(NumbaType, minitypes.ObjectType):
    is_none = True

    def __str__(self):
        return "None Type"

class CTypesFunctionType(NumbaType, minitypes.ObjectType):
    is_ctypes_function = True

    def __init__(self, ctypes_func, restype, argtypes, **kwds):
        super(CTypesFunctionType, self).__init__(**kwds)
        self.ctypes_func = ctypes_func
        self.signature = minitypes.FunctionType(return_type=restype,
                                                args=argtypes)

    def __repr__(self):
        return "<ctypes function %s>" % (self.signature,)

class SizedPointerType(NumbaType, minitypes.PointerType):
    size = None
    is_sized_pointer = True

class CastType(NumbaType, minitypes.ObjectType):

    is_cast = True

    def __init__(self, dst_type, **kwds):
        super(CastType, self).__init__(**kwds)
        self.dst_type = dst_type

    def __repr__(self):
        return "<cast(%s)>" % self.dst_type


class ExtensionType(NumbaType, minitypes.ObjectType):

    is_extension = True
    is_final = False

    def __init__(self, py_class, **kwds):
        super(ExtensionType, self).__init__(**kwds)
        assert isinstance(py_class, type), "Must be a new-style class"
        self.name = py_class.__name__
        self.py_class = py_class
        self.symtab = {}  # attr_name -> attr_type
        self.methods = [] # (method_name, func_signature)
        self.methoddict = {} # method_name -> (func_signature, vtab_index)

        self.vtab_offset = extension_types.compute_vtab_offset(py_class)
        self.attr_offset = extension_types.compute_attrs_offset(py_class)
        self.attribute_struct = None
        self.vtab_type = None

        self.parent_attr_struct = None
        self.parent_vtab_type = None

    def add_method(self, method_name, method_signature):
        if method_name in self.methoddict:
            # Patch current signature after type inference
            signature = self.get_signature(method_name)
            assert method_signature.args == signature.args
            if signature.return_type is None:
                signature.return_type = method_signature.return_type
            else:
                assert signature.return_type == method_signature.return_type, \
                                                            method_signature
        else:
            self.methoddict[method_name] = (method_signature, len(self.methods))
            self.methods.append((method_name, method_signature))

    def get_signature(self, method_name):
        signature, vtab_offset = self.methoddict[method_name]
        return signature

    def set_attributes(self, attribute_list):
        """
        Create the symbol table and attribute struct from a list of
        (varname, attribute_type)
        """
        import numba.symtab

        self.attribute_struct = numba.struct(attribute_list)
        self.symtab.update([(name, numba.symtab.Variable(type))
                               for name, type in attribute_list])

    def __repr__(self):
        return "<Extension %s>" % self.name


class ClosureType(NumbaType, minitypes.ObjectType):
    """
    Type of closures and inner functions.
    """

    is_closure = True

    def __init__(self, signature, **kwds):
        super(ClosureType, self).__init__(**kwds)
        self.signature = signature
        self.closure = None

    def __repr__(self):
        return "<closure(%s)>" % self.signature

class ClosureScopeType(ExtensionType):
    """
    Type of the enclosing scope for closures. This is always passed in as
    first argument to the function.
    """

    is_closure_scope = True
    is_final = True

    def __init__(self, py_class, parent_scope, **kwds):
        super(ClosureScopeType, self).__init__(py_class, **kwds)
        self.parent_scope = parent_scope
        self.unmangled_symtab = None

        if self.parent_scope is None:
            self.scope_prefix = ""
        else:
            self.scope_prefix = self.parent_scope.scope_prefix + "0"

#
### Types participating in type graph cycles
#
class UnresolvedType(NumbaType):

    is_unresolved = True
    rank = 1

    def __init__(self, variable, **kwds):
        super(UnresolvedType, self).__init__(**kwds)
        self.variable = variable

    def simplify(self):
        return self.resolve() is self

    def resolve(self):
        return self.variable.type

class PromotionType(UnresolvedType):

    is_promotion = True

    def __init__(self, variable, context, types, **kwds):
        super(PromotionType, self).__init__(variable, **kwds)
        self.context = context
        self.types = types

    def simplify(self, seen=None):
        """
        Simplify a promotion type tree:

            promote(int_, float_)
                -> float_

            promote(deferred(x), promote(float_, double), int_, promote(<self>))
                -> promote(deferred(x), double)

            promote(deferred(x), deferred(y))
                -> promote(deferred(x), deferred(y))
        """
        if seen is None:
            seen = set()

        # Find all types in the type graph and eliminate nested promotion types
        types = set([self])
        seen.add(self)
        for type in self.types:
            if type in seen:
                continue

            if type.is_unresolved:
                type = type.resolve()

            if type.is_promotion:
                type.simplify(seen)
                types.update(type.types)
            elif type.is_unresolved:
                # Get the resolved type or the type itself from the deferred
                # type
                types.add(type.variable.type)
            else:
                types.add(type)

        types.remove(self)
        resolved_types = [type for type in types if not type.is_unresolved]
        unresolved_types = [type for type in types if type.is_deferred]

        self.variable.type = self
        if not resolved_types:
            # Everything is deferred
            return False
        else:
            # Simplify as much as possible
            result_type = resolved_types[0]
            for type in resolved_types[1:]:
                result_type = self.context.promote_types(result_type, type)

            if len(resolved_types) == len(types):
                self.variable.type = result_type
            else:
                self.types = set([result_type] + unresolved_types)

            return True

    @classmethod
    def promote(cls, *types):
        var = Variable(None)
        type = PromotionType(var, types)
        type.resolve()
        return type.variable.type

    def __repr__(self):
        return "promote(%s)" % ", ".join(str(type) for type in self.types)

class DeferredType(UnresolvedType):
    """
    We don't know what the type is at the point we need a type, so we create
    a deferred type.

        Depends on: self.variable.type

    Example:

        def func():
            for i in range(10):
                # type(x) = phi(undef, deferred(x_1)) = phi(deferred(x_1))
                if i > 1:
                    print x   # type is deferred(x_1)
                x = ...       # resolve deferred(x_1) to type(...)
    """

    is_deferred = True

    def __repr__(self):
        return "<deferred(%s)>" % (self.variable.unmangled_name,)

    def to_llvm(self, context):
        assert self.resolved_type, self
        return self.resolved_type.to_llvm(context)

class UnanalyzableType(UnresolvedType):
    """
    A type that indicates the statement cannot be analyzed without
    """

    is_unanalyzable = True

tuple_ = TupleType()
phi = PHIType()
none = NoneType()
uninitialized = UninitializedType()

intp = minitypes.npy_intp

#
### Type shorthands
#

O = object_
b1 = bool_
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


    def __init__(self, context):
        super(NumbaTypeMapper, self).__init__(context)
        # self.ctypes_func_type = type(ctypes.CFUNCTYPE(ctypes.c_int))
        # libc = ctypes.CDLL(ctypes.util.find_library('c'))
        # self.ctypes_func_type2 = type(libc.printf)

    def to_llvm(self, type):
        if type.is_array:
            return llvm_types._numpy_array
        elif type.is_complex:
            lbase_type = type.base_type.to_llvm(self.context)
            return llvm.core.Type.struct([lbase_type, lbase_type])
        elif type.is_py_ssize_t:
            return llvm_types._llvm_py_ssize_t
        elif type.is_object:
            return llvm_types._pyobject_head_struct_p

        return super(NumbaTypeMapper, self).to_llvm(type)

    def from_python(self, value):
        if isinstance(value, np.ndarray):
            dtype = map_dtype(value.dtype)
            return minitypes.ArrayType(dtype, value.ndim,
                                       is_c_contig=value.flags['C_CONTIGUOUS'],
                                       is_f_contig=value.flags['F_CONTIGUOUS'])
        elif isinstance(value, tuple):
            return tuple_
        elif isinstance(value, types.ModuleType):
            return ModuleType(value)
        # elif isinstance(value, (self.ctypes_func_type, self.ctypes_func_type2)):
        elif hasattr(value, 'errcheck'):
            # ugh, ctypes
            if value.argtypes is None:
                return object_

            restype = convert_from_ctypes(value.restype)
            argtypes = map(convert_from_ctypes, value.argtypes)
            return CTypesFunctionType(value, restype, argtypes)
        elif isinstance(value, minitypes.Type):
            return CastType(dst_type=value)
        elif hasattr(type(value), '__numba_ext_type'):
            return type(value).__numba_ext_type
        else:
            return super(NumbaTypeMapper, self).from_python(value)

    def promote_types(self, type1, type2):
        if (type1.is_array or type2.is_array) and not \
            (type1.is_array and type2.is_array):
            if type1.is_array:
                array_type = type1
                other_type = type2
            else:
                array_type = type2
                other_type = type1

            type = copy.copy(array_type)
            type.dtype = self.promote_types(array_type.dtype, other_type)
            return type
        elif type1.is_deferred or type2.is_deferred:
            return [type1, type2][type2.is_deferred]

        return super(NumbaTypeMapper, self).promote_types(type1, type2)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
