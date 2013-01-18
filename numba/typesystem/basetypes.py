import __builtin__
import math

import numpy as np
# from numpy.ctypeslib import _typecodes

import numba
from numba import  extension_types, error
from numba.minivect.minitypes import *
from numba.minivect.minitypes import map_dtype
from numba.minivect import minitypes, minierror

#------------------------------------------------------------------------
# Numba's extension of the minivect type system
#------------------------------------------------------------------------

# Patch repr of objects to print "object_" instead of "PyObject *"
minitypes.ObjectType.__repr__ = lambda self: "object_"

class NumbaType(minitypes.Type):
    is_numba_type = True

class NumbaKeyHashingType(minitypes.KeyHashingType):
    is_numba_type = True

#------------------------------------------------------------------------
# Python Types
#------------------------------------------------------------------------

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
        self.is_math_module = module is math

    def __repr__(self):
        if self.is_numpy_module:
            return 'numpy'
        else:
            return 'ModuleType'

    @property
    def comparison_type_list(self):
        return (self.module,)

class ModuleAttributeType(NumbaKeyHashingType, minitypes.ObjectType):
    is_module_attribute = True

    module = None
    attr = None

    def __init__(self, **kwds):
        super(ModuleAttributeType, self).__init__(**kwds)
        base_name, dot, rest = self.module.__name__.partition(".")
        self.is_numpy_attribute = base_name == "numpy"

    def __repr__(self):
        return "%s.%s" % (self.module.__name__, self.attr)

    @property
    def value(self):
        return getattr(self.module, self.attr)

    @property
    def key(self):
        return (self.module, self.attr)

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

class NumpyDtypeType(NumbaType, minitypes.ObjectType):
    is_numpy_dtype = True
    dtype = None # NumPy dtype type

    def resolve(self):
        return map_dtype(self.dtype)

    def __repr__(self):
        return "NumpyDtype(%s)" % self.resolve()

class ResolvedNumpyDtypeType(NumbaType, minitypes.ObjectType):
    is_numpy_dtype = True
    dtype_type = None # numba dtype type

    def resolve(self):
        return self.dtype_type

    def __repr__(self):
        return "NumpyDtype(%s)" % self.resolve()

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
    name = None

    def __repr__(self):
        return "global(%s)" % self.name

class BuiltinType(NumbaType, minitypes.ObjectType):
    is_builtin = True

    def __init__(self, name, **kwds):
        super(BuiltinType, self).__init__(**kwds)
        self.name = name
        self.func = getattr(__builtin__, name)

    def __repr__(self):
        return "builtin(%s)" % self.name

class RangeType(NumbaType, minitypes.ObjectType):
    is_range = True

    def __repr__(self):
        return "range(...)"

class NoneType(NumbaType, minitypes.ObjectType):
    is_none = True

    def __repr__(self):
        return "<type(None)>"

#------------------------------------------------------------------------
# Function Types
#------------------------------------------------------------------------

class CTypesFunctionType(NumbaType, minitypes.ObjectType):
    is_ctypes_function = True

    def __init__(self, ctypes_func, restype, argtypes, **kwds):
        super(CTypesFunctionType, self).__init__(**kwds)
        self.ctypes_func = ctypes_func
        self.signature = minitypes.FunctionType(return_type=restype,
                                                args=argtypes)

    def __repr__(self):
        return "<ctypes function %s>" % (self.signature,)

class AutojitType(NumbaType, minitypes.ObjectType):
    """
    Type for autojitting functions.
    """

    is_autojit_function = True

    def __init__(self, autojit_func, **kwds):
        super(AutojitType, self).__init__(**kwds)
        self.autojit_func = autojit_func

    def __repr__(self):
        return "<autojit(%s)>" % self.autojit_func

class JitType(NumbaType, minitypes.ObjectType):
    """
    Type for autojitting functions.
    """

    is_jit_function = True

    def __init__(self, jit_func, **kwds):
        super(JitType, self).__init__(**kwds)
        self.jit_func = jit_func

    def __repr__(self):
        return "<jit(%s)>" % self.jit_func

#------------------------------------------------------------------------
# Pointer Types
#------------------------------------------------------------------------

class NULLType(NumbaType):
    """
    Null pointer type that can be compared or assigned to any other
    pointer type.
    """

    is_null = True

    def __repr__(self):
        return "<type(NULL)>"

class CTypesPointerType(NumbaType):
    def __init__(self, pointer_type, address, **kwds):
        super(CTypesPointer, self).__init__(**kwds)
        self.pointer_type = pointer_type
        self.address = address

class SizedPointerType(NumbaType, minitypes.PointerType):
    """
    A pointer with knowledge of its range.

    E.g. an array's 'shape' or 'strides' attribute.
    This also allow tuple unpacking.
    """

    size = None
    is_sized_pointer = True

class CastType(NumbaType, minitypes.ObjectType):
    """
    A type instance in user code. e.g. double(value). The Name node will have
    a cast-type with dst_type 'double'.
    """

    is_cast = True

    subtypes = ['dst_type']

    def __init__(self, dst_type, **kwds):
        super(CastType, self).__init__(**kwds)
        self.dst_type = dst_type

    def __repr__(self):
        return "<cast(%s)>" % self.dst_type


#------------------------------------------------------------------------
# Aggregate Types
#------------------------------------------------------------------------

class struct(minitypes.struct):
    __doc__ = minitypes.struct.__doc__

    def ref(self):
        return ReferenceType(self)

#    def __repr__(self):
#        return "struct(...)"

#------------------------------------------------------------------------
# References
#------------------------------------------------------------------------

class ReferenceType(NumbaType):
    """
    A reference to an (primitive or Python) object. This is passed as a
    pointer and dereferences automatically.

    Currently only supported for structs.
    """

    is_reference = True

    subtypes = ['referenced_type']

    def __init__(self, referenced_type, **kwds):
        super(ReferenceType, self).__init__(**kwds)
        self.referenced_type = referenced_type

    def to_llvm(self, context):
        return self.referenced_type.pointer().to_llvm(context)

    def __repr__(self):
        return "%r.ref()" % (self.referenced_type,)


#------------------------------------------------------------------------
# END OF TYPE DEFINITIONS
#------------------------------------------------------------------------

tuple_ = TupleType()
none = NoneType()
null_type = NULLType()
intp = minitypes.npy_intp


if __name__ == '__main__':
    import doctest
    doctest.testmod()
