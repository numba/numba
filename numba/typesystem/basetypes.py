import __builtin__ as builtins
import math

import numpy as np
# from numpy.ctypeslib import _typecodes

import numba
from numba import  error
import numba.typesystem
from numba.minivect.minitypes import *
from numba.minivect.minitypes import map_dtype
from numba.minivect import minitypes

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

class IteratorType(NumbaType, minitypes.ObjectType):
    is_iterator = True
    subtypes = ['base_type']

    def __init__(self, iterable_type, **kwds):
        super(IteratorType, self).__init__(**kwds)
        self.iterable_type = iterable_type
        self.base_type = numba.typesystem.element_type(iterable_type)

    def __repr__(self):
        return "iterator<%s>" % (self.iterable_type,)

class KnownValueType(NumbaType, minitypes.ObjectType):
    """
    Type which is associated with a known value or well-defined symbolic
    expression:

        np.add          => np.add
        np.add.reduce   => (np.add, "reduce")

    (Remember that unbound methods like np.add.reduce are transient, i.e.
     np.add.reduce is not np.add.reduce).
    """

    is_known_value = True

    def __init__(self, value, **kwds):
        super(KnownValueType, self).__init__(**kwds)
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, KnownValueType) and self.value == other.value

    def __repr__(self):
        return "kwown_value(%s)" % (self.value,)

class ModuleType(KnownValueType):
    """
    Represents a type for modules.

    Attributes:
        is_numpy_module: whether the module is the numpy module
        module: in case of numpy, the numpy module or a submodule
    """

    is_module = True
    is_numpy_module = False
    is_object = True

    def __init__(self, module, **kwds):
        super(ModuleType, self).__init__(module, **kwds)

        self.is_numpy_module = module is np
        self.is_numba_module = module is numba
        self.is_math_module = module is math

    @property
    def module(self):
        return self.value

    def __repr__(self):
        if self.is_numpy_module:
            return 'numpy'
        else:
            return 'ModuleType'

    @property
    def comparison_type_list(self):
        return (self.module,)

class ModuleAttributeType(KnownValueType):
    """
    Attribute of a module. E.g. np.sin
    """

    is_module_attribute = True
    is_object = True

    module = None
    attr = None

    def __init__(self, module, attr, **kwds):
        self.module = module
        self.attr = attr
        value = getattr(self.module, self.attr)

        super(ModuleAttributeType, self).__init__(value, **kwds)

        base_name, dot, rest = self.module.__name__.partition(".")
        self.is_numpy_attribute = base_name == "numpy"

    def __repr__(self):
        return "%s.%s" % (self.module.__name__, self.attr)

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
    dtype = None # Numby dtype type

    subtypes = ["dtype"]

    def __init__(self, dtype, **kwds):
        super(NumpyDtypeType, self).__init__(**kwds)
        self.dtype = dtype

    def __repr__(self):
        return "NumpyDtype(%s)" % self.dtype

class TBAAType(NumbaType):
    """
    Type based alias analysis type. See numba/metadata.py.
    """

    is_tbaa = True

    def __init__(self, name, root, **kwds):
        super(TBAAType, self).__init__(**kwds)
        self.name = name
        self.root = root

    @property
    def comparison_type_list(self):
        return [self.name, self.root]

    def __repr__(self):
        return "tbaa(%s)" % self.name

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

class GlobalType(KnownValueType):
    is_global = True
    name = None

    def __init__(self, name, func_globals, position_node=None, **kwds):
        try:
            value = func_globals[name]
        except KeyError, e:
            raise error.NumbaError(position_node, "No global named %s" % (e,))

        super(GlobalType, self).__init__(value, **kwds)
        self.name = name

    def __repr__(self):
        return "global(%s)" % self.name

class BuiltinType(KnownValueType):

    is_builtin = True

    def __init__(self, name, **kwds):
        value = getattr(builtins, name)
        super(BuiltinType, self).__init__(value, **kwds)

        self.name = name
        self.func = self.value

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
        super(CTypesPointerType, self).__init__(**kwds)
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

    def __repr__(self):
        return "%r<%s>" % (self.base_type.pointer(), self.size)

    def __eq__(self, other):
        if other.is_sized_pointer:
            return (self.base_type == other.base_type and
                    self.size == other.size)
        return other.is_pointer and self.base_type == other.base_type

    def __hash__(self):
        return hash(self.base_type.pointer())

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

none = NoneType()
null_type = NULLType()
intp = minitypes.npy_intp

const_qualifiers = frozenset(["const"])

numpy_array = TBAAType("numpy array", root=object_)
numpy_shape = TBAAType("numpy shape", root=intp.pointer(),
                       qualifiers=const_qualifiers)
numpy_strides = TBAAType("numpy strides", root=intp.pointer(),
                         qualifiers=const_qualifiers)
numpy_ndim = TBAAType("numpy flags", root=int_.pointer())
numpy_dtype = TBAAType("numpy dtype", root=object_)
numpy_base = TBAAType("numpy base", root=object_)
numpy_flags = TBAAType("numpy flags", root=int_.pointer())

iteration_target_type = TBAAType("iteration target", root=char.pointer())
unique_tbaa_type = TBAAType("unique", root=char.pointer())

if __name__ == '__main__':
    import doctest
    doctest.testmod()
