import collections

from numba.nodes import *
import numba.nodes
from numba.ndarray_helpers import PyArrayAccessor

#----------------------------------------------------------------------------
# External Utilities
#----------------------------------------------------------------------------

def is_constant_index(node):
    return (isinstance(node, ast.Index) and
            isinstance(node.value, ConstNode))

def is_newaxis(node):
    v = node.variable
    return (is_constant_index(node) and
            node.value.pyval is None) or v.type.is_newaxis or v.type.is_none

def is_ellipsis(node):
    return is_constant_index(node) and node.value.pyval is Ellipsis


#----------------------------------------------------------------------------
# Internal Utilities
#----------------------------------------------------------------------------

def _const_int(X):
    return llvm.core.Constant.int(llvm.core.Type.int(), X)

#----------------------------------------------------------------------------
# NumPy Array Attributes
#----------------------------------------------------------------------------

class DataPointerNode(Node):

    _fields = ['node', 'slice']

    def __init__(self, node, slice, ctx):
        self.node = node
        self.slice = slice
        self.type = node.type.dtype
        self.variable = Variable(self.type)
        self.ctx = ctx

    def data_descriptors(self, llvm_value, builder):
        '''
        Returns a tuple of (dptr, strides)
        - dptr:    a pointer of the data buffer
        - strides: a pointer to an array of stride information;
                   has `ndim` elements.
        '''
        acc = PyArrayAccessor(builder, llvm_value)
        return acc.data, acc.strides

    def subscript(self, translator, llvm_value, indices):
        builder = translator.builder
        caster = translator.caster
        context = translator.context

        dptr, strides = self.data_descriptors(llvm_value, builder)
#        data_ty = self.type.to_llvm(context)
#        data_ptr_ty = llvm.core.Type.pointer(data_ty)
#        ptr = builder.bitcast(dptr, data_ptr_ty)
#        return ptr

        ndim = self.ndim

        offset = _const_int(0)

        if not isinstance(indices, collections.Iterable):
            indices = (indices,)

        for i, index in zip(range(ndim), indices):
            # why is the indices reversed?
            stride_ptr = builder.gep(strides, [_const_int(i)])
            stride = builder.load(stride_ptr, invariant=True)
            index = caster.cast(index, stride.type)
            offset = caster.cast(offset, stride.type)
            offset = builder.add(offset, builder.mul(index, stride))

        data_ty = self.type.to_llvm(context)
        data_ptr_ty = llvm.core.Type.pointer(data_ty)

        dptr_plus_offset = builder.gep(dptr, [offset])

        ptr = builder.bitcast(dptr_plus_offset, data_ptr_ty)
        return ptr

    @property
    def ndim(self):
        return self.node.type.ndim

    def __repr__(self):
        return "%s.data" % self.node

class ArrayAttributeNode(Node):
    is_read_only = True

    _fields = ['array']

    def __init__(self, attribute_name, array):
        self.array = array
        self.attr_name = attribute_name

        self.array_type = array.variable.type
        if attribute_name == 'ndim':
            type = minitypes.int_
        elif attribute_name in ('shape', 'strides'):
            type = typesystem.SizedPointerType(typesystem.intp,
                                                size=self.array_type.ndim)
        elif attribute_name == 'data':
            type = self.array_type.dtype.pointer()
        else:
            raise error._UnknownAttribute(attribute_name)

        self.type = type

    def __repr__(self):
        return "%s.%s" % (self.array, self.attr_name)

class ShapeAttributeNode(ArrayAttributeNode):
    # NOTE: better do this at code generation time, and not depend on
    #       variable.lvalue
    _fields = ['array']

    def __init__(self, array):
        super(ShapeAttributeNode, self).__init__('shape', array)
        self.array = array
        self.element_type = typesystem.intp
        self.type = minitypes.CArrayType(self.element_type,
                                         array.variable.type.ndim)

#----------------------------------------------------------------------------
# NumPy Array Creation
#----------------------------------------------------------------------------

class ArrayNewNode(Node):
    """
    Allocate a new array given the attributes.
    """

    _fields = ['data', 'shape', 'strides', 'base']

    def __init__(self, type, data, shape, strides, base=None, **kwargs):
        super(ArrayNewNode, self).__init__(**kwargs)
        self.type = type
        self.data = data
        self.shape = shape
        self.strides = strides
        self.base = base

class ArrayNewEmptyNode(Node):
    """
    Allocate a new array with data.
    """

    _fields = ['shape']

    def __init__(self, type, shape, is_fortran=False, **kwargs):
        super(ArrayNewEmptyNode, self).__init__(**kwargs)
        self.type = type
        self.shape = shape
        self.is_fortran = is_fortran


#----------------------------------------------------------------------------
# Nodes for NumPy calls
#----------------------------------------------------------------------------

shape_type = npy_intp.pointer()
void_p = void.pointer()

class MultiArrayAPINode(NativeCallNode):

    def __init__(self, name, signature, args):
        super(MultiArrayAPINode, self).__init__(signature, args,
                                                llvm_func=None)
        self.func_name = name

def PyArray_NewFromDescr(args):
    """
    Low-level specialized equivalent of ArrayNewNode
    """
    signature = object_(
        object_,    # subtype
        object_,    # descr
        int_,       # ndim
        shape_type, # shape
        shape_type, # strides
        void_p,     # data
        int_,       # flags
        object_,    # obj
    )

    return MultiArrayAPINode('PyArray_NewFromDescr', signature, args)

def PyArray_SetBaseObject(args):
    signature = int_(object_, object_)
    return MultiArrayAPINode('PyArray_SetBaseObject', signature, args)

def PyArray_UpdateFlags(args):
    return MultiArrayAPINode('PyArray_UpdateFlags', void(object_, int_), args)

def PyArray_Empty(args, name='PyArray_Empty'):
    nd, shape, dtype, fortran = args
    return_type = minitypes.ArrayType(dtype, nd)
    signature = return_type(
                int_,                   # nd
                npy_intp.pointer(),     # shape
                object_,                # dtype
                int_)                   # fortran
    return MultiArrayAPINode(name, signature, args)

def PyArray_Zeros(args):
    return PyArray_Empty(args, name='PyArray_Zeros')
