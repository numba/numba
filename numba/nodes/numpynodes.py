# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import collections

from numba import typesystem
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

def get_shape(builder, tbaa, shape_pointer, ndim):
    "Load the shape values from an ndarray"
    shape_metadata = tbaa.get_metadata(typesystem.numpy_shape)

    for i in range(ndim):
        shape_ptr = builder.gep(shape_pointer, [_const_int(i)])
        extent = builder.load(shape_ptr)
        extent.set_metadata("tbaa", shape_metadata)
        yield extent

def get_strides(builder, tbaa, strides_pointer, ndim):
    "Load the stride values from an ndarray"
    stride_metadata = tbaa.get_metadata(typesystem.numpy_strides)

    for i in range(ndim):
        stride_ptr = builder.gep(strides_pointer, [_const_int(i)])
        stride = builder.load(stride_ptr)
        stride.set_metadata("tbaa", stride_metadata)
        yield stride

#----------------------------------------------------------------------------
# Utilities for ndarray attribute preloading
#----------------------------------------------------------------------------

# NOTE: See also optimize.py:Preloader

def make_shape_phis(context, builder, attr, name, ndim):
    "Make phi values for the extents of an array variable merge"
    ltype = npy_intp.to_llvm(context)
    for i in range(ndim):
        yield builder.phi(ltype, "%s%d(%s)" % (attr, i, name))

def make_strides_phis(context, builder, name, ndim):
    "Make phi values for the strides of an array variable merge"
    ltype = typesystem.numpy_strides.to_llvm(context)
    for i in range(ndim):
        yield builder.phi(ltype, "strides%d(%s)" % name)

def make_preload_phi(context, builder, phi_node):
    """
    Build phi values for preloaded data/shape/stride values.
    """
    var = phi_node.variable
    name = var.unmangled_name
    ndim = var.type.ndim

    if var.preload_data:
        ltype = char.pointer().to_llvm(context)
        var.preloaded_data = builder.phi(ltype,
                                         "data(%s)" % name)
    if var.preload_shape:
        var.preloaded_shape = tuple(make_shape_phis(context, builder,
                                                    "shape", name, ndim))
    if var.preload_strides:
        var.preloaded_strides = tuple(make_shape_phis(context, builder,
                                                      "strides", name, ndim))

def update_preloaded_phi(phi_var, incoming_var, llvm_incoming_block):
    if phi_var.preload_data:
        phi_var.preloaded_data.add_incoming(incoming_var.preloaded_data,
                                            llvm_incoming_block)

    if phi_var.preload_shape:
        for extent, incoming in zip(phi_var.preloaded_shape,
                                    incoming_var.preloaded_shape):
            extent.add_incoming(incoming, llvm_incoming_block)

    if phi_var.preload_strides:
        for stride, incoming in zip(phi_var.preloaded_strides,
                                    incoming_var.preloaded_strides):
            stride.add_incoming(incoming, llvm_incoming_block)


#----------------------------------------------------------------------------
# NumPy Array Attributes
#----------------------------------------------------------------------------

class DataPointerNode(ExprNode):

    _fields = ['node', 'slice']

    def __init__(self, node, slice, ctx):
        self.node = node
        self.slice = slice
        self.type = node.type.dtype
        self.variable = Variable(self.type)
        self.ctx = ctx

    def data_descriptors(self, builder, tbaa, llvm_value):
        '''
        Returns a tuple of (dptr, strides)
        - dptr:    a pointer of the data buffer
        - strides: a pointer to an array of stride information;
                   has `ndim` elements.
        '''
        acc = PyArrayAccessor(builder, llvm_value, tbaa, self.type)

        var = self.node.variable

        # Load the data pointer. Use preloaded value if available
        if var.preload_data:
            # print "using preloaded data", var.preloaded_data
            dptr = var.preloaded_data
        else:
            dptr = acc.data

        # Load the strides. Use preloaded values if available
        if var.preload_strides:
            # print "using preloaded strides", var.preloaded_strides
            strides = var.preloaded_strides
        else:
            strides_pointer = acc.strides
            strides = get_strides(builder, tbaa, strides_pointer, self.ndim)

        assert dptr is not None
        assert strides is not None

        return dptr, strides

    def subscript(self, translator, tbaa, llvm_value, indices):
        builder = translator.builder
        caster = translator.caster
        context = translator.context

        offset = _const_int(0)

        if not isinstance(indices, collections.Iterable):
            indices = (indices,)

        dptr, strides = self.data_descriptors(builder, tbaa, llvm_value)

        for i, (stride, index) in enumerate(zip(strides, indices)):
            index = caster.cast(index, stride.type, unsigned=False)
            offset = caster.cast(offset, stride.type, unsigned=False)
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

class ArrayAttributeNode(ExprNode):
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

class ArrayNewNode(ExprNode):
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

class ArrayNewEmptyNode(ExprNode):
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
