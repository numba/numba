# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import collections

import llvm.core

from numba import typesystem
from numba.typesystem import tbaa
from numba.nodes import *
from numba.ndarray_helpers import PyArrayAccessor, NumpyArray

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

def get_shape(builder, tbaa_metadata, shape_pointer, ndim):
    "Load the shape values from an ndarray"
    shape_metadata = tbaa_metadata.get_metadata(tbaa_metadata.numpy_shape)

    for i in range(ndim):
        shape_ptr = builder.gep(shape_pointer, [_const_int(i)])
        extent = builder.load(shape_ptr)
        extent.set_metadata("tbaa", shape_metadata)
        yield extent

def get_strides(builder, tbaa_metadata, strides_pointer, ndim):
    "Load the stride values from an ndarray"
    stride_metadata = tbaa_metadata.get_metadata(tbaa.numpy_strides)

    for i in range(ndim):
        stride_ptr = builder.gep(strides_pointer, [_const_int(i)])
        stride = builder.load(stride_ptr)
        stride.set_metadata("tbaa", stride_metadata)
        yield stride

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
            type = int_
        elif attribute_name in ('shape', 'strides'):
            type = typesystem.sized_pointer(typesystem.npy_intp,
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
        self.element_type = typesystem.npy_intp
        self.type = typesystem.carray(self.element_type,
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
    return_type = typesystem.array(dtype, nd)
    signature = return_type(
                int_,                   # nd
                npy_intp.pointer(),     # shape
                object_,                # dtype
                int_)                   # fortran
    return MultiArrayAPINode(name, signature, args)

def PyArray_Zeros(args):
    return PyArray_Empty(args, name='PyArray_Zeros')
