# -*- coding: utf-8 -*-
"""
AST nodes for native slicing.
"""
from __future__ import print_function, division, absolute_import

import ast

import numba
from numba import *
from numba import nodes

class SliceDimNode(nodes.ExprNode):
    """
    Array is sliced, and this dimension contains an integer index or newaxis.
    """

    _fields = ['subslice']

    def __init__(self, subslice, src_dim, dst_dim, **kwargs):
        super(SliceDimNode, self).__init__(**kwargs)
        self.subslice = subslice
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.type = subslice.type

        # PyArrayAccessor wrapper of llvm fake PyArrayObject value
        # set by NativeSliceNode
        self.view_accessor = None
        self.view_copy_accessor = None

class SliceSliceNode(SliceDimNode):
    """
    Array is sliced, and this dimension contains a slice.
    """

    _fields = ['start', 'stop', 'step']

    def __init__(self, subslice, src_dim, dst_dim, **kwargs):
        super(SliceSliceNode, self).__init__(subslice, src_dim, dst_dim,
                                             **kwargs)
        self.start = subslice.lower and nodes.CoercionNode(subslice.lower, npy_intp)
        self.stop = subslice.upper and nodes.CoercionNode(subslice.upper, npy_intp)
        self.step = subslice.step and nodes.CoercionNode(subslice.step, npy_intp)

class BroadcastNode(nodes.ExprNode):
    """
    Broadcast a bunch of operands:

        - set strides of single-sized dimensions to zero
        - find big shape
    """

    _fields = ['operands', 'check_errors']

    def __init__(self, array_type, operands, **kwargs):
        super(BroadcastNode, self).__init__(**kwargs)
        self.operands = operands

        self.shape_type = numba.carray(npy_intp, array_type.ndim)
        self.array_type = array_type
        self.type = npy_intp.pointer()

        self.broadcast_retvals = {}
        self.check_errors = []

        for op in operands:
            if op.type.is_array:
                # TODO: Put the raise code in a separate basic block and jump
                return_value = nodes.LLVMValueRefNode(int_, None)
                check_error = nodes.CheckErrorNode(
                        return_value, 0, exc_type=ValueError,
                        exc_msg="Shape mismatch while broadcasting")

                self.broadcast_retvals[op] = return_value
                self.check_errors.append(check_error)

def create_slice_dim_node(subslice, *args):
    if subslice.type.is_slice:
        return SliceSliceNode(subslice, *args)
    else:
        return SliceDimNode(subslice, *args)

class NativeSliceNode(nodes.ExprNode):
    """
    Aggregate of slices in all dimensions.

    In nopython context, uses a fake stack-allocated PyArray struct.

    In python context, it builds an actual heap-allocated numpy array.
    In this case, the following attributes are patched during code generation
    time that sets the llvm values:

        dst_data, dst_shape, dst_strides
    """

    _fields = ['value', 'subslices', 'build_array_node']

    def __init__(self, type, value, subslices, nopython, **kwargs):
        super(NativeSliceNode, self).__init__(**kwargs)
        value = nodes.CloneableNode(value)

        self.type = type
        self.value = value
        self.subslices = subslices

        self.shape_type = numba.carray(npy_intp, type.ndim)
        self.nopython = nopython
        if not nopython:
            self.build_array_node = self.build_array()
        else:
            self.build_array_node = None

    def mark_nopython(self):
        self.nopython = True
        self.build_array_node = None

    def build_array(self):
        self.dst_data = nodes.LLVMValueRefNode(void.pointer(), None)
        self.dst_shape = nodes.LLVMValueRefNode(self.shape_type, None)
        self.dst_strides = nodes.LLVMValueRefNode(self.shape_type, None)
        array_node = nodes.ArrayNewNode(
                self.type, self.dst_data, self.dst_shape, self.dst_strides,
                base=self.value.clone)
        return nodes.CoercionNode(array_node, self.type)


def rewrite_slice(node, nopython):
    """
    Rewrites array slices to its native equivalent without
    using the Python API.

        node:       ast.Subscript with an array type as result
        nopython:   whether the node is encountered in a nopython context
    """
    # assert self.nopython

    if isinstance(node.slice, ast.ExtSlice):
        dims = node.slice.dims
    else:
        assert not isinstance(node.slice, ast.Ellipsis)
        dims = [node.slice]

    slices = []
    src_dim = 0
    dst_dim = 0

    all_slices = True
    for subslice in dims:
        slices.append(create_slice_dim_node(subslice, src_dim, dst_dim))

        if subslice.type.is_slice:
            src_dim += 1
            dst_dim += 1
        elif nodes.is_newaxis(subslice):
            all_slices = False
            dst_dim += 1
        else:
            assert subslice.type.is_int
            all_slices = False
            src_dim += 1

    #if all_slices and all(empty(subslice) for subslice in slices):
    #    return node.value

    # print node, node.type
    return NativeSliceNode(node.type, node.value, slices, nopython)


class MarkNoPython(ast.NodeVisitor):
    """
    Mark array slicing nodes as nopython, which allows them to use
    stack-allocated fake arrays.
    """

    def visit_NativeSliceNode(self, node):
        node.mark_nopython()
        self.generic_visit(node)
        return node

def mark_nopython(ast):
    MarkNoPython().visit(ast)
