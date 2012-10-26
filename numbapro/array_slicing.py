import ast

from llvm.core import Type, inline_function
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder

from numba import *
from numba import (visitors, nodes, error,
                   ast_type_inference, ast_translate,
                   ndarray_helpers)
from numba.minivect import minitypes
from numbapro.vectorize.gufunc import PyArray

class SliceDimNode(nodes.Node):
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
        self.start = subslice.lower
        self.stop = subslice.upper
        self.step = subslice.step

def create_slice_dim_node(subslice, *args):
    if subslice.type.is_slice:
        return SliceSliceNode(subslice, *args)
    else:
        return SliceDimNode(subslice, *args)

class NativeSliceNode(nodes.Node):
    """
    Aggregate of slices in all dimensions.
    """

    _fields = ['value', 'subslices']

    def __init__(self, type, value, subslices, **kwargs):
        super(NativeSliceNode, self).__init__(**kwargs)
        self.type = type
        self.value = value
        self.subslices = subslices

class SliceRewriterMixin(ast_type_inference.NumpyMixin,
                         visitors.NoPythonContextMixin):
    """
    Visitor mixin that rewrites slices to its native equivalent without
    using the Python API.

    Only works in a nopython context!
    """

    def _rewrite_slice(self, node):
        # assert self.nopython

        assert isinstance(node.slice, ast.ExtSlice)
        slices = []
        src_dim = node.value.type.ndim - 1
        dst_dim = node.type.ndim - 1
        assert node.value.type.ndim == len(node.slice.dims)

        all_slices = True
        for subslice in node.slice.dims:
            slices.append(create_slice_dim_node(subslice, src_dim, dst_dim))

            src_dim -= 1
            is_newaxis = self._is_newaxis(subslice)
            if subslice.type.is_slice or is_newaxis:
                all_slices = all_slices and not is_newaxis
                dst_dim -= 1
            else:
                assert subslice.type.is_int
                all_slices = False

        assert src_dim + 1 == 0
        assert dst_dim + 1 == 0

        #if all_slices and all(empty(subslice) for subslice in slices):
        #    return node.value

        return NativeSliceNode(node.type, node.value, slices)

    def visit_Subscript(self, node):
        node = super(SliceRewriterMixin, self).visit_Subscript(node)
        if (isinstance(node, ast.Subscript) and node.value.type.is_array and
                node.type.is_array):
            node = self._rewrite_slice(node)
        return node

class NativeSliceCodegenMixin(object): # ast_translate.LLVMCodeGenerator):

    def __init__(self, *args, **kwds):
        super(NativeSliceCodegenMixin, self).__init__(*args, **kwds)

        newaxis_func_def = NewAxis()
        self.newaxis_func = newaxis_func_def(self.mod)

        index_func_def = IndexAxis()
        self.index_func = index_func_def(self.mod)

    def visit_NativeSliceNode(self, node):
        array_ltype = PyArray.llvm_type()
        shape_ltype = npy_intp.to_llvm(self.context)

        view = self.visit(node.value)
        view_copy = self.llvm_alloca(array_ltype)
        self.builder.store(self.builder.load(view), view_copy)

        view_accessor = ndarray_helpers.PyArrayAccessor(self.builder, view)
        view_copy_accessor = ndarray_helpers.PyArrayAccessor(self.builder,
                                                             view_copy)

        shape_type = minitypes.CArrayType(npy_intp, node.type.ndim)
        shape = self.alloca(shape_type)
        strides = self.alloca(shape_type)

        view_copy_accessor.data = view_accessor.data
        view_copy_accessor.dimensions = self.builder.bitcast(shape, shape_ltype)
        view_copy_accessor.strides = strides

        for subslice in node.subslices:
            subslice.view_accessor = view_accessor
            subslice.view_copy_accessor = view_copy_accessor

        self.visitlist(node.subslices)
        return view_copy

    def visit_SliceSliceNode(self, node):
        start, stop, step = node.start, node.stop, node.step

        if start is not None:
            start = self.visit(node.start)
        if stop is not None:
            stop = self.visit(node.stop)
        if step is not None:
            step = self.visit(node.step)

        slice_func_def = SliceArray(node.src_dim, node.dst_dim,
                                    start, stop, step)
        slice_func = slice_func_def(self.mod)

        data = node.view_copy_accessor.data
        in_shape = node.view_accessor.shape
        in_strides = node.view_accessor.strides
        out_shape = node.view_copy_accessor.shape
        out_strides = node.view_copy_accessor.strides

        data_p = slice_func(data, in_shape, in_strides, out_shape, out_strides)
        node.view_copy_accessor.data = data_p

        return None

    def visit_SliceDimNode(self, node):
        acc_copy = node.view_copy_accessor
        acc = node.view_accessor
        if node.type.is_int:
            value = self.visit(node.subslice)
            acc_copy.data = self.index_func(acc_copy.data,
                                            acc.shape, acc.strides,
                                            node.src_dim, value)
        else:
            self.newaxis_func(acc_copy.shape, acc_copy.strides, node.dst_dim)

        return None

class SliceArray(CDefinition):

    _name_ = "slice"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),
        ('in_shape', C.pointer(C.npy_intp)),
        ('in_strides', C.pointer(C.npy_intp)),
        ('out_shape', C.pointer(C.npy_intp)),
        ('out_strides', C.pointer(C.npy_intp))
    ]

    def _adjust_given_index(self, extent, negative_step, index, is_start):
        # Tranliterate the below code to llvm cbuilder

        # For the start index in start:stop:step, do:
        # if have_start:
        #     if start < 0:
        #         start += shape
        #         if start < 0:
        #             start = 0
        #     elif start >= shape:
        #         if negative_step:
        #             start = shape - 1
        #         else:
        #             start = shape
        # else:
        #     if negative_step:
        #         start = shape - 1
        #     else:
        #         start = 0

        # For the stop index, do:
        # if stop is not None:
        #     if stop < 0:
        #         stop += extent
        #         if stop < 0:
        #             stop = 0
        #     elif stop > extent:
        #         stop = extent
        # else:
        #     if negative_step:
        #         stop = -1
        #     else:
        #         stop = extent

        with self.parent.ifelse(index < 0) as ifelse:
            with ifelse.then():
                index += extent
                if index < 0:
                    index.assign(0)

            with ifelse.otherwise():
                with self.parent.ifelse(index >= extent) as ifelse:
                    if is_start:
                        # index is 'start' index
                        with self.parent.ifelse(negative_step) as ifelse:
                            with ifelse.then():
                                index.assign(extent - 1)
                            with ifelse.otherwise():
                                index.assign(extent)
                    else:
                        # index is 'stop' index. Stop is exclusive, to
                        # we don't care about the sign of the step
                        index.assign(extent)

    def _set_default_index(self, default1, default2, negative_step, start):
        with self.parent.ifelse(negative_step) as ifelse:
            with ifelse.then():
                start.assign(default1)
            with ifelse.otherwise():
                start.assign(default2)

    def adjust_index(self, extent, negative_step, index, default1, default2,
                     is_start=False):
        if index is not None:
            self._adjust_given_index(extent, negative_step, index, is_start)
        else:
            index = self.parent.var(C.npy_intp)
            self._set_default_index(default1, default2, negative_step, index)

        return index

    def body(self, data, in_shape, in_strides, out_shape, out_strides):
        start = self.start
        stop = self.stop
        step = self.step

        dim = self.src_dimension
        stride = strides[dim]
        extent = shape[dim]

        negative_step = step < 0

        start = self.adjust_index(extent, negative_step, start, extent - 1, 0,
                                  is_start=True)
        stop = self.adjust_index(extent, negative_step, start, -1, extent)
        if step is None:
            step = 1

        new_extent = (stop - start) / step
        with self.parent.ifelse((stop - start) % step != 0) as ifelse:
            with ifelse.then():
                new_extent += 1

        with self.parent.ifelse(new_extent < 0) as ifelse:
            with ifelse.then():
                new_extent.assign(0)

        # if extent == 1, set stride to 0 for broadcasting
        with self.parent.ifelse(new_extent == 1) as ifelse:
            with ifelse.then():
                stride.assign(0)

        dst_dim = self.dst_dimension

        result = self.parent.var(data.type, name='result')
        result.assign(data + start * stride)
        shape[dim] = new_extent
        strides[dim] = stride * step

        self.ret(result)

    @classmethod # specializing classes is bad, ameliorate
    def specialize(cls, src_dimension, dst_dimension, start, stop, step):
        cls.FuncDef = func_def
        cls.src_dimension = src_dimension
        cls.dst_dimension = dst_dimension
        cls.start = start
        cls.stop = stop
        cls.step = step

class IndexAxis(CDefinition):

    _name_ = "index"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),
        ('in_shape', C.pointer(C.npy_intp)),
        ('in_strides', C.pointer(C.npy_intp)),
        ('src_dim', C.npy_intp),
        ('index', C.npy_intp),
    ]

    def body(self, data, in_shape, in_strides, index, src_dim):
        result = self.parent.var(data.type, name='result')
        result.assign(data + in_strides[src_dim] * index)
        self.ret(result)

    @classmethod
    def specialize(cls):
        pass

class NewAxis(CDefinition):

    _name_ = "newaxis"
    _argtys_ = [
        ('out_shape', C.pointer(C.npy_intp)),
        ('out_strides', C.pointer(C.npy_intp)),
        ('dst_dim', C.npy_intp),
    ]

    def body(self, out_shape, out_strides, dst_dim):
        out_shape[dst_dim] = 1
        out_strides[dst_dim] = 0

    @classmethod
    def specialize(cls):
        pass
