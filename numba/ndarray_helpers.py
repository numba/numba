# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
# For reference:
#    typedef struct {
#    PyObject_HEAD                   // indices (skipping the head)
#    char *data;                     // 0
#    int nd;                         // 1
#    int *dimensions, *strides;      // 2, 3
#    PyObject *base;                 // 4
#    PyArray_Descr *descr;           // 5
#    int flags;                      // 6
#    } PyArrayObject;

from numba import *
from numba.typesystem import tbaa
from numba.llvm_types import _head_len, _int32
import llvm.core as lc

const_int = lambda X: lc.Constant.int(_int32, X)

def set_metadata(tbaa, instr, type):
    if type is not None:
        metadata = tbaa.get_metadata(type)
        instr.set_metadata("tbaa", metadata)

def make_property(type=None, invariant=True):
    """
    type: The type to be used for TBAA annotation
    """
    def decorator(access_func):
        def load(self):
            instr = self.builder.load(access_func(self))
            if self.tbaa:
                set_metadata(self.tbaa, instr, type)
            return instr

        def store(self, value):
            ptr = access_func(self)
            instr = self.builder.store(value, ptr)
            if self.tbaa:
                set_metadata(self.tbaa, instr, type)

        return property(load, store)

    return decorator

class PyArrayAccessor(object):
    """
    Convenient access to a the native fields of a NumPy array.

    builder: llvmpy IRBuilder
    pyarray_ptr: pointer to the numpy array
    tbaa: metadata.TBAAMetadata instance
    """

    def __init__(self, builder, pyarray_ptr, tbaa=None, dtype=None):
        self.builder = builder
        self.pyarray_ptr = pyarray_ptr
        self.tbaa = tbaa # this maybe None
        self.dtype = dtype

    def _get_element(self, idx):
        indices = [const_int(0), const_int(_head_len + idx)]
        ptr = self.builder.gep(self.pyarray_ptr, indices)
        return ptr

    def get_data(self):
        instr = self.builder.load(self._get_element(0))
        if self.tbaa:
            set_metadata(self.tbaa, instr, self.dtype.pointer())
        return instr

    def set_data(self, value):
        instr = self.builder.store(value, self._get_element(0))
        if self.tbaa:
            set_metadata(self.tbaa, instr, self.dtype.pointer())

    data = property(get_data, set_data, "The array.data attribute")

    def typed_data(self, context):
        data = self.data
        ltype = self.dtype.pointer().to_llvm(context)
        return self.builder.bitcast(data, ltype)

    @make_property(tbaa.numpy_ndim)
    def ndim(self):
        return self._get_element(1)

    @make_property(tbaa.numpy_shape.pointer().qualify("const"))
    def dimensions(self):
        return self._get_element(2)

    shape = dimensions

    @make_property(tbaa.numpy_strides.pointer().qualify("const"))
    def strides(self):
        return self._get_element(3)

    @make_property(tbaa.numpy_base)
    def base(self):
        return self._get_element(4)

    @make_property(tbaa.numpy_dtype)
    def descr(self):
        return self._get_element(5)

    @make_property(tbaa.numpy_flags)
    def flags(self):
        return self._get_element(6)
    
