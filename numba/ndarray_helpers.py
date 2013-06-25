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

import abc

from numba import *
from numba import typedefs
from numba.typesystem import tbaa
from numba.llvm_types import _head_len, _int32, _LLVMCaster, constant_int
import llvm.core as lc

def _const_int(X):
    return lc.Constant.int(lc.Type.int(), X)

def ptr_at(builder, ptr, idx):
    return builder.gep(ptr, [_const_int(idx)])

def load_at(builder, ptr, idx):
    return builder.load(ptr_at(builder, ptr, idx))

def store_at(builder, ptr, idx, val):
    builder.store(val, ptr_at(builder, ptr, idx))


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
        self.tbaa = tbaa # this may be None
        self.dtype = dtype

    def _get_element(self, idx):
        indices = [constant_int(0), constant_int(_head_len + idx)]
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


class Array(object):
    """
    Interface for foreign arrays, like LLArray
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def from_type(cls, llvm_dtype):
        """
        Given an LLVM representation of the dtype, return the LLVM array type
        representation
        """

    @abc.abstractproperty
    def data(self):
        """Return the data pointer of this array (for A.data)"""

    @abc.abstractproperty
    def shape_ptr(self):
        """Return the shape pointer of this array (for A.shape[0], etc)"""

    @abc.abstractproperty
    def strides_ptr(self):
        """Return the strides pointer of this array (for A.strides[0], etc)"""

    @abc.abstractproperty
    def shape(self):
        """Return the extents as a list of loaded LLVM values"""

    @abc.abstractproperty
    def strides(self):
        """Return the strides as a list of loaded LLVM values"""

    @abc.abstractproperty
    def ndim(self):
        """Return the dimensionality of this array as an LLVM constant"""

    @abc.abstractmethod
    def getptr(self, *indices):
        """Compute an element pointer given LLVM indices into the array"""


class NumpyArray(Array):
    """
    LLArray compatible inferface for NumPy's ndarray
    """

    _strides_ptr = None
    _strides = None
    _shape_ptr = None
    _shape = None
    _data_ptr = None
    _freefuncs = []
    _freedata = []

    def __init__(self, pyarray_ptr, builder, tbaa=None, type=None):
        self.type = type
        self.nd = type.ndim
        self.array_type = pyarray_ptr.type.pointee

        # LLVM attributes
        self.arr = PyArrayAccessor(builder, pyarray_ptr, tbaa, type.dtype)
        self.builder = builder
        self._shape = None
        self._strides = None
        self.caster = _LLVMCaster(builder)

    @classmethod
    def from_type(cls, llvm_dtype):
        return typedefs.PyArray.pointer().to_llvm()

    @property
    def data(self):
        if not self._data_ptr:
            self._data_ptr = self.arr.get_data()
        return self._data_ptr

    @property
    def shape_ptr(self):
        if self._shape_ptr is None:
            self._shape_ptr = self.arr.shape
        return self._shape_ptr

    @property
    def strides_ptr(self):
        if self._strides_ptr is None:
            self._strides_ptr = self.arr.strides
        return self._strides_ptr

    @property
    def shape(self):
        if not self._shape:
            self._shape = self.preload(self.shape_ptr, self.nd)
        return self._shape

    @property
    def strides(self):
        if not self._strides:
            self._strides = self.preload(self.strides_ptr, self.nd)
        return self._strides

    @property
    def ndim(self):
        return _const_int(self.nd)

    def getptr(self, *indices):
        offset = _const_int(0)
        for i, (stride, index) in enumerate(zip(self.strides, indices)):
            index = self.caster.cast(index, stride.type, unsigned=False)
            offset = self.caster.cast(offset, stride.type, unsigned=False)
            offset = self.builder.add(offset, self.builder.mul(index, stride))

        data_ty = self.type.dtype.to_llvm()
        data_ptr_ty = lc.Type.pointer(data_ty)

        dptr_plus_offset = self.builder.gep(self.data, [offset])

        ptr = self.builder.bitcast(dptr_plus_offset, data_ptr_ty)
        return ptr

    # Misc, optional methods

    @property
    def itemsize(self):
        raise NotImplementedError

    def preload(self, ptr, count=None):
        assert count is not None
        return [load_at(self.builder, ptr, i) for i in range(count)]