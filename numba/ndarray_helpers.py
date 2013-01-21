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
from numba import typesystem
from numba.llvm_types import _head_len, _int32
import llvm.core as lc

const_int = lambda X: lc.Constant.int(_int32, X)

def make_property(type=None):
    """

    :param type: The type to be used for TBAA annotation
    """

    def set_metadata(tbaa, instr):
        if type is not None:
            metadata = tbaa.get_metadata(type)
            instr.set_metadata("tbaa", metadata)

    def decorator(access_func):
        def load(self):
            instr = self.builder.load(access_func(self))
            set_metadata(self.tbaa, instr)
            return instr

        def store(self, value):
            ptr = access_func(self)
            instr = self.builder.store(value, ptr)
            set_metadata(self.tbaa, instr)

        return property(load, store)

    return decorator

class PyArrayAccessor(object):
    def __init__(self, builder, pyarray_ptr, tbaa):
        self.builder = builder
        self.pyarray_ptr = pyarray_ptr
        self.tbaa = tbaa

    def _get_element(self, idx):
        indices = map(const_int, [0, _head_len + idx])
        ptr = self.builder.gep(self.pyarray_ptr, indices)
        return ptr

    @make_property()
    def data(self):
        return self._get_element(0)

    @make_property(typesystem.numpy_ndim)
    def ndim(self):
        return self._get_element(1)

    @make_property(typesystem.numpy_shape.pointer())
    def dimensions(self):
        return self._get_element(2)

    shape = dimensions

    @make_property(typesystem.numpy_strides.pointer())
    def strides(self):
        return self._get_element(3)

    @make_property(typesystem.numpy_base)
    def base(self):
        return self._get_element(4)

    @make_property(typesystem.numpy_dtype)
    def descr(self):
        return self._get_element(5)

    @make_property(typesystem.numpy_flags)
    def flags(self):
        return self._get_element(6)
    
