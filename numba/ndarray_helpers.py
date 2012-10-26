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

from numba.llvm_types import _head_len, _int32
import llvm.core as lc

const_int = lambda X: lc.Constant.int(_int32, X)

def make_property(access_func):
    def load(self):
        return self.builder.load(access_func(self))

    def store(self, value):
        ptr = access_func(self)
        self.builder.store(value, ptr)

    return property(load, store)

class PyArrayAccessor(object):
    def __init__(self, builder, pyarray_ptr):
        self.builder = builder
        self.pyarray_ptr = pyarray_ptr
    
    def _get_element(self, idx):
        indices = map(const_int, [0, _head_len + idx])
        ptr = self.builder.gep(self.pyarray_ptr, indices)
        return ptr

    @make_property
    def data(self):
        return self._get_element(0)

    @make_property
    def ndim(self):
        return self._get_element(1)

    @make_property
    def dimensions(self):
        return self._get_element(2)

    shape = dimensions

    @make_property
    def strides(self):
        return self._get_element(3)

    @make_property
    def base(self):
        return self._get_element(4)

    @make_property
    def descr(self):
        return self._get_element(5)

    @make_property
    def flags(self):
        return self._get_element(6)
    
