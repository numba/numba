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

class PyArrayAccessor(object):
    def __init__(self, builder, pyarray_ptr):
        self.builder = builder
        self.pyarray_ptr = pyarray_ptr

    def _get_element(self, idx):
        indices = map(const_int, [0, _head_len + idx])
        ptr = self.builder.gep(self.pyarray_ptr, indices)
        return self.builder.load(ptr)

    @property
    def data(self):
        return self._get_element(0)

    @property
    def ndim(self):
        return self._get_element(1)

    @property
    def dimensions(self):
        return self._get_element(2)

    @property
    def strides(self):
        return self._get_element(3)

    @property
    def base(self):
        return self._get_element(4)

    @property
    def descr(self):
        return self._get_element(5)

    @property
    def flags(self):
        return self._get_element(6)


