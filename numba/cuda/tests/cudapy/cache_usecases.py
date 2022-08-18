from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys


class UseCase:
    """
    Provide a way to call a kernel as if it were a function.

    This allows the CUDA cache tests to closely match the CPU cache tests, and
    also to support calling cache use cases as njitted functions. The class
    wraps a function that takes an array for the return value and arguments,
    and provides an interface that accepts arguments, launches the kernel
    appropriately, and returns the stored return value.

    The return type is inferred from the type of the first argument, unless it
    is explicitly overridden by the ``retty`` kwarg.
    """
    def __init__(self, func, retty=None):
        self._func = func
        self._retty = retty

    def __call__(self, *args):
        array_args = [np.asarray(arg) for arg in args]
        if self._retty:
            array_return = np.ndarray((), dtype=self._retty)
        else:
            array_return = np.zeros_like(array_args[0])

        self._call(array_return, *array_args)
        return array_return[()]

    @property
    def func(self):
        return self._func


class CUDAUseCase(UseCase):
    def _call(self, ret, *args):
        self._func[1, 1](ret, *args)


@cuda.jit(cache=True)
def add_usecase_kernel(r, x, y):
    r[()] = x[()] + y[()] + Z


@cuda.jit(cache=False)
def add_nocache_usecase_kernel(r, x, y):
    r[()] = x[()] + y[()] + Z


add_usecase = CUDAUseCase(add_usecase_kernel)
add_nocache_usecase = CUDAUseCase(add_nocache_usecase_kernel)

Z = 1


# Inner / outer cached / uncached cases

@cuda.jit(cache=True)
def inner(x, y):
    return x + y + Z


@cuda.jit(cache=True)
def outer_kernel(r, x, y):
    r[()] = inner(-y[()], x[()])


@cuda.jit(cache=False)
def outer_uncached_kernel(r, x, y):
    r[()] = inner(-y[()], x[()])


outer = CUDAUseCase(outer_kernel)
outer_uncached = CUDAUseCase(outer_uncached_kernel)


# Exercise returning a record instance.  This used to hardcode the dtype
# pointer's value in the bitcode.

packed_record_type = np.dtype([('a', np.int8), ('b', np.float64)])
aligned_record_type = np.dtype([('a', np.int8), ('b', np.float64)], align=True)

packed_arr = np.empty(2, dtype=packed_record_type)
for i in range(packed_arr.size):
    packed_arr[i]['a'] = i + 1
    packed_arr[i]['b'] = i + 42.5

aligned_arr = np.array(packed_arr, dtype=aligned_record_type)


@cuda.jit(cache=True)
def record_return(r, ary, i):
    r[()] = ary[i]


record_return_packed = CUDAUseCase(record_return, retty=packed_record_type)
record_return_aligned = CUDAUseCase(record_return, retty=aligned_record_type)


# Closure test cases

def make_closure(x):
    @cuda.jit(cache=True)
    def closure(r, y):
        r[()] = x + y[()]

    return CUDAUseCase(closure)


closure1 = make_closure(3)
closure2 = make_closure(5)
closure3 = make_closure(7)
closure4 = make_closure(9)


# Ambiguous / renamed functions

@cuda.jit(cache=True)
def ambiguous_function(r, x):
    r[()] = x[()] + 2


renamed_function1 = CUDAUseCase(ambiguous_function)


@cuda.jit(cache=True)
def ambiguous_function(r, x):
    r[()] = x[()] + 6


renamed_function2 = CUDAUseCase(ambiguous_function)


# Simple use case for multiprocessing test

@cuda.jit(cache=True)
def simple_usecase_kernel(r, x):
    r[()] = x[()]


simple_usecase_caller = CUDAUseCase(simple_usecase_kernel)


class _TestModule(CUDATestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        self.assertPreciseEqual(mod.add_usecase(2, 3), 6)
        self.assertPreciseEqual(mod.outer_uncached(3, 2), 2)
        self.assertPreciseEqual(mod.outer(3, 2), 2)

        packed_rec = mod.record_return_packed(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(packed_rec), (2, 43.5))
        aligned_rec = mod.record_return_aligned(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(aligned_rec), (2, 43.5))

        mod.simple_usecase_caller(2)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
