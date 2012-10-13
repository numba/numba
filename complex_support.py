import ctypes
import numpy as np

from minitypes import *

### Taken from Numba ###

# NOTE: The following ctypes structures were inspired by Joseph
# Heller's response to python-list question about ctypes complex
# support.  In that response, he said these were only suitable for
# Linux.  Might our milage vary?

class ComplexMixin (object):
    def _get(self):
        # FIXME: Ensure there will not be a loss of precision here!
        return self._numpy_ty_(self.real + (self.imag * 1j))

    def _set(self, value):
        self.real = value.real
        self.imag = value.imag

    value = property(_get, _set)

    @classmethod
    def from_param(cls, param):
        ret_val = cls()
        ret_val.value = param
        return ret_val

    @classmethod
    def make_ctypes_prototype_wrapper(cls, ctypes_prototype):
        '''This is a hack so that functions that return a complex type
        will construct a new Python value from the result, making the
        Numba compiled function a drop-in replacement for a Python
        function.'''
        # FIXME: See if there is some way of avoiding this additional
        # wrapper layer.
        def _make_complex_result_wrapper(in_func):
            ctypes_function = ctypes_prototype(in_func)
            def _complex_result_wrapper(*args, **kws):
                # Return the value property, not the ComplexMixin
                # instance built by ctypes.
                result = ctypes_function(*args, **kws)
                return result.value
            return _complex_result_wrapper
        return _make_complex_result_wrapper

class Complex64(ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_float), ('imag', ctypes.c_float)]
    _numpy_ty_ = np.complex64

class Complex128(ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]
    _numpy_ty_ = np.complex128

if hasattr(np, 'complex256'):
    class Complex256(ctypes.Structure, ComplexMixin):
        _fields_ = [('real', ctypes.c_longdouble), ('imag', ctypes.c_longdouble)]
        _numpy_ty_ = np.complex256
else:
    Complex256 = None

### End Taken from Numba ###