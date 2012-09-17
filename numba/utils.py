import opcode
import ctypes

from numpy import complex64, complex128

from .minivect import miniast, minitypes
from . import _numba_types

def itercode(code):
    """Return a generator of byte-offset, opcode, and argument
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    n = len(code)
    while i < n:
        c = code[i]
        num = i
        op = ord(c)
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536L

        delta = yield num, op, oparg
        if delta is not None:
            abs_rel, dst = delta
            assert abs_rel == 'abs' or abs_rel == 'rel'
            i = dst if abs_rel == 'abs' else i + dst

def debugout(*args):
    '''This is a magic function.  If you use it in compiled functions,
    Numba should generate code for displaying the received value.'''
    if __debug__:
        print("debugout (non-translated): %s" % (''.join((str(arg)
                                                          for arg in args)),))


class NumbaContext(miniast.LLVMContext):
    # debug = True
    # debug_elements = True

    # Accept dynamic arguments
    astbuilder_cls = miniast.DynamicArgumentASTBuilder

    shape_type = minitypes.npy_intp.pointer()
    strides_type = shape_type

    def init(self):
        self.astbuilder = self.astbuilder_cls(self)
        self.typemapper = _numba_types.NumbaTypeMapper(self)

def get_minivect_context():
    return NumbaContext()

context = get_minivect_context()

# NOTE: The following ctypes structures were inspired by Joseph
# Heller's response to python-list question about ctypes complex
# support.  In that response, he said these were only suitable for
# Linux.  Might our milage vary?

class ComplexMixin (object):
    def _get (self):
        # FIXME: Ensure there will not be a loss of precision here!
        return self._numpy_ty_(self.real + (self.imag * 1j))

    def _set (self, value):
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
                return ctypes_function(*args, **kws).value
            return _complex_result_wrapper
        return _make_complex_result_wrapper

class Complex64 (ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_float), ('imag', ctypes.c_float)]
    _numpy_ty_ = complex64

class Complex128 (ctypes.Structure, ComplexMixin):
    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]
    _numpy_ty_ = complex128