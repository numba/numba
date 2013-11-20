from ctypes import CDLL, Structure, POINTER, c_longlong, c_uint, c_int, \
    c_ubyte, c_char_p, byref, pointer
from numba import autojit, jit, void, c_string_type, int_, typeof, object_, void
import cdecimal
import random


class mpd_context(Structure):
    
    _fields_ = [('prec', c_longlong),
                ('emax', c_longlong),
                ('emin', c_longlong),
                ('trap', c_uint),
                ('status', c_uint),
                ('newtrap', c_uint),
                ('round', c_int),
                ('clamp', c_int),
                ('allcr', c_int)]

class mpd_t(Structure):

    _fields_ = [('flags', c_ubyte),
                ('exp', c_longlong),
                ('digits', c_longlong),
                ('len', c_longlong),
                ('alloc', c_longlong),
                ('data', POINTER(c_uint))]


dll = CDLL('/usr/local/lib/libmpdec.so.2.3')

dll.mpd_new.argtypes = []
dll.mpd_new.restype = POINTER(mpd_t)

dll.mpd_set_string.argtypes = [POINTER(mpd_t), c_char_p, POINTER(mpd_context)]
dll.mpd_set_string.restype = None

dll.mpd_to_sci.argtypes = [POINTER(mpd_t), c_int]
dll.mpd_to_sci.restype = c_char_p

dll.mpd_add.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_add.restype = None

dll.mpd_sub.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_sub.restype = None

dll.mpd_mul.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_mul.restype = None

dll.mpd_div.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_div.restype = None

mpd_new_func = dll.mpd_new
mpd_set_string_func = dll.mpd_set_string
mpd_to_sci_func = dll.mpd_to_sci
mpd_add_func = dll.mpd_add
mpd_sub_func = dll.mpd_sub
mpd_mul_func = dll.mpd_mul
mpd_div_func = dll.mpd_div


context = mpd_context()
context_ref = pointer(context)
dll.mpd_init(byref(context))


@jit
class NumbaDecimal(object):
    
    @void(c_string_type)
    def __init__(self, value):
        
        with nopython:
            self.mpd = mpd_new_func()
            mpd_set_string_func(self.mpd, value, context_ref)

    @c_string_type()
    def __repr__(self):
       
        with nopython:
            result = mpd_to_sci_func(self.mpd, 0)
        return result


@jit(NumbaDecimal.exttype(NumbaDecimal.exttype, NumbaDecimal.exttype))
def add(left, right):
    with nopython:
        mpd_result = mpd_new_func()
        mpd_add_func(mpd_result, left.mpd, right.mpd, context_ref)
    return NumbaDecimal(mpd_to_sci_func(mpd_result, 0))

@jit(NumbaDecimal.exttype(NumbaDecimal.exttype, NumbaDecimal.exttype))
def sub(left, right):
    with nopython:
        mpd_result = mpd_new_func()
        mpd_sub_func(mpd_result, left.mpd, right.mpd, context_ref)
    return NumbaDecimal(mpd_to_sci_func(mpd_result, 0))

@jit(NumbaDecimal.exttype(NumbaDecimal.exttype, NumbaDecimal.exttype))
def mul(left, right):
    with nopython:
        mpd_result = mpd_new_func()
        mpd_mul_func(mpd_result, left.mpd, right.mpd, context_ref)
    return NumbaDecimal(mpd_to_sci_func(mpd_result, 0))

@jit(NumbaDecimal.exttype(NumbaDecimal.exttype, NumbaDecimal.exttype))
def div(left, right):
    with nopython:
        mpd_result = mpd_new_func()
        mpd_div_func(mpd_result, left.mpd, right.mpd, context_ref)
    return NumbaDecimal(mpd_to_sci_func(mpd_result, 0))


@autojit
def numba_test(num1, num2):
    x = NumbaDecimal(num1)
    y = NumbaDecimal(num2)
    return mul(x, y)

@autojit
def python_test(num1, num2):
    x = cdecimal.Decimal(num1)
    y = cdecimal.Decimal(num2)
    return x * y

def benchmark_numba():
    numba_test(str(random.random()), str(random.random()))

def benchmark_python():
    python_test(str(random.random()), str(random.random()))

import timeit

timer = timeit.Timer("benchmark_numba()", "from __main__ import benchmark_numba")
print 'Numba extension type times:', timer.repeat(repeat=3, number=100000)

timer = timeit.Timer("benchmark_python()", "from __main__ import benchmark_python")
print 'Python cdecimal times:', timer.repeat(repeat=3, number=100000)
