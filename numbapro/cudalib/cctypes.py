from __future__ import absolute_import, print_function, division
from ctypes import c_float, c_double, Structure


class c_complex(Structure):
    _fields_ = [('real', c_float), ('imag', c_float)]

    def __init__(self, real=0, imag=0):
        if isinstance(real, complex):
            real, imag = real.real, real.imag
        super(c_complex, self).__init__(real, imag)


class c_double_complex(Structure):
    _fields_ = [('real', c_double), ('imag', c_double)]

    def __init__(self, real=0, imag=0):
        if isinstance(real, complex):
            real, imag = real.real, real.imag
        super(c_double_complex, self).__init__(real, imag)
