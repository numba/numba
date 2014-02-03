# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *

class ComplexSupportMixin(object):
    "Support for complex numbers"

    def _generate_complex_op(self, op, arg1, arg2):
        (r1, i1), (r2, i2) = self._extract(arg1), self._extract(arg2)
        real, imag = op(r1, i1, r2, i2)
        return self._create_complex(real, imag)

    def _extract(self, value):
        "Extract the real and imaginary parts of the complex value"
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1))

    def _create_complex(self, real, imag):
        assert real.type == imag.type, (str(real.type), str(imag.type))
        complex = llvm.core.Constant.undef(llvm.core.Type.struct([real.type,
                                                                  real.type]))
        complex = self.builder.insert_value(complex, real, 0)
        complex = self.builder.insert_value(complex, imag, 1)
        return complex

    def _promote_complex(self, src_type, dst_type, value):
        "Promote a complex value to value with a larger or smaller complex type"
        real, imag = self._extract(value)

        if dst_type.is_complex:
            dst_type = dst_type.base_type
        dst_ltype = dst_type.to_llvm(self.context)

        real = self.caster.cast(real, dst_ltype)
        imag = self.caster.cast(imag, dst_ltype)
        return self._create_complex(real, imag)

    def _complex_add(self, arg1r, arg1i, arg2r, arg2i):
        return (self.builder.fadd(arg1r, arg2r),
                self.builder.fadd(arg1i, arg2i))

    def _complex_sub(self, arg1r, arg1i, arg2r, arg2i):
        return (self.builder.fsub(arg1r, arg2r),
                self.builder.fsub(arg1i, arg2i))

    def _complex_mul(self, arg1r, arg1i, arg2r, arg2i):
        return (self.builder.fsub(self.builder.fmul(arg1r, arg2r),
                                  self.builder.fmul(arg1i, arg2i)),
                self.builder.fadd(self.builder.fmul(arg1i, arg2r),
                                  self.builder.fmul(arg1r, arg2i)))

    def _complex_div(self, arg1r, arg1i, arg2r, arg2i):
        divisor = self.builder.fadd(self.builder.fmul(arg2r, arg2r),
                                    self.builder.fmul(arg2i, arg2i))
        return (self.builder.fdiv(
            self.builder.fadd(self.builder.fmul(arg1r, arg2r),
                              self.builder.fmul(arg1i, arg2i)),
            divisor),
                self.builder.fdiv(
                    self.builder.fsub(self.builder.fmul(arg1i, arg2r),
                                      self.builder.fmul(arg1r, arg2i)),
                    divisor))

    def _complex_floordiv(self, arg1r, arg1i, arg2r, arg2i):
        real, imag = self._complex_div(arg1r, arg1i, arg2r, arg2i)
        long_type = long_.to_llvm(self.context)
        real = self.caster.cast(real, long_type, unsigned=False)
        imag = self.caster.cast(imag, long_type, unsigned=False)
        real = self.caster.cast(real, arg1r.type, unsigned=False)
        imag = self.caster.cast(imag, arg1r.type, unsigned=False)
        return real, imag

