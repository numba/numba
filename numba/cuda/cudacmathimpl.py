import cmath
from numba.cuda.extending import overload
from numba.cpython import cmathimpl

ov_funcs = [
    (cmath.acosh, cmathimpl.impl_cmath_acosh),
    (cmath.cosh, cmathimpl.impl_cmath_cosh),
    (cmath.log10, cmathimpl.impl_cmath_log10),
    (cmath.tanh, cmathimpl.impl_cmath_tanh),
    (cmath.sinh, cmathimpl.impl_cmath_sinh),
    (cmath.polar, cmathimpl.polar_impl),
    (cmath.phase, cmathimpl.phase_impl),
    (cmath.rect, cmathimpl.impl_cmath_rect)
]

for func, impl in ov_funcs:
    overload(func)(impl)
