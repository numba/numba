from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase

import numpy as np
from numba import cuda, njit, types

from numba.core import cgutils
from numba.core.extending import (lower_builtin, make_attribute_wrapper,
                                  models, register_model, type_callable,
                                  typeof_impl)
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr


class Interval:
    """
    A half-open interval on the real number line.
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


class IntervalType(types.Type):
    def __init__(self):
        super().__init__(name='Interval')


interval_type = IntervalType()


@typeof_impl.register(Interval)
def typeof_interval(val, c):
    return interval_type


@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return interval_type
    return typer


@register_model(IntervalType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('lo', types.float64),
            ('hi', types.float64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IntervalType, 'lo', 'lo')
make_attribute_wrapper(IntervalType, 'hi', 'hi')


@lower_builtin(Interval, types.Float, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    return interval._getvalue()


@cuda_registry.register_attr
class Interval_attrs(AttributeTemplate):
    key = IntervalType

    def resolve_width(self, mod):
        return types.float64


@cuda_lower_attr(IntervalType, 'width')
def cuda_Interval_width(context, builder, sig, arg):
    lo = builder.extract_value(arg, 0)
    hi = builder.extract_value(arg, 1)
    return builder.fsub(hi, lo)


@njit
def inside_interval(interval, x):
    """Tests attribute access"""
    return interval.lo <= x < interval.hi


@njit
def interval_width(interval):
    """Tests property access"""
    return interval.width


@njit
def sum_intervals(i, j):
    """Tests the Interval constructor"""
    return Interval(i.lo + j.lo, i.hi + j.hi)


@skip_on_cudasim('Dispatcher objects not used in the simulator')
class TestExtending(CUDATestCase):
    def test_simple(self):
        @cuda.jit
        def kernel(arr):
            x = Interval(1.0, 3.0)
            arr[0] = x.hi + x.lo
            arr[1] = x.width
            arr[2] = inside_interval(x, 2.5)
            arr[3] = inside_interval(x, 3.5)
            arr[4] = interval_width(x)

            y = Interval(7.5, 9.0)
            z = sum_intervals(x, y)
            arr[5] = z.lo
            arr[6] = z.hi

        out = np.zeros(7)

        kernel[1, 1](out)

        np.testing.assert_allclose(out, [ 4,   2,   1,   0,   2,   8.5, 12 ])


if __name__ == '__main__':
    unittest.main()
