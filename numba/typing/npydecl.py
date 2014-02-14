import numpy
from numba import types
from numba.typing.templates import (AttributeTemplate, AbstractTemplate,
                                    builtin_global, builtin, signature)


@builtin
class NumpyModuleAttribute(AttributeTemplate):
    key = types.Module(numpy)

    def resolve_absolute(self, mod):
        return types.Function(Numpy_absolute)

    def resolve_exp(self, mod):
        return types.Function(Numpy_exp)

    def resolve_sqrt(self, mod):
        return types.Function(Numpy_sqrt)

    def resolve_sin(self, mod):
        return types.Function(Numpy_sin)

    def resolve_cos(self, mod):
        return types.Function(Numpy_cos)

    def resolve_tan(self, mod):
        return types.Function(Numpy_tan)

    def resolve_add(self, mod):
        return types.Function(Numpy_add)

    def resolve_subtract(self, mod):
        return types.Function(Numpy_subtract)

    def resolve_multiply(self, mod):
        return types.Function(Numpy_multiply)

    def resolve_divide(self, mod):
        return types.Function(Numpy_divide)

    def resolve_negative(self, mod):
        return types.Function(Numpy_negative)

    def resolve_sqrt(self, mod):
        return types.Function(Numpy_sqrt)


class Numpy_unary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        nargs = len(args)
        if nargs == 2:
            [inp, out] = args
            if isinstance(inp, types.Array) and isinstance(out, types.Array):
                return signature(out, inp, out)
        elif nargs == 1:
            [inp] = args
            if inp in types.number_domain:
                return signature(types.float64, types.float64)


class Numpy_sqrt(Numpy_unary_ufunc):
    key = numpy.sqrt


class Numpy_absolute(Numpy_unary_ufunc):
    key = numpy.absolute


class Numpy_sin(Numpy_unary_ufunc):
    key = numpy.sin


class Numpy_cos(Numpy_unary_ufunc):
    key = numpy.cos


class Numpy_tan(Numpy_unary_ufunc):
    key = numpy.tan


class Numpy_exp(Numpy_unary_ufunc):
    key = numpy.exp


class Numpy_sqrt(Numpy_unary_ufunc):
    key = numpy.sqrt


class Numpy_negative(Numpy_unary_ufunc):
    key = numpy.negative


class Numpy_binary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [vx, wy, out] = args
        if (isinstance(vx, types.Array) and isinstance(wy, types.Array) and
                isinstance(out, types.Array)):
            if vx.dtype != wy.dtype or vx.dtype != out.dtype:
                # TODO handle differing dtypes
                return
            return signature(out, vx, wy, out)


class Numpy_add(Numpy_binary_ufunc):
    key = numpy.add


class Numpy_subtract(Numpy_binary_ufunc):
    key = numpy.subtract


class Numpy_multiply(Numpy_binary_ufunc):
    key = numpy.multiply


class Numpy_divide(Numpy_binary_ufunc):
    key = numpy.divide


builtin_global(numpy, types.Module(numpy))
builtin_global(numpy.absolute, types.Function(Numpy_absolute))
builtin_global(numpy.exp, types.Function(Numpy_exp))
builtin_global(numpy.sin, types.Function(Numpy_sin))
builtin_global(numpy.cos, types.Function(Numpy_cos))
builtin_global(numpy.tan, types.Function(Numpy_tan))
builtin_global(numpy.add, types.Function(Numpy_add))
builtin_global(numpy.subtract, types.Function(Numpy_subtract))
builtin_global(numpy.multiply, types.Function(Numpy_multiply))
builtin_global(numpy.divide, types.Function(Numpy_divide))
builtin_global(numpy.sqrt, types.Function(Numpy_sqrt))
builtin_global(numpy.negative, types.Function(Numpy_negative))


