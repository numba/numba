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

    def resolve_sinh(self, mod):
        return types.Function(Numpy_sinh)

    def resolve_cosh(self, mod):
        return types.Function(Numpy_cosh)

    def resolve_tanh(self, mod):
        return types.Function(Numpy_tanh)

    def resolve_arccos(self, mod):
        return types.Function(Numpy_arccos)

    def resolve_arcsin(self, mod):
        return types.Function(Numpy_arcsin)

    def resolve_arctan(self, mod):
        return types.Function(Numpy_arctan)

    def resolve_arctan2(self, mod):
        return types.Function(Numpy_arctan2)

    def resolve_arccosh(self, mod):
        return types.Function(Numpy_arccosh)

    def resolve_arcsinh(self, mod):
        return types.Function(Numpy_arcsinh)

    def resolve_arctanh(self, mod):
        return types.Function(Numpy_arctanh)

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

    def resolve_floor(self, mod):
        return types.Function(Numpy_floor)

    def resolve_ceil(self, mod):
        return types.Function(Numpy_ceil)

    def resolve_trunc(self, mod):
        return types.Function(Numpy_trunc)

    def resolve_sign(self, mod):
        return types.Function(Numpy_sign)

class Numpy_unary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        nargs = len(args)
        if nargs == 2:
            [inp, out] = args
            if isinstance(inp, types.Array) and isinstance(out, types.Array):
                return signature(out, inp, out)
            elif inp in types.number_domain and isinstance(out, types.Array):
                return signature(out, inp, out)
        elif nargs == 1:
            [inp] = args
            if inp in types.number_domain:
                if hasattr(self, "scalar_out_type"):
                    return signature(self.scalar_out_type, inp)
                else:
                    return signature(inp, inp)


class Numpy_sqrt(Numpy_unary_ufunc):
    key = numpy.sqrt
    scalar_out_type = types.float64


class Numpy_absolute(Numpy_unary_ufunc):
    key = numpy.absolute


class Numpy_sin(Numpy_unary_ufunc):
    key = numpy.sin
class Numpy_cos(Numpy_unary_ufunc):
    key = numpy.cos
class Numpy_tan(Numpy_unary_ufunc):
    key = numpy.tan


class Numpy_sinh(Numpy_unary_ufunc):
    key = numpy.sinh
class Numpy_cosh(Numpy_unary_ufunc):
    key = numpy.cosh
class Numpy_tanh(Numpy_unary_ufunc):
    key = numpy.tanh


class Numpy_arccos(Numpy_unary_ufunc):
    key = numpy.arccos
class Numpy_arcsin(Numpy_unary_ufunc):
    key = numpy.arcsin
class Numpy_arctan(Numpy_unary_ufunc):
    key = numpy.arctan


class Numpy_arccosh(Numpy_unary_ufunc):
    key = numpy.arccosh
class Numpy_arcsinh(Numpy_unary_ufunc):
    key = numpy.arcsinh
class Numpy_arctanh(Numpy_unary_ufunc):
    key = numpy.arctanh


class Numpy_exp(Numpy_unary_ufunc):
    key = numpy.exp


class Numpy_negative(Numpy_unary_ufunc):
    key = numpy.negative


class Numpy_floor(Numpy_unary_ufunc):
    key = numpy.floor


class Numpy_ceil(Numpy_unary_ufunc):
    key = numpy.ceil


class Numpy_trunc(Numpy_unary_ufunc):
    key = numpy.trunc


class Numpy_sign(Numpy_unary_ufunc):
    key = numpy.sign


class Numpy_binary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        nargs = len(args)
        if nargs == 3:
            [inp1, inp2, out] = args
            if isinstance(inp1, types.Array) and \
                    isinstance(inp2, types.Array) and \
                    isinstance(out, types.Array):
                return signature(out, inp1, inp2, out)
            elif inp1 in types.number_domain and \
                    inp2 in types.number_domain and \
                    isinstance(out, types.Array):
                return signature(out, inp1, inp2, out)
        elif nargs == 2:
            [inp1, inp2] = args
            if inp1 in types.number_domain and inp2 in types.number_domain:
                if hasattr(self, "scalar_out_type"):
                    return signature(self.scalar_out_type, inp1, inp2)
                else:
                    return signature(inp1, inp1, inp2)


class Numpy_add(Numpy_binary_ufunc):
    key = numpy.add


class Numpy_subtract(Numpy_binary_ufunc):
    key = numpy.subtract


class Numpy_multiply(Numpy_binary_ufunc):
    key = numpy.multiply


class Numpy_divide(Numpy_binary_ufunc):
    key = numpy.divide


class Numpy_arctan2(Numpy_binary_ufunc):
    key = numpy.arctan2


builtin_global(numpy, types.Module(numpy))
builtin_global(numpy.absolute, types.Function(Numpy_absolute))
builtin_global(numpy.exp, types.Function(Numpy_exp))
builtin_global(numpy.sin, types.Function(Numpy_sin))
builtin_global(numpy.cos, types.Function(Numpy_cos))
builtin_global(numpy.tan, types.Function(Numpy_tan))
builtin_global(numpy.sinh, types.Function(Numpy_sinh))
builtin_global(numpy.cosh, types.Function(Numpy_cosh))
builtin_global(numpy.tanh, types.Function(Numpy_tanh))
builtin_global(numpy.arccos, types.Function(Numpy_arccos))
builtin_global(numpy.arcsin, types.Function(Numpy_arcsin))
builtin_global(numpy.arctan, types.Function(Numpy_arctan))
builtin_global(numpy.arctan2, types.Function(Numpy_arctan2))
builtin_global(numpy.arccosh, types.Function(Numpy_arccosh))
builtin_global(numpy.arcsinh, types.Function(Numpy_arcsinh))
builtin_global(numpy.arctanh, types.Function(Numpy_arctanh))
builtin_global(numpy.add, types.Function(Numpy_add))
builtin_global(numpy.subtract, types.Function(Numpy_subtract))
builtin_global(numpy.multiply, types.Function(Numpy_multiply))
builtin_global(numpy.divide, types.Function(Numpy_divide))
builtin_global(numpy.sqrt, types.Function(Numpy_sqrt))
builtin_global(numpy.negative, types.Function(Numpy_negative))
builtin_global(numpy.floor, types.Function(Numpy_floor))
builtin_global(numpy.ceil, types.Function(Numpy_ceil))
builtin_global(numpy.trunc, types.Function(Numpy_trunc))
builtin_global(numpy.sign, types.Function(Numpy_sign))


