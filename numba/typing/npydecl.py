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

    def resolve_exp2(self, mod):
        return types.Function(Numpy_exp2)

    def resolve_expm1(self, mod):
        return types.Function(Numpy_expm1)

    def resolve_log(self, mod):
        return types.Function(Numpy_log)

    def resolve_log2(self, mod):
        return types.Function(Numpy_log2)

    def resolve_log10(self, mod):
        return types.Function(Numpy_log10)

    def resolve_log1p(self, mod):
        return types.Function(Numpy_log1p)

    def resolve_sqrt(self, mod):
        return types.Function(Numpy_sqrt)

    def resolve_arctan2(self, mod):
        return types.Function(Numpy_arctan2)

    def resolve_deg2rad(self, mod):
        return types.Function(Numpy_deg2rad)

    def resolve_rad2deg(self, mod):
        return types.Function(Numpy_rad2deg)

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


def _numpy_unary_ufunc(name):
    the_key = eval("numpy."+name) # obtain the appropriate symbol for the key.
    class typing_class(Numpy_unary_ufunc):
        key = the_key

    # Add the resolve method to NumpyModuleAttribute
    setattr(NumpyModuleAttribute, "resolve_"+name, lambda s, m: types.Function(typing_class))
    builtin_global(the_key, types.Function(typing_class))


#register for funcs
_autoregister_unary_ufuncs = [ 
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "arcsinh", "arccosh", "arctanh" ]
for func in _autoregister_unary_ufuncs:
    _numpy_unary_ufunc(func)
del(_autoregister_unary_ufuncs)

class Numpy_sqrt(Numpy_unary_ufunc):
    key = numpy.sqrt
    scalar_out_type = types.float64


class Numpy_absolute(Numpy_unary_ufunc):
    key = numpy.absolute


class Numpy_exp(Numpy_unary_ufunc):
    key = numpy.exp
class Numpy_exp2(Numpy_unary_ufunc):
    key = numpy.exp2
class Numpy_expm1(Numpy_unary_ufunc):
    key = numpy.expm1

class Numpy_log(Numpy_unary_ufunc):
    key = numpy.log
class Numpy_log2(Numpy_unary_ufunc):
    key = numpy.log2
class Numpy_log10(Numpy_unary_ufunc):
    key = numpy.log10
class Numpy_log1p(Numpy_unary_ufunc):
    key = numpy.log1p

class Numpy_deg2rad(Numpy_unary_ufunc):
    key = numpy.deg2rad
class Numpy_rad2deg(Numpy_unary_ufunc):
    key = numpy.rad2deg

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
            if isinstance(out, types.Array) and \
                    (isinstance(inp1, types.Array) or inp1 in types.number_domain) or \
                    (isinstance(inp2, types.Array) or inp2 in types.number_domain):
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
builtin_global(numpy.exp2, types.Function(Numpy_exp2))
builtin_global(numpy.expm1, types.Function(Numpy_expm1))
builtin_global(numpy.log, types.Function(Numpy_log))
builtin_global(numpy.log2, types.Function(Numpy_log2))
builtin_global(numpy.log10, types.Function(Numpy_log10))
builtin_global(numpy.log1p, types.Function(Numpy_log1p))
builtin_global(numpy.arctan2, types.Function(Numpy_arctan2))
builtin_global(numpy.deg2rad, types.Function(Numpy_deg2rad))
builtin_global(numpy.rad2deg, types.Function(Numpy_rad2deg))
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


