import numpy
from numba import types
from numba.typing.templates import (AttributeTemplate, AbstractTemplate,
                                    Registry, signature)

registry = Registry()
builtin_global = registry.register_global
builtin_attr = registry.register_attr


@builtin_attr
class NumpyModuleAttribute(AttributeTemplate):
    # note: many unary ufuncs are added later on, using setattr
    key = types.Module(numpy)

    def resolve_arctan2(self, mod):
        return types.Function(Numpy_arctan2)

    def resolve_add(self, mod):
        return types.Function(Numpy_add)

    def resolve_subtract(self, mod):
        return types.Function(Numpy_subtract)

    def resolve_multiply(self, mod):
        return types.Function(Numpy_multiply)

    def resolve_divide(self, mod):
        return types.Function(Numpy_divide)



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
        scalar_out_type = types.float64

    # Add the resolve method to NumpyModuleAttribute
    setattr(NumpyModuleAttribute, "resolve_"+name, lambda s, m: types.Function(typing_class))
    builtin_global(the_key, types.Function(typing_class))


# list of unary ufuncs to register
_autoregister_unary_ufuncs = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "exp", "exp2", "expm1",
    "log", "log2", "log10", "log1p",
    "absolute", "negative", "floor", "ceil", "trunc", "sign",
    "sqrt",
    "deg2rad", "rad2deg"]
for func in _autoregister_unary_ufuncs:
    _numpy_unary_ufunc(func)
del(_autoregister_unary_ufuncs)



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
builtin_global(numpy.arctan2, types.Function(Numpy_arctan2))
builtin_global(numpy.add, types.Function(Numpy_add))
builtin_global(numpy.subtract, types.Function(Numpy_subtract))
builtin_global(numpy.multiply, types.Function(Numpy_multiply))
builtin_global(numpy.divide, types.Function(Numpy_divide))


