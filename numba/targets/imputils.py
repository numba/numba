import functools
from numba.typing import signature
from numba import cgutils, types


def implement(func, return_type, *args):
    def wrapper(impl):
        @functools.wraps(impl)
        def res(context, builder, tys, args):
            ret = impl(context, builder, tys, args)
            return ret
        res.signature = signature(return_type, *args)
        res.key = func
        return res
    return wrapper


def impl_attribute(ty, attr, rtype):
    def wrapper(impl):
        @functools.wraps(impl)
        def res(context, builder, typ, value):
            ret = impl(context, builder, typ, value)
            return ret
        res.return_type = rtype
        res.key = (ty, attr)
        return res
    return wrapper


def user_function(func, fndesc):
    @implement(func, fndesc.restype, *fndesc.argtypes)
    def imp(context, builder, tys, args):
        func = context.declare_function(cgutils.get_module(builder), fndesc)
        status, retval = context.call_function(builder, func, args)
        # TODO handling error
        return retval
    return imp


def python_attr_impl(cls, attr, atyp):
    @impl_attribute(cls, attr, atyp)
    def imp(context, builder, typ, value):
        api = context.get_python_api(builder)
        aval = api.object_getattr_string(value, attr)
        with cgutils.ifthen(builder, cgutils.is_null(builder, aval)):
            context.return_exc(builder)

        if isinstance(atyp, types.Method):
            return aval
        else:
            nativevalue = api.to_native_value(aval, atyp)
            api.decref(aval)
            return nativevalue
    return imp


#-------------------------------------------------------------------------------

BUILTINS = []
BUILTIN_ATTRS = []


def builtin(impl):
    BUILTINS.append(impl)
    return impl


def builtin_attr(impl):
    BUILTIN_ATTRS.append(impl)
    return impl