from __future__ import print_function, absolute_import, division
import functools
from numba.typing import signature
from numba import cgutils, types


def implement(func, *argtys):
    def wrapper(impl):
        @functools.wraps(impl)
        def res(context, builder, sig, args):
            ret = impl(context, builder, sig, args)
            return ret
        res.signature = signature(types.Any, *argtys)
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
    def imp(context, builder, sig, args):
        func = context.declare_function(cgutils.get_module(builder), fndesc)
        status, retval = context.call_function(builder, func, fndesc.argtypes,
                                               args)
        with cgutils.if_unlikely(builder, status.err):
            context.return_errcode_propagate(builder, status.code)
        return retval
    imp.signature = signature(fndesc.restype, *fndesc.argtypes)
    imp.key = func
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