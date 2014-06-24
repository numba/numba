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
        def res(context, builder, typ, value, attr):
            ret = impl(context, builder, typ, value)
            return ret

        res.return_type = rtype
        res.key = (ty, attr)
        return res

    return wrapper


def impl_attribute_generic(ty):
    def wrapper(impl):
        @functools.wraps(impl)
        def res(context, builder, typ, value, attr):
            ret = impl(context, builder, typ, value, attr)
            return ret

        res.key = (ty, None)
        return res

    return wrapper


def user_function(func, fndesc, libs):
    def imp(context, builder, sig, args):
        func = context.declare_function(cgutils.get_module(builder), fndesc)
        status, retval = context.call_function(builder, func, fndesc.restype,
                                               fndesc.argtypes, args)
        with cgutils.if_unlikely(builder, status.err):
            context.return_errcode_propagate(builder, status.code)
        return retval

    imp.signature = signature(fndesc.restype, *fndesc.argtypes)
    imp.key = func
    imp.libs = tuple(libs)
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


class Registry(object):
    def __init__(self):
        self.functions = []
        self.attributes = []

    def register(self, item):
        self.functions.append(item)
        return item

    def register_attr(self, item):
        self.attributes.append(item)
        return item


builtin_registry = Registry()
builtin = builtin_registry.register
builtin_attr = builtin_registry.register_attr


def auto_attributes(module, typemap, register):
    """
    Automatically generate attribute function for a module

    Args
    -----
    module
        python module

    typemap
        A mapping of python type to numba type

    register
        a function to register a attribute implementation
    """

    def attr_impl(valty, val):
        def core(context, builder, typ, value):
            # builder, typ, value are unused here
            return context.get_constant(valty, val)

        return core

    nbmod = types.Module(module)

    for name in dir(module):
        val = getattr(module, name)
        vty = type(val)

        if vty in typemap:
            valty = typemap[vty]   # get numba type

            register(impl_attribute(nbmod, name, valty)(
                attr_impl(valty, val)))
