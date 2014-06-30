"""
Utilities to simplify the boilerplate for native lowering.
"""

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


def iterator_impl(iterable_type, iterator_type):
    """
    Decorator a given class as implementing *iterator_type*
    (by providing an `iternext()` method).
    """

    def wrapper(cls):
        # These are unbound methods
        iternext = cls.iternext

        @iternext_impl
        def iternext_wrapper(context, builder, sig, args, result):
            (value,) = args
            iterobj = cls(context, builder, value)
            return iternext(iterobj, context, builder, result)

        builtin(implement('iternext', iterator_type)(iternext_wrapper))
        return cls

    return wrapper


class _IternextResult(object):
    """
    A result wrapper for iteration, passed by iternext_impl() into the
    wrapped function.
    """
    __slots__ = ('_context', '_builder', '_pairobj')

    def __init__(self, context, builder, pairobj):
        self._context = context
        self._builder = builder
        self._pairobj = pairobj

    def set_exhausted(self):
        """
        Mark the iterator as exhausted.
        """
        self._pairobj.second = self._builder.get_constant(types.boolean, False)

    def set_valid(self, is_valid=True):
        """
        Mark the iterator as valid according to *is_valid* (which must
        be either a Python boolean or a LLVM inst).
        """
        if is_valid in (False, True):
            is_valid = self._builder.get_constant(types.boolean, is_valid)
        self._pairobj.second = is_valid

    def yield_(self, value):
        """
        Mark the iterator as yielding the given *value* (a LLVM inst).
        """
        self._pairobj.first = value


def iternext_impl(func):
    """
    Wrap the given iternext() implementation so that it gets passed
    an _IternextResult() object easing the returning of the iternext()
    result pair.

    The wrapped function will be called with the following signature:
        (context, builder, sig, args, iternext_result)
    """
    from .builtins import make_pair

    def wrapper(context, builder, sig, args):
        pair_type = sig.return_type
        cls = make_pair(pair_type.first_type, pair_type.second_type)
        pairobj = cls(context, builder)
        func(context, builder, sig, args,
             _IternextResult(context, builder, pairobj))
        return pairobj._getvalue()
    return wrapper



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

