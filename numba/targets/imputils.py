"""
Utilities to simplify the boilerplate for native lowering.
"""

from __future__ import print_function, absolute_import, division

import collections
import functools

from .. import typing, cgutils, types

def implement(func, *argtys):
    def wrapper(impl):
        try:
            sigs = impl.function_signatures
        except AttributeError:
            sigs = impl.function_signatures = []
        sigs.append((func, typing.signature(types.Any, *argtys)))
        return impl

    return wrapper


def impl_attribute(ty, attr, rtype=None):
    def wrapper(impl):
        real_impl = impl
        while hasattr(real_impl, "__wrapped__"):
            real_impl = real_impl.__wrapped__

        @functools.wraps(impl)
        def res(context, builder, typ, value, attr):
            ret = real_impl(context, builder, typ, value)
            return ret

        if rtype is None:
            res.signature = typing.signature(types.Any, ty)
        else:
            res.signature = typing.signature(rtype, ty)
        res.attr = attr
        res.__wrapped__ = impl
        return res

    return wrapper


def impl_attribute_generic(ty):
    def wrapper(impl):
        real_impl = impl
        while hasattr(real_impl, "__wrapped__"):
            real_impl = real_impl.__wrapped__

        @functools.wraps(impl)
        def res(context, builder, typ, value, attr):
            ret = real_impl(context, builder, typ, value, attr)
            return ret

        res.signature = typing.signature(types.Any, ty)
        res.attr = None
        res.__wrapped__ = impl
        return res

    return wrapper


def user_function(fndesc, libs):
    """
    A wrapper inserting code calling Numba-compiled *fndesc*.
    """

    def imp(context, builder, sig, args):
        func = context.declare_function(cgutils.get_module(builder), fndesc)
        # env=None assumes this is a nopython function
        status, retval = context.call_conv.call_function(
            builder, func, fndesc.restype, fndesc.argtypes, args, env=None)
        with cgutils.if_unlikely(builder, status.is_error):
            context.call_conv.return_status_propagate(builder, status)
        return retval

    imp.signature = typing.signature(fndesc.restype, *fndesc.argtypes)
    imp.libs = tuple(libs)
    return imp


def user_generator(gendesc, libs):
    """
    A wrapper inserting code calling Numba-compiled *gendesc*.
    """

    def imp(context, builder, sig, args):
        func = context.declare_function(cgutils.get_module(builder), gendesc)
        # env=None assumes this is a nopython function
        status, retval = context.call_conv.call_function(
            builder, func, gendesc.restype, gendesc.argtypes, args, env=None)
        # Return raw status for caller to process StopIteration
        return status, retval

    imp.libs = tuple(libs)
    return imp


def python_attr_impl(cls, attr, atyp):
    @impl_attribute(cls, attr, atyp)
    def imp(context, builder, typ, value):
        api = context.get_python_api(builder)
        aval = api.object_getattr_string(value, attr)
        with cgutils.ifthen(builder, cgutils.is_null(builder, aval)):
            context.call_conv.return_exc(builder)

        if isinstance(atyp, types.Method):
            return aval
        else:
            native = api.to_native_value(aval, atyp)
            assert native.cleanup is None
            api.decref(aval)
            return native.value

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
        self._pairobj.second = self._context.get_constant(types.boolean, False)

    def set_valid(self, is_valid=True):
        """
        Mark the iterator as valid according to *is_valid* (which must
        be either a Python boolean or a LLVM inst).
        """
        if is_valid in (False, True):
            is_valid = self._context.get_constant(types.boolean, is_valid)
        self._pairobj.second = is_valid

    def yield_(self, value):
        """
        Mark the iterator as yielding the given *value* (a LLVM inst).
        """
        self._pairobj.first = value

    def is_valid(self):
        """
        Return whether the iterator is marked valid.
        """
        return self._context.get_argument_value(self._builder,
                                                types.boolean,
                                                self._pairobj.second)

    def yielded_value(self):
        """
        Return the iterator's yielded value, if any.
        """
        return self._pairobj.first


def iternext_impl(func):
    """
    Wrap the given iternext() implementation so that it gets passed
    an _IternextResult() object easing the returning of the iternext()
    result pair.

    The wrapped function will be called with the following signature:
        (context, builder, sig, args, iternext_result)
    """

    def wrapper(context, builder, sig, args):
        pair_type = sig.return_type
        cls = context.make_pair(pair_type.first_type, pair_type.second_type)
        pairobj = cls(context, builder)
        func(context, builder, sig, args,
             _IternextResult(context, builder, pairobj))
        return pairobj._getvalue()
    return wrapper


def call_getiter(context, builder, iterable_type, val):
    """
    Call the `getiter()` implementation for the given *iterable_type*
    of value *val*, and return the corresponding LLVM inst.
    """
    getiter_sig = typing.signature(iterable_type.iterator_type, iterable_type)
    getiter_impl = context.get_function('getiter', getiter_sig)
    return getiter_impl(builder, (val,))


def call_iternext(context, builder, iterator_type, val):
    """
    Call the `iternext()` implementation for the given *iterator_type*
    of value *val*, and return a convenience _IternextResult() object
    reflecting the results.
    """
    itemty = iterator_type.yield_type
    pair_type = types.Pair(itemty, types.boolean)
    paircls = context.make_pair(pair_type.first_type, pair_type.second_type)
    iternext_sig = typing.signature(pair_type, iterator_type)
    iternext_impl = context.get_function('iternext', iternext_sig)
    val = iternext_impl(builder, (val,))
    return _IternextResult(context, builder, paircls(context, builder, val))


class Registry(object):
    def __init__(self):
        self.functions = []
        self.attributes = []

    def register(self, impl):
        sigs = impl.function_signatures
        impl.function_signatures = []
        self.functions.append((impl, sigs))
        return impl

    def register_attr(self, item):
        curr_item = item
        while hasattr(curr_item, '__wrapped__'):
            self.attributes.append(curr_item)
            curr_item = curr_item.__wrapped__
        return item


builtin_registry = Registry()
builtin = builtin_registry.register
builtin_attr = builtin_registry.register_attr


class _StructRegistry(object):
    """
    A registry of factories of cgutils.Structure classes.
    """

    def __init__(self):
        self.impls = {}

    def register(self, type_class):
        """
        Register a Structure factory function for the given *type_class*
        (i.e. a subclass of numba.types.Type).
        """
        assert issubclass(type_class, types.Type)
        def decorator(func):
            self.impls[type_class] = func
            return func
        return decorator

    def match(self, typ):
        """
        Return the Structure factory function for the given Numba type
        instance *typ*.
        """
        return self.impls[typ.__class__]


struct_registry = _StructRegistry()
struct_factory = struct_registry.register


class _TypeRegistry(object):

    def __init__(self):
        self.factories = {}

    def register(self, type_class):
        """
        Register a LLVM type factory function for the given *type_class*
        (i.e. a subclass of numba.types.Type).
        """
        assert issubclass(type_class, types.Type)
        def decorator(func):
            self.factories[type_class] = func
            return func
        return decorator

    def match(self, typ):
        """
        Return the LLVM type factory function for the given Numba type
        instance *typ*.
        """
        return self.factories[typ.__class__]


type_registry = _TypeRegistry()
type_factory = type_registry.register
