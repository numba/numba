
import inspect

from numba import types

# Exported symbols
from .typing.typeof import typeof_impl
from .typing.templates import infer, infer_getattr
from .targets.imputils import (
    lower_builtin, lower_getattr, lower_getattr_generic,
    lower_setattr, lower_setattr_generic, lower_cast)
from .datamodel import models, register_default as register_model
from .pythonapi import box, unbox, reflect, NativeValue


def type_callable(func):
    """
    Decorate a function as implementing typing for the callable *func*.
    *func* can be a callable object (probably a global) or a string
    denoting a built-in operation (such 'getitem' or '__array_wrap__')
    """
    from .typing.templates import CallableTemplate, infer, infer_global
    if not callable(func) and not isinstance(func, str):
        raise TypeError("`func` should be a function or string")
    try:
        func_name = func.__name__
    except AttributeError:
        func_name = str(func)

    def decorate(typing_func):
        def generic(self):
            return typing_func(self.context)

        name = "%s_CallableTemplate" % (func_name,)
        bases = (CallableTemplate,)
        class_dict = dict(key=func, generic=generic)
        template = type(name, bases, class_dict)
        infer(template)
        if hasattr(func, '__module__'):
            infer_global(func, types.Function(template))

    return decorate


def overload(func):
    """
    A decorator marking the decorated function as typing and implementing
    *func* in nopython mode.

    The decorated function will have the same formal parameters as *func*
    and be passed the Numba types of those parameters.  It should return
    a function implementing *func* for the given types.

    Here is an example implementing len() for tuple types::

        @overload(len)
        def tuple_len(seq):
            if isinstance(seq, types.BaseTuple):
                n = len(seq)
                def len_impl(seq):
                    return n
                return len_impl

    """
    from .typing.templates import make_overload_template, infer_global

    def decorate(overload_func):
        template = make_overload_template(func, overload_func)
        infer(template)
        if hasattr(func, '__module__'):
            infer_global(func, types.Function(template))
        return overload_func

    return decorate


def overload_attribute(typ, attr):
    """
    A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode.

    Here is an example implementing .nbytes for array types::

        @overload_attribute(types.Array, 'nbytes')
        def array_nbytes(arr):
            def get(arr):
                return arr.size * arr.itemsize
            return get
    """
    # TODO implement setters
    from .typing.templates import make_overload_attribute_template

    def decorate(overload_func):
        template = make_overload_attribute_template(typ, attr, overload_func)
        infer_getattr(template)
        return overload_func

    return decorate


def overload_method(typ, attr):
    """
    A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode.

    Here is an example implementing .take() for array types::

        @overload_method(types.Array, 'take')
        def array_take(arr, indices):
            if isinstance(indices, types.Array):
                def take_impl(arr, indices):
                    n = indices.shape[0]
                    res = np.empty(n, arr.dtype)
                    for i in range(n):
                        res[i] = arr[indices[i]]
                    return res
                return take_impl
    """
    from .typing.templates import make_overload_method_template

    def decorate(overload_func):
        template = make_overload_method_template(typ, attr, overload_func)
        infer_getattr(template)
        return overload_func

    return decorate


def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    """
    Make an automatic attribute wrapper exposing member named *struct_attr*
    as a read-only attribute named *python_attr*.
    The given *typeclass*'s model must be a StructModel subclass.
    """
    from .typing.templates import AttributeTemplate
    from .datamodel import default_manager
    from .datamodel.models import StructModel
    from .targets.imputils import impl_ret_borrowed
    from . import cgutils

    if not isinstance(typeclass, type) or not issubclass(typeclass, types.Type):
        raise TypeError("typeclass should be a Type subclass, got %s"
                        % (typeclass,))

    def get_attr_fe_type(typ):
        """
        Get the Numba type of member *struct_attr* in *typ*.
        """
        model = default_manager.lookup(typ)
        if not isinstance(model, StructModel):
            raise TypeError("make_struct_attribute_wrapper() needs a type "
                            "with a StructModel, but got %s" % (model,))
        return model.get_member_fe_type(struct_attr)

    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = typeclass

        def generic_resolve(self, typ, attr):
            if attr == python_attr:
                return get_attr_fe_type(typ)

    @lower_getattr(typeclass, python_attr)
    def struct_getattr_impl(context, builder, typ, val):
        val = cgutils.create_struct_proxy(typ)(context, builder, value=val)
        attrty = get_attr_fe_type(typ)
        attrval = getattr(val, struct_attr)
        return impl_ret_borrowed(context, builder, attrty, attrval)
