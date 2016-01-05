
import inspect

from numba import types

# Exported symbols
from .typing.typeof import typeof_impl
from .targets.imputils import builtin, builtin_cast, implement, impl_attribute
from .datamodel import models, register_default as register_model
from .pythonapi import box, unbox, reflect, NativeValue


def type_callable(func):
    """
    Decorate a function as implementing typing for the callable *func*.
    *func* can be a callable object (probably a global) or a string
    denoting a built-in operation (such 'getitem' or '__array_wrap__')
    """
    from .typing.templates import CallableTemplate, builtin, builtin_global
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
        builtin(template)
        if hasattr(func, '__module__'):
            builtin_global(func, types.Function(template))

    return decorate


def overload(func):
    """
    TODO docstring
    """
    # XXX Should overload() return a jitted wrapper calling the
    # function?  This way it would also be usable from pure Python
    # code, like a regular jitted function
    from .typing.templates import make_overload_template, builtin_global

    def decorate(overload_func):
        template = make_overload_template(func, overload_func)
        ty = types.Function(template)
        if hasattr(func, '__module__'):
            builtin_global(func, ty)
        return overload_func

    return decorate


def overload_attribute(typ, attr):
    """
    TODO docstring
    """
    from .typing.templates import make_overload_attribute_template, builtin_attr

    def decorate(overload_func):
        template = make_overload_attribute_template(typ, attr, overload_func)
        builtin_attr(template)
        return overload_func

    return decorate


def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    """
    Make an automatic attribute wrapper exposing member named *struct_attr*
    as a read-only attribute named *python_attr*.
    The given *typeclass*'s model must be a StructModel subclass.
    """
    # XXX should this work for setters as well?
    from .typing.templates import builtin_attr, AttributeTemplate
    from .datamodel import default_manager
    from .datamodel.models import StructModel
    from .targets.imputils import (builtin_attr as target_attr,
                                   impl_attribute, impl_ret_borrowed)
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

    @builtin_attr
    class StructAttribute(AttributeTemplate):
        key = typeclass

        def generic_resolve(self, typ, attr):
            if attr == python_attr:
                return get_attr_fe_type(typ)

    @target_attr
    @impl_attribute(typeclass, python_attr)
    def struct_getattr_impl(context, builder, typ, val):
        val = cgutils.create_struct_proxy(typ)(context, builder, value=val)
        attrty = get_attr_fe_type(typ)
        attrval = getattr(val, struct_attr)
        return impl_ret_borrowed(context, builder, attrty, attrval)
