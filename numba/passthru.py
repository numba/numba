###############################################################################################################
# Tools to ferry arbitrary Python objects through nopython mode
#
# This has two typical use-cases:
#   1. ferrying data structures not currently supported by Numba into `objmode` blocks via `PassThruContainer`
#   2. creating extension types that are simplified representations of the Python class and keep a link to the
#      Python object
#
# Example for the latter case (``test.test_passthru.PassThruComplex`` for a working example):
#
#   >>> class BloatedPythonClass(object):
#   >>>     def __init__(numba_supported_attribute, not_numba_supported_attribute):
#   >>>         self.numba_supported_attribute = numba_supported_attribute
#   >>>         self.not_numba_supported_attribute = not_numba_supported_attribute
#   >>>
#   >>> BloatedPythonClassModel(models.StructModel):
#   >>>     def __init__(self, dmm, fe_typ):
#   >>>         members = [
#   >>>             ('parent', PassThruType('OpaqueBloatedPythonClass')),
#   >>>             ('numba_supported_attribute', NumbaSupportedAttributeType),
#   >>>         ]
#   >>>         super(BloatedPythonClassModel, self).__init__(dmm, fe_typ, members)
#   >>>
#
# The full Python object is passed through *nopython-mode* in the ``parent`` attribute (obviously nothing can
# be done with it there), any Numba supported attributes could be made available for *nopython-mode*. Since the
# boxer has access to the ``parent`` object you might even reflect changes done to the attributes available in
# *nopython-mode* back onto the parent (though there are good reason to be careful with this, see history of
# list/set support https://github.com/numba/numba/issues/3546).
###############################################################################################################

from numba import cgutils, types
from numba.datamodel import models
from numba.extending import make_attribute_wrapper, overload, overload_method, register_model, type_callable
from numba.pythonapi import NativeValue, unbox, box
from numba.six import PY3
from numba.targets.imputils import lower_builtin
from numba.typing.typeof import typeof_impl

from operator import is_, eq
from llvmlite.llvmpy.core import Constant


__all__ = ['PassThruContainer', 'pass_thru_container_type']


NULL = Constant.null(cgutils.voidptr_t)
opaque_pyobject = types.Opaque('Opaque(PyObject)')


class PassThruType(types.Type):
    """Wraps arbitrary Python objects to pass around *nopython-mode*. The created MemInfo will aquire a
       reference to the Python object.
    """
    def __init__(self, name=None):
        super(PassThruType, self).__init__(name or self.__class__.__name__)


pass_thru_type = PassThruType()


@register_model(PassThruType)
class PassThruModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('meminfo', types.MemInfoPointer(opaque_pyobject)),
        ]
        super(PassThruModel, self).__init__(dmm, fe_typ, members)


@unbox(PassThruType)
def unbox_pass_thru_type(typ, obj, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder)
    pass_thru.meminfo = context.pyapi.nrt_meminfo_new_from_pyobject(obj, obj)

    return NativeValue(pass_thru._getvalue())


@box(PassThruType)
def box_pass_thru_type(typ, val, context):
    val = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)
    obj = context.context.nrt.meminfo_data(context.builder, val.meminfo)

    context.pyapi.incref(obj)
    context.context.nrt.decref(context.builder, typ, val._getvalue())

    return obj


class PassThruContainer(object):
    """A container to ferry arbitrary Python objects through *nopython* mode. The only operation supported
       in *nopython-mode* is ``==``. Two instances of ``PassThruContainer`` are equal if the wrapped objects
       are identical, ie if ``a.obj is b.obj``.
    """
    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    def __eq__(self, other):
        return isinstance(other, PassThruContainer) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)


class PassThruContainerType(PassThruType):
    def __init__(self):
        super(PassThruContainerType, self).__init__()


pass_thru_container_type = PassThruContainerType()


@typeof_impl.register(PassThruContainer)
def type_pass_thru_container(val, context):
    return pass_thru_container_type


@register_model(PassThruContainerType)
class PassThruContainerModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('container', pass_thru_type),
            ('wrapped_obj', opaque_pyobject)
        ]
        super(PassThruContainerModel, self).__init__(dmm, fe_typ, members)


make_attribute_wrapper(PassThruContainerType, 'wrapped_obj', 'wrapped_obj')


@unbox(PassThruContainerType)
def unbox_pass_thru_container_type(typ, obj, context):
    container = cgutils.create_struct_proxy(typ)(context.context, context.builder)

    container.container = context.unbox(pass_thru_type, obj).value

    wrapped_obj = context.pyapi.object_getattr_string(obj, "obj")
    context.pyapi.decref(wrapped_obj)
    container.wrapped_obj = wrapped_obj

    return NativeValue(container._getvalue())


@box(PassThruContainerType)
def box_pass_thru_container_type(typ, val, context):
    val = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)

    return context.box(pass_thru_type, val.container)


@lower_builtin(int, types.Opaque)
def opaque_to_int(context, builder, sig, args):
    return builder.ptrtoint(args[0], cgutils.intp_t)


@type_callable(int)
def type_opaque_to_int(context):
    def opaque_to_int_typer(typ):
        if isinstance(typ, types.Opaque):
            return types.intp if not PY3 else types.uintp

    return opaque_to_int_typer


@lower_builtin(is_, types.Opaque, types.Opaque)
def opaque_is(context, builder, sig, args):
    """
    Implementation for `x is y` for Opaque types. `x is y` iff the pointers are equal
    """
    lhs_type, rhs_type = sig.args
    # the lhs and rhs have the same type
    if lhs_type == rhs_type:
        lhs_ptr = builder.ptrtoint(args[0], cgutils.intp_t)
        rhs_ptr = builder.ptrtoint(args[1], cgutils.intp_t)

        return builder.icmp_unsigned('==', lhs_ptr, rhs_ptr)
    else:
        return cgutils.false_bit


@overload(eq)
def pass_thru_container_eq(x, y):
    if x is PassThruContainerType():
        if y is PassThruContainerType():
            def pass_thru_container_pass_thru_container_eq_impl(x, y):
                return x.wrapped_obj is y.wrapped_obj

            return pass_thru_container_pass_thru_container_eq_impl
        else:
            def pass_thru_container_any_eq_impl(x, y):
                return False

            return pass_thru_container_any_eq_impl


@overload_method(PassThruContainerType, '__hash__')
def pass_thru_container_hash_overload(container):
    def pass_thru_container_hash_impl(container):
        return int(container.wrapped_obj)

    return pass_thru_container_hash_impl
