from .. import cgutils, types
from ..datamodel import models
from ..extending import make_attribute_wrapper, overload, register_model, type_callable
from ..pythonapi import NativeValue, unbox, box
from ..six import PY3
from ..targets.imputils import lower_builtin
from ..typing.typeof import typeof_impl

from operator import is_, eq
from llvmlite.llvmpy.core import Constant


__all__ = ['PassThruContainer']


NULL = Constant.null(cgutils.voidptr_t)
opaque_pyobject = types.Opaque('Opaque(PyObject)')


class PassThruType(types.Type):
    def __init__(self):
        super(PassThruType, self).__init__(self.__class__.__name__)


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
    pass_thru.meminfo = context.pyapi.nrt_meminfo_new_from_pyobject(obj, obj)    # store obj on meminfo.data

    return NativeValue(pass_thru._getvalue())


@box(PassThruType)
def box_pass_thru_type(typ, val, context):
    val = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)
    obj = context.context.nrt.meminfo_data(context.builder, val.meminfo)

    # obj is allowed to be NULL to make this re-usable for derived pass through
    # types that can be nopython created, it is the responsibility of the boxer
    # of the derived type to clear the error
    with context.builder.if_else(cgutils.is_null(context.builder, obj)) as (no_python_created, python_created):
        with no_python_created:
            context.pyapi.err_set_string(
                "PyExc_NotImplementedError", "Native creation of '{}' not implemented.".format(typ.name)
            )

        with python_created:
            context.pyapi.incref(obj)

    context.context.nrt.decref(context.builder, typ, val._getvalue())

    return obj


class PassThru(object):
    pass


@type_callable(PassThru)
def type_pass_thru_constructor(context):
    def pass_thru_constructor_typer():
        return pass_thru_type

    return pass_thru_constructor_typer


@lower_builtin(PassThru)
def pass_thru_constructor(context, builder, sig, args):
    typ = sig.return_type

    pass_thru = cgutils.create_struct_proxy(typ)(context, builder)

    meminfo = context.nrt.meminfo_alloc(builder, context.get_constant(types.intp, 0))
    context.nrt.meminfo_set_data(builder, meminfo, NULL)

    pass_thru.meminfo = meminfo

    return pass_thru._getvalue()


class PassThruContainer(object):
    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    def __eq__(self, other):
        return isinstance(other, PassThruContainer) and self.obj is other.obj

    def __hash__(self):
        # TODO: probably not the best choice but easy to implement on native side
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
def generic_passthru_eq(xty, yty):
    if xty is PassThruContainerType():
        if yty is PassThruContainerType():
            def generic_passthru_generic_passthru_eq(x, y):
                return x.wrapped_obj is y.wrapped_obj

            return generic_passthru_generic_passthru_eq
        else:
            def generic_passthru_any_eq(x, y):
                return False

            return generic_passthru_any_eq


@type_callable(hash)
def type_hash_passthrucontainer(context):
    def hash_passthrucontainer_typer(container):
        if container is PassThruContainerType():
            return types.intp if not PY3 else types.uintp
        else:
            return types.intp

    return hash_passthrucontainer_typer


@lower_builtin(hash, PassThruContainerType())
def passthru_container_hash(context, builder, sig, args):
    typ, = sig.args

    val = cgutils.create_struct_proxy(typ)(context, builder, value=args[0])

    ll_return_type = context.get_value_type(sig.return_type)
    ptr_as_int = builder.ptrtoint(val.wrapped_obj, ll_return_type)

    if PY3:
        # for py3 we have to emulate the cast to unsigned long from PyLong_FromVoidPtr
        bb_not_casted = builder.basic_block

        from sys import maxsize as MAXSIZE
        with builder.if_then(cgutils.is_neg_int(builder, ptr_as_int)):
            bb_casted = builder.basic_block
            casted_res = builder.sub(ptr_as_int, context.get_constant(types.intp, MAXSIZE))

        res = builder.phi(cgutils.intp_t)
        res.add_incoming(ptr_as_int, bb_not_casted)
        res.add_incoming(casted_res, bb_casted)

        return res

    else:
        return ptr_as_int
