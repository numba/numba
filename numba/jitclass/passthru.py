from ..datamodel import default_manager as data_model_manager, models
from ..extending import overload, register_model
from ..pythonapi import NativeValue, unbox, box
from ..runtime.nrtdynmod import _debug_print
from ..targets.imputils import lower_getattr, lower_builtin
from ..typing.templates import infer_getattr, AttributeTemplate
from ..typing.typeof import typeof_impl

from . import _box

from operator import is_, eq

from llvmlite import ir
from llvmlite.llvmpy.core import Constant
from .. import cgutils, types


__all__ = ['PassThruContainer']


NULL = Constant.null(cgutils.voidptr_t)


def create_pass_thru_native(context, builder, typ, meminfo=NULL):
    payload_type = typ.payload_type

    pass_thru = cgutils.create_struct_proxy(typ)(context, builder)

    with builder.if_else(cgutils.is_not_null(builder, meminfo)) as (meminfo_exists, create_meminfo):
        with meminfo_exists:
            # meminfo already exists, incref only
            bb_meminfo_exists = builder.basic_block
            pass_thru.meminfo = meminfo
            context.nrt.incref(builder, typ, pass_thru._getvalue())

        with create_meminfo:
            # create a new meminfo
            bb_meminfo_created = builder.basic_block
            ll_type = context.get_value_type(payload_type)
            size = ir.Constant(cgutils.intp_t, context.get_abi_sizeof(ll_type))

            dtor = imp_dtor(context, builder.module, typ)
            pass_thru.meminfo = context.nrt.meminfo_alloc_dtor(builder, size, dtor)

            # nullify the parent
            payload = typ.get_payload(context, builder, pass_thru)
            payload.parent = cgutils.get_null_value(payload.parent.type)

    created_new = builder.phi(cgutils.bool_t)
    created_new.add_incoming(cgutils.false_bit, bb_meminfo_exists)
    created_new.add_incoming(cgutils.true_bit, bb_meminfo_created)

    pass_thru.data = builder.bitcast(context.nrt.meminfo_data(builder, pass_thru.meminfo), ll_type.as_pointer())

    return pass_thru, created_new


def imp_dtor(context, module, typ):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = ir.FunctionType(ir.VoidType(), [llvoidptr, llsize, llvoidptr])

    fname = "_Dtor.{0}".format(typ.name)
    dtor_fn = module.get_or_insert_function(dtor_ftype, name=fname)

    if dtor_fn.is_declaration:
        # Define
        dtor_fn.linkage = 'internal'
        builder = ir.IRBuilder(dtor_fn.append_basic_block())

        alloc_fe_type = typ.payload_type
        alloc_type = context.get_value_type(alloc_fe_type)

        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
        data = builder.load(ptr)

        context.nrt.decref(builder, alloc_fe_type, data)

        builder.ret_void()

    return dtor_fn


def _access_box_member(context, builder, obj, member_offset):
    # Access member by byte offset
    offset = context.get_constant(types.uintp, member_offset)
    llvoidptr = ir.IntType(8).as_pointer()
    ptr = cgutils.pointer_add(builder, obj, offset)
    casted = builder.bitcast(ptr, llvoidptr.as_pointer())
    return builder.load(casted)


def _set_box_member(context, builder, obj, member_offset, value):
    # Access member by byte offset
    offset = context.get_constant(types.uintp, member_offset)
    ptr = cgutils.pointer_add(builder, obj, offset)
    casted = builder.bitcast(ptr, cgutils.voidptr_t.as_pointer())
    builder.store(value, casted)


def unbox_pass_thru_type(typ, obj, context):
    payload_type = typ.payload_type
    meminfo_type = context.context.get_value_type(
        data_model_manager[typ].get_member_fe_type('meminfo')
    )

    if _debug_print:
        cgutils.printf(context.builder, "**** unboxing {} form pyobject %p\n".format(typ), obj)

    # load from Python object
    meminfo = context.builder.bitcast(
        _access_box_member(context.context, context.builder, obj, _box.box_meminfoptr_offset),
        meminfo_type
    )
    pass_thru, created_new = create_pass_thru_native(context.context, context.builder, typ, meminfo)

    # unbox the payload even if we don't need to (b/c it already was unboxed) as we cannot cleanup otherwise
    native_payload = context.unbox(payload_type, obj)

    with context.builder.if_else(created_new) as (first_unbox, already_unboxed):
        with first_unbox:
            #  obj was not unboxed before
            #  write pointer to the meminfo onto the Box atttributes and incref as Box dtor will decref once
            context.context.nrt.incref(context.builder, typ, pass_thru._getvalue())
            _set_box_member(
                context.context,
                context.builder,
                obj,
                _box.box_meminfoptr_offset,
                context.builder.bitcast(pass_thru.meminfo, cgutils.voidptr_t)
            )
            _set_box_member(
                context.context,
                context.builder,
                obj,
                _box.box_dataptr_offset,
                context.builder.bitcast(pass_thru.data, cgutils.voidptr_t)
            )

            context.builder.store(native_payload.value, pass_thru.data)

        with already_unboxed:
            # get rid of the ref to payload
            context.context.nrt.decref(context.builder, payload_type, native_payload.value)

    if _debug_print:
        cgutils.printf(
            context.builder,
            "**** unboxed {} form pyobject %p to meminfo %p, py_refcount=%u, nrt_refcount=%u\n".format(typ),
            obj,
            pass_thru.meminfo,
            context.builder.load(obj),
            context.builder.load(pass_thru.meminfo)
        )
    is_error = cgutils.is_not_null(context.builder, context.pyapi.err_occurred())

    if native_payload.cleanup is not None:
        def cleanup():
            with context.builder.if_then(created_new):
                native_payload.cleanup()
    else:
        cleanup = None

    return NativeValue(pass_thru._getvalue(), is_error=is_error, cleanup=cleanup)


def box_pass_thru_type(typ, val, context):
    payload_type = typ.payload_type

    val = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)
    payload = typ.get_payload(context.context, context.builder, val)
    parent = payload.parent

    if _debug_print:
        cgutils.printf(
            context.builder,
            "**** boxing {} form pyobject %p, meminfo %p\n".format(typ),
            parent,
            val.meminfo
        )

    with context.builder.if_else(cgutils.is_not_null(context.builder, parent)) as (python_created, no_python_created):
        with python_created:
            context.pyapi.incref(parent)
            context.context.nrt.decref(context.builder, typ, val._getvalue())
            bb_python_created = context.builder.basic_block

        with no_python_created:
            # val was created on the native side, aquire a new ref as the boxer will release one
            context.context.nrt.incref(context.builder, payload_type, payload._getvalue())
            new_parent = context.box(payload_type, payload._getvalue())
            payload.parent = new_parent

            # check we have successfully boxed the payload before trying to store something onto the pointer
            with context.builder.if_then(cgutils.is_not_null(context.builder, new_parent)):
                # take the ref to the meminfo onto the Box
                _set_box_member(
                    context.context,
                    context.builder,
                    new_parent,
                    _box.box_meminfoptr_offset,
                    context.builder.bitcast(val.meminfo, cgutils.voidptr_t)
                )
                _set_box_member(
                    context.context,
                    context.builder,
                    new_parent,
                    _box.box_dataptr_offset,
                    context.builder.bitcast(val.data, cgutils.voidptr_t)
                )

            bb_no_python_created = context.builder.basic_block

    obj = context.builder.phi(cgutils.voidptr_t)
    obj.add_incoming(parent, bb_python_created)
    obj.add_incoming(new_parent, bb_no_python_created)

    with context.builder.if_else(cgutils.is_not_null(context.builder, obj)) as (success, fail):
        with success:
            if _debug_print:
                cgutils.printf(
                    context.builder, "**** boxed {} to pyobject %p, refcounts (py, numba): %p, %p\n".format(typ),
                    obj,
                    context.builder.load(obj),
                    context.builder.load(val.meminfo)
                )

        with fail:
            if _debug_print:
                cgutils.printf(
                    context.builder, "**** failed to boxed {} to from %p\n".format(typ),
                    val.meminfo
                )

    return obj


class PassThruPayloadType(types.Type):
    def __init__(self, passthru_type):
        self.passthru_type = passthru_type
        super(PassThruPayloadType, self).__init__("PassThruPayloadType({})".format(passthru_type.name))

    @property
    def key(self):
        return self.passthru_type


class PassThruTypeBase(types.Hashable):
    nopython_attrs = []

    def __init__(self):
        super(PassThruTypeBase, self).__init__(self.__class__.__name__)

    @property
    def payload_type(self):
        payload_type_class, _, _ = _passthru_types[self.__class__]
        return payload_type_class(self)

    def get_payload(self, context, builder, value):
        payload_ptr = builder.bitcast(context.nrt.meminfo_data(builder, value.meminfo), value.data.type)
        payload = context.make_helper(builder, self.payload_type, ref=payload_ptr)

        return payload


@register_model(PassThruPayloadType)
class PassThruPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', types.pyobject)
        ]
        members.extend(
            [(k, v(fe_type.passthru_type)) for k, v in fe_type.passthru_type.nopython_attrs]
        )
        super(PassThruPayloadModel, self).__init__(dmm, fe_type, members)


class PassThruModel(models.StructModel):
    payload_type = lambda fe_type: None

    def __init__(self, dmm, fe_typ):
        dtype = types.Opaque('Opaque.' + str(self.payload_type.__name__))
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
            ('data', types.CPointer(self.payload_type(fe_typ))),
        ]
        super(PassThruModel, self).__init__(dmm, fe_typ, members)


def unbox_passthru_payload(typ, obj, context):
    payload = cgutils.create_struct_proxy(typ)(context.context, context.builder)
    payload.parent = obj

    return NativeValue(payload._getvalue())


_passthru_types = {}


def create_pass_thru_type(typ, unbox_payload=unbox_passthru_payload):
    name = typ.__name__
    nopython_attrs = typ.nopython_attrs

    if 'Type' not in name:
        raise ValueError("Pass-through types should be named like 'MyPassThruType'")

    payload_type, model, payload_model = _passthru_types.get(typ, (None, None, None))
    if payload_type is None:
        payload_type = type(name.replace('Type', 'PayloadType', 1), (PassThruPayloadType,), {})

        model = type(name.replace('Type', 'Model', 1), (PassThruModel,), dict(payload_type=payload_type))
        payload_model = type(
            name.replace('Type', 'PayloadModel', 1),
            (PassThruPayloadModel,), dict(spec=typ.nopython_attrs)
        )

        _passthru_types[typ] = payload_type, model, payload_model

        register_model(typ)(model)
        register_model(payload_type)(payload_model)

        unbox(typ)(unbox_pass_thru_type)
        box(typ)(box_pass_thru_type)

        unbox(payload_type)(unbox_payload)

        @infer_getattr
        class MemManagedObjectAttrs(AttributeTemplate):
            key = typ

            def resolve(self, value, attr):
                # TODO: for some reason pyobject is not considered precise, try Dummy instead
                if attr == 'meminfo':
                    return types.MemInfoPointer(value.payload_type)
                elif attr == 'parent':
                    return types.Opaque(
                        'Opaque(PyObject)')  # admitting this is a PyObject* seems to stop no-python mode (?)
                else:
                    try:
                        dm = data_model_manager[value.payload_type]
                        return dm.get_member_fe_type(attr)
                    except KeyError:
                        pass

        @lower_getattr(typ, 'meminfo')
        def get_meminfo_impl(context, builder, typ, value):
            value = cgutils.create_struct_proxy(typ)(context, builder, value=value)
            return value.meminfo

        def getter_factory(name):
            @lower_getattr(typ, name)
            def impl(context, builder, typ, value):
                payload_type = typ.payload_type
                payload_dm = data_model_manager[payload_type]
                member_type = payload_dm.get_member_fe_type(name)

                value = cgutils.create_struct_proxy(typ)(context, builder, value=value)
                payload = typ.get_payload(context, builder, value)

                attr = getattr(payload, name)
                context.nrt.incref(builder, member_type, attr)

                return attr

            return impl

        getter_factory('parent')
        for name, _ in typ.nopython_attrs:
            getter_factory(name)

    elif typ.nopython_attrs != nopython_attrs:
        raise ValueError("Pass-through type '{}' already defined with different nopython_attrs.".format(name))

    return payload_type


class PassThruContainer(_box.Box):
    def __init__(self, obj):
        super(PassThruContainer, self).__init__()
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, PassThruContainer) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)     # TODO: probably not the best choice but easy to implement on native side


class PassThruContainerType(PassThruTypeBase):
    nopython_attrs = [('wrapped_obj', lambda _: types.Opaque('Opaque(PyObject)'))]

    def __init__(self):
        super(PassThruContainerType, self).__init__()


def unbox_passthru_container_payload(typ, obj, context):
    payload = cgutils.create_struct_proxy(typ)(context.context, context.builder)
    payload.parent = obj

    wrapped_obj = context.pyapi.object_getattr_string(obj, 'obj')
    payload.wrapped_obj = wrapped_obj       # borrow ref to obj.obj
    context.pyapi.decref(wrapped_obj)

    is_error = cgutils.is_not_null(context.builder, context.pyapi.err_occurred())
    return NativeValue(payload._getvalue(), is_error=is_error)


PassThruContainerPayloadType = create_pass_thru_type(
    PassThruContainerType, unbox_payload=unbox_passthru_container_payload
)


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


@lower_builtin(hash, PassThruContainerType())
def passthru_container_hash(context, builder, sig, args):
    typ, = sig.args

    val = cgutils.create_struct_proxy(typ)(context, builder, value=args[0])
    payload = typ.get_payload(context, builder, val)
    ptr = payload.wrapped_obj

    ll_return_type = context.get_value_type(sig.return_type)
    return builder.ptrtoint(ptr, ll_return_type)


@typeof_impl.register(PassThruContainer)
def type_passthru_container(val, context):
    return PassThruContainerType()
