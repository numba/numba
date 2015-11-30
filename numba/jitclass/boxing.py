from __future__ import print_function, absolute_import
from numba import types, cgutils
from numba.pythonapi import box, unbox, NativeValue
from numba.runtime.nrt import MemInfo
from numba.typing.typeof import typeof_impl
from numba import njit
from numba.six import exec_
from llvmlite import ir

_accessor_code_template = """
def accessor(__numba_self_):
    return __numba_self_.{0}
"""


def _generate_accessor(field):
    source = _accessor_code_template.format(field)
    glbls = {}
    exec_(source, glbls)
    accessor = glbls['accessor']
    return njit(accessor)


class BoxedJitClassInstance(object):
    __slots__ = '_meminfo', '_meminfoptr', '_dataptr', '_typ'

    def __new__(cls, meminfoptr, dataptr, typ):
        dct = {'__slots__': ()}
        # Inject attributes as class properties
        for field in typ.struct:
            fn = _generate_accessor(field)
            dct[field] = property(fn)
        # Create a new subclass
        newcls = type(cls.__name__, (cls,), dct)
        return object.__new__(newcls)

    def __init__(self, meminfoptr, dataptr, typ):
        self._meminfo = MemInfo(meminfoptr)
        self._meminfoptr = meminfoptr
        self._dataptr = dataptr
        self._typ = typ


@typeof_impl.register(BoxedJitClassInstance)
def _typeof_boxed_jitclass_instance(val, c):
    return val._typ


@box(types.ClassInstanceType)
def box_jitclass(c, typ, val):
    meminfo, dataptr = cgutils.unpack_tuple(c.builder, val)

    lluintp = c.context.get_data_type(types.uintp)

    addr_meminfo = c.pyapi.from_native_value(types.uintp,
                                             c.builder.ptrtoint(meminfo,
                                                                lluintp))
    addr_dataptr = c.pyapi.from_native_value(types.uintp,
                                             c.builder.ptrtoint(dataptr,
                                                                lluintp))

    # XXX: relies on runtime address
    int_addr_typ = c.context.get_constant(types.uintp, id(typ))

    int_addr_boxcls = c.context.get_constant(types.uintp,
                                             id(BoxedJitClassInstance))

    typ_obj = c.builder.inttoptr(int_addr_typ, c.pyapi.pyobj)
    box_cls = c.builder.inttoptr(int_addr_boxcls, c.pyapi.pyobj)

    args = [addr_meminfo, addr_dataptr, typ_obj]
    res = c.pyapi.call_function_objargs(box_cls, args)

    # Clean up
    for obj in args:
        c.pyapi.decref(obj)

    return res


@unbox(types.ClassInstanceType)
def unbox_jitclass(c, typ, val):
    struct_cls = cgutils.create_struct_proxy(typ)
    inst = struct_cls(c.context, c.builder)

    int_meminfo = c.pyapi.object_getattr_string(val, "_meminfoptr")
    int_dataptr = c.pyapi.object_getattr_string(val, "_dataptr")

    ptr_meminfo = c.pyapi.long_as_voidptr(int_meminfo)
    ptr_dataptr = c.pyapi.long_as_voidptr(int_dataptr)

    c.pyapi.decref(int_meminfo)
    c.pyapi.decref(int_dataptr)

    inst.meminfo = c.builder.bitcast(ptr_meminfo, inst.meminfo.type)
    inst.data = c.builder.bitcast(ptr_dataptr, inst.data.type)

    ret = inst._getvalue()

    c.context.nrt_incref(c.builder, typ, ret)

    # XXX: cleanup to reversed the incref
    return NativeValue(ret, is_error=c.pyapi.c_api_error())


@unbox(types.ImmutableClassRefType)
def unbox_structref(c, typ, val):
    # XXX: not implemented
    struct_cls = cgutils.create_struct_proxy(typ.instance_type)
    ret = struct_cls(c.context, c.builder)._getpointer()
    return NativeValue(ret)


@unbox(types.ImmutableClassInstanceType)
def unbox_structinst(c, typ, val):
    # XXX: not implemented
    struct_cls = cgutils.create_struct_proxy(typ)
    ret = struct_cls(c.context, c.builder)._getvalue()
    return NativeValue(ret)


@box(types.ImmutableClassInstanceType)
def box_structinst(c, typ, val):
    # XXX: not implemented
    return ir.Constant(c.pyapi.pyobj, None)
