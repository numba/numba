from __future__ import print_function, absolute_import
from numba import types, cgutils
from numba.pythonapi import box, unbox
from numba.runtime.nrt import MemInfo

class BoxedJitClassInstance(object):
    __slots__ = '_meminfo', '_dataptr', '_typ'

    def __init__(self, meminfoptr, dataptr, typ):
        self._meminfo = MemInfo(meminfoptr)
        self._dataptr = dataptr
        self._typ = typ
        # XXX: impl nrt_decref


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

    for obj in args:
        c.pyapi.decref(obj)

    return res
