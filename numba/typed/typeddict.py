"""
Python wrapper that connects CPython interpreter to the numba dictobject.
"""
from numba.core.types import DictType, TypeRef
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, cgutils
from numba.core.extending import (
    overload_method,
    overload,
    box,
    unbox,
    NativeValue,
    type_callable,
)
from numba.typed import dictobject
from numba.typed.api import Dict
from numba.core.typing import signature


@njit
def _make_dict(keyty, valty):
    return dictobject._as_meminfo(dictobject.new_dict(keyty, valty))


@njit
def _length(d):
    return len(d)


@njit
def _setitem(d, key, value):
    d[key] = value


@njit
def _getitem(d, key):
    return d[key]


@njit
def _delitem(d, key):
    del d[key]


@njit
def _contains(d, key):
    return key in d


@njit
def _get(d, key, default):
    return d.get(key, default)


@njit
def _setdefault(d, key, default):
    return d.setdefault(key, default)


@njit
def _iter(d):
    return list(d.keys())


@njit
def _popitem(d):
    return d.popitem()


@njit
def _copy(d):
    return d.copy()


def _from_meminfo_ptr(ptr, dicttype):
    from .api import Dict
    d = Dict(meminfo=ptr, dcttype=dicttype)
    return d


# XXX: should we have a better way to classmethod
@overload_method(TypeRef, 'empty')
def typeddict_empty(cls, key_type, value_type):
    if cls.instance_type is not DictType:
        return

    def impl(cls, key_type, value_type):
        return dictobject.new_dict(key_type, value_type)

    return impl


@box(types.DictType)
def box_dicttype(typ, val, c):
    context = c.context
    builder = c.builder

    # XXX deduplicate
    ctor = cgutils.create_struct_proxy(typ)
    dstruct = ctor(context, builder, value=val)
    # Returns the plain MemInfo
    boxed_meminfo = c.box(
        types.MemInfoPointer(types.voidptr),
        dstruct.meminfo,
    )

    modname = c.context.insert_const_string(
        c.builder.module, 'numba.typed.typeddict',
    )
    typeddict_mod = c.pyapi.import_module_noblock(modname)
    fmp_fn = c.pyapi.object_getattr_string(typeddict_mod, '_from_meminfo_ptr')

    dicttype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

    result_var = builder.alloca(c.pyapi.pyobj)
    builder.store(cgutils.get_null_value(c.pyapi.pyobj), result_var)
    with builder.if_then(cgutils.is_not_null(builder, dicttype_obj)):
        res = c.pyapi.call_function_objargs(
            fmp_fn, (boxed_meminfo, dicttype_obj),
        )
        c.pyapi.decref(fmp_fn)
        c.pyapi.decref(typeddict_mod)
        c.pyapi.decref(boxed_meminfo)
        builder.store(res, result_var)
    return builder.load(result_var)


@unbox(types.DictType)
def unbox_dicttype(typ, val, c):
    context = c.context

    # Check that `type(val) is Dict`
    dict_type = c.pyapi.unserialize(c.pyapi.serialize_object(Dict))
    valtype = c.pyapi.object_type(val)
    same_type = c.builder.icmp_unsigned("==", valtype, dict_type)

    with c.builder.if_else(same_type) as (then, orelse):
        with then:
            miptr = c.pyapi.object_getattr_string(val, '_opaque')

            mip_type = types.MemInfoPointer(types.voidptr)
            native = c.unbox(mip_type, miptr)

            mi = native.value

            argtypes = mip_type, typeof(typ)

            def convert(mi, typ):
                return dictobject._from_meminfo(mi, typ)

            sig = signature(typ, *argtypes)
            nil_typeref = context.get_constant_null(argtypes[1])
            args = (mi, nil_typeref)
            is_error, dctobj = c.pyapi.call_jit_code(convert , sig, args)
            # decref here because we are stealing a reference.
            c.context.nrt.decref(c.builder, typ, dctobj)

            c.pyapi.decref(miptr)
            bb_unboxed = c.builder.basic_block

        with orelse:
            # Raise error on incorrect type
            c.pyapi.err_format(
                "PyExc_TypeError",
                "can't unbox a %S as a %S",
                valtype, dict_type,
            )
            bb_else = c.builder.basic_block

    # Phi nodes to gather the output
    dctobj_res = c.builder.phi(dctobj.type)
    is_error_res = c.builder.phi(is_error.type)

    dctobj_res.add_incoming(dctobj, bb_unboxed)
    dctobj_res.add_incoming(dctobj.type(None), bb_else)

    is_error_res.add_incoming(is_error, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)

    # cleanup
    c.pyapi.decref(dict_type)
    c.pyapi.decref(valtype)

    return NativeValue(dctobj_res, is_error=is_error_res)


@type_callable(DictType)
def typeddict_call(context):
    """
    Defines typing logic for ``Dict()``.
    Produces Dict[undefined, undefined]
    """
    def typer():
        return types.DictType(types.undefined, types.undefined)
    return typer


@overload(numba_typeref_ctor)
def impl_numba_typeref_ctor(cls):
    """
    Defines ``Dict()``, the type-inferred version of the dictionary ctor.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise DictType.

    See also: `redirect_type_ctor` in numba/cpython/bulitins.py
    """
    dict_ty = cls.instance_type
    if not isinstance(dict_ty, types.DictType):
        msg = "expecting a DictType but got {}".format(dict_ty)
        return  # reject
    # Ensure the dictionary is precisely typed.
    if not dict_ty.is_precise():
        msg = "expecting a precise DictType but got {}".format(dict_ty)
        raise errors.LoweringError(msg)

    key_type = types.TypeRef(dict_ty.key_type)
    value_type = types.TypeRef(dict_ty.value_type)

    def impl(cls):
        # Simply call .empty() with the key/value types from *cls*
        return Dict.empty(key_type, value_type)

    return impl
