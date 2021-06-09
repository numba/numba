"""
Python wrapper that connects CPython interpreter to the numba dictobject.
"""
from collections.abc import MutableMapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
    overload,
    box,
    unbox,
    NativeValue,
    type_callable,
    overload_classmethod,
)
from numba.typed import dictobject
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
    d = Dict(meminfo=ptr, dcttype=dicttype)
    return d


class Dict(MutableMapping):
    """A typed-dictionary usable in Numba compiled functions.

    Implements the MutableMapping interface.
    """

    def __new__(cls, dcttype=None, meminfo=None):
        if config.DISABLE_JIT:
            return dict.__new__(dict)
        else:
            return object.__new__(cls)

    @classmethod
    def empty(cls, key_type, value_type):
        """Create a new empty Dict with *key_type* and *value_type*
        as the types for the keys and values of the dictionary respectively.
        """
        if config.DISABLE_JIT:
            return dict()
        else:
            return cls(dcttype=DictType(key_type, value_type))

    def __init__(self, **kwargs):
        """
        For users, the constructor does not take any parameters.
        The keyword arguments are for internal use only.

        Parameters
        ----------
        dcttype : numba.core.types.DictType; keyword-only
            Used internally for the dictionary type.
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        """
        if kwargs:
            self._dict_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._dict_type = None

    def _parse_arg(self, dcttype, meminfo=None):
        if not isinstance(dcttype, DictType):
            raise TypeError('*dcttype* must be a DictType')

        if meminfo is not None:
            opaque = meminfo
        else:
            opaque = _make_dict(dcttype.key_type, dcttype.value_type)
        return dcttype, opaque

    @property
    def _numba_type_(self):
        if self._dict_type is None:
            raise TypeError("invalid operation on untyped dictionary")
        return self._dict_type

    @property
    def _typed(self):
        """Returns True if the dictionary is typed.
        """
        return self._dict_type is not None

    def _initialise_dict(self, key, value):
        dcttype = types.DictType(typeof(key), typeof(value))
        self._dict_type, self._opaque = self._parse_arg(dcttype)

    def __getitem__(self, key):
        if not self._typed:
            raise KeyError(key)
        else:
            return _getitem(self, key)

    def __setitem__(self, key, value):
        if not self._typed:
            self._initialise_dict(key, value)
        return _setitem(self, key, value)

    def __delitem__(self, key):
        if not self._typed:
            raise KeyError(key)
        _delitem(self, key)

    def __iter__(self):
        if not self._typed:
            return iter(())
        else:
            return iter(_iter(self))

    def __len__(self):
        if not self._typed:
            return 0
        else:
            return _length(self)

    def __contains__(self, key):
        if len(self) == 0:
            return False
        else:
            return _contains(self, key)

    def __str__(self):
        buf = []
        for k, v in self.items():
            buf.append("{}: {}".format(k, v))
        return '{{{0}}}'.format(', '.join(buf))

    def __repr__(self):
        body = str(self)
        prefix = str(self._dict_type)
        return "{prefix}({body})".format(prefix=prefix, body=body)

    def get(self, key, default=None):
        if not self._typed:
            return default
        return _get(self, key, default)

    def setdefault(self, key, default=None):
        if not self._typed:
            if default is not None:
                self._initialise_dict(key, default)
        return _setdefault(self, key, default)

    def popitem(self):
        if len(self) == 0:
            raise KeyError('dictionary is empty')
        return _popitem(self)

    def copy(self):
        return _copy(self)


@overload_classmethod(types.DictType, 'empty')
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
