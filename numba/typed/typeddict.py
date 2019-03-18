"""
Python wrapper that connects CPython interpreter to the numba dictobject.
"""
from collections import MutableMapping

from numba.types import DictType, TypeRef
from numba import njit, dictobject, types, cgutils
from numba.extending import (
    overload_method,
    box,
    unbox,
    NativeValue
)


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
    @classmethod
    def empty(cls, key_type, value_type):
        """Create a new empty Dict with *key_type* and *value_type*
        as the types for the keys and values of the dictionary respectively.
        """
        return cls(dcttype=DictType(key_type, value_type))

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        dcttype : numba.types.DictType; keyword-only
            The dictionary type
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        """
        self._dict_type, self._opaque = self._parse_arg(**kwargs)

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
        return self._dict_type

    def __getitem__(self, key):
        return _getitem(self, key)

    def __setitem__(self, key, value):
        return _setitem(self, key, value)

    def __delitem__(self, key):
        _delitem(self, key)

    def __iter__(self):
        return iter(_iter(self))

    def __len__(self):
        return _length(self)

    def __contains__(self, key):
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
        return _get(self, key, default)

    def setdefault(self, key, default=None):
        return _setdefault(self, key, default)

    def popitem(self):
        return _popitem(self)

    def copy(self):
        return _copy(self)


# XXX: should we have a better way to classmethod
@overload_method(TypeRef, 'empty')
def typeddict_empty(cls,  key_type, value_type):
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

    res = c.pyapi.call_function_objargs(fmp_fn, (boxed_meminfo, dicttype_obj))
    c.pyapi.decref(fmp_fn)
    c.pyapi.decref(typeddict_mod)
    c.pyapi.decref(boxed_meminfo)
    return res


@unbox(types.DictType)
def unbox_dicttype(typ, val, c):
    context = c.context
    builder = c.builder

    miptr = c.pyapi.object_getattr_string(val, '_opaque')

    native = c.unbox(types.MemInfoPointer(types.voidptr), miptr)

    mi = native.value
    ctor = cgutils.create_struct_proxy(typ)
    dstruct = ctor(context, builder)

    data_pointer = context.nrt.meminfo_data(builder, mi)
    data_pointer = builder.bitcast(
        data_pointer,
        dictobject.ll_dict_type.as_pointer(),
    )

    dstruct.data = builder.load(data_pointer)
    dstruct.meminfo = mi

    dctobj = dstruct._getvalue()
    c.pyapi.decref(miptr)

    return NativeValue(dctobj)
