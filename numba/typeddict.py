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
    return dictobject._box(dictobject.new_dict(keyty, valty))


@njit
def _length(d):
    # d = dictobject._unbox(*opaque)
    return len(d)


@njit
def _setitem(opaque, key, value):
    d = dictobject._unbox(*opaque)
    d[key] = value


@njit
def _getitem(opaque, key):
    d = dictobject._unbox(*opaque)
    return d[key]


@njit
def _delitem(opaque, key):
    d = dictobject._unbox(*opaque)
    del d[key]


@njit
def _contains(opaque, key):
    d = dictobject._unbox(*opaque)
    return key in d


@njit
def _get(opaque, key, default):
    d = dictobject._unbox(*opaque)
    return d.get(key, default)


@njit
def _setdefault(opaque, key, default):
    d = dictobject._unbox(*opaque)
    return d.setdefault(key, default)


@njit
def _iter(opaque):
    d = dictobject._unbox(*opaque)
    return list(d.keys())


def _from_meminfo_ptr(ptr, dicttype):
    d = TypedDict(meminfo=ptr, dcttype=dicttype)
    return d


class TypedDict(MutableMapping):
    @classmethod
    def empty(cls, key_type, value_type):
        """
        """
        dcttype = DictType(key_type, value_type)
        return cls(dcttype=dcttype)

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        dcttype : numba.types.DictType; keyword-only
            The dictionary type
        """
        # if len(kwargs) != 1:
        #     raise TypeError("too many keyword parameters")
        dcttype = kwargs['dcttype']
        if not isinstance(dcttype, DictType):
            raise TypeError('*dcttype* must be a DictType')
        self._dict_type = dcttype
        if 'meminfo' in kwargs:
            ptr = kwargs['meminfo']
        else:
            ptr = _make_dict(
                self._dict_type.key_type,
                self._dict_type.value_type,
            )
        self._opaque = (ptr, self._dict_type)

    @property
    def _numba_type_(self):
        return self._dict_type

    def __getitem__(self, key):
        return _getitem(self._opaque, key)

    def __setitem__(self, key, value):
        return _setitem(self._opaque, key, value)

    def __delitem__(self, key):
        _delitem(self._opaque, key)

    def __iter__(self):
        return iter(_iter(self._opaque))

    def __len__(self):
        return _length(self)

    def __contains__(self, key):
        return _contains(self._opaque, key)

    def get(self, key, default=None):
        return _get(self._opaque, key, default)

    def setdefault(self, key, default=None):
        return _setdefault(self._opaque, key, default)


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
    # context.nrt.incref(builder, typ, val)
    ctor = cgutils.create_struct_proxy(typ)
    dstruct = ctor(context, builder, value=val)
    # Returns the plain MemInfo
    boxed_meminfo = c.box(
        types.MemInfoPointer(types.voidptr),
        dstruct.meminfo,
    )

    numba_name = c.context.insert_const_string(c.builder.module, 'numba')
    numba_mod = c.pyapi.import_module_noblock(numba_name)
    typeddict_mod = c.pyapi.object_getattr_string(numba_mod, 'typeddict')
    fmp_fn = c.pyapi.object_getattr_string(typeddict_mod, '_from_meminfo_ptr')

    dicttype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

    res = c.pyapi.call_function_objargs(fmp_fn, (boxed_meminfo, dicttype_obj))
    c.pyapi.decref(boxed_meminfo)
    c.pyapi.decref(fmp_fn)
    c.pyapi.decref(numba_mod)
    c.pyapi.decref(typeddict_mod)
    return res


@unbox(types.DictType)
def unbox_dicttype(typ, val, c):
    context = c.context
    builder = c.builder

    opaque = c.pyapi.object_getattr_string(val, '_opaque')
    miptr = c.pyapi.tuple_getitem(opaque, 0)

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
    c.pyapi.decref(opaque)

    return NativeValue(dctobj)
