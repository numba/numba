
"""
Python wrapper that connects CPython interpreter to the numba listobject.
"""
from numba.six import MutableSequence
from numba.types import ListType, TypeRef
from numba.targets.imputils import numba_typeref_ctor
from numba import listobject
from numba import njit, types, cgutils, errors, typeof
from numba.extending import (
    overload_method,
    overload,
    box,
    unbox,
    NativeValue,
    type_callable,
)


@njit
def _make_list(itemty):
    return listobject._as_meminfo(listobject.new_list(itemty))


@njit
def _length(l):
    return len(l)


@njit
def _append(l, item):
    l.append(item)


@njit
def _setitem(l, i, item):
    l[i] = item


@njit
def _getitem(l, i):
    return l[i]


@njit
def _contains(l, item):
    return item in l


@njit
def _count(l, item):
    return l.count(item)


def _from_meminfo_ptr(ptr, listtype):
    return List(meminfo=ptr, lsttype=listtype)


class List(MutableSequence):
    """A typed-list usable in Numba compiled functions.

    Implements the MutableSequence interface.
    """
    @classmethod
    def empty_list(cls, item_type):
        """Create a new empty List with *item_type* as the type for the items
        of the list .
        """
        return cls(lsttype=ListType(item_type))

    def __init__(self, **kwargs):
        """
        For users, the constructor does not take any parameters.
        The keyword arguments are for internal use only.

        Parameters
        ----------
        lsttype : numba.types.ListType; keyword-only
            Used internally for the list type.
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        """
        if kwargs:
            self._list_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._list_type = None

    def _parse_arg(self, lsttype, meminfo=None):
        if not isinstance(lsttype, ListType):
            raise TypeError('*lsttype* must be a ListType')

        if meminfo is not None:
            opaque = meminfo
        else:
            opaque = _make_list(lsttype.item_type)
        return lsttype, opaque

    @property
    def _numba_type_(self):
        if self._list_type is None:
            raise TypeError("invalid operation on untyped list")
        return self._list_type

    @property
    def _typed(self):
        """Returns True if the list is typed.
        """
        return self._list_type is not None

    def _initialise_list(self, item):
        lsttype = types.ListType(typeof(item))
        self._list_type, self._opaque = self._parse_arg(lsttype)

    def __len__(self):
        if not self._typed:
            return 0
        else:
            return _length(self)

    def append(self, item):
        if not self._typed:
            self._initialise_list(item)
        _append(self, item)

    def __setitem__(self, i, item):
        if not self._typed:
            self._initialise_list(item)
        _setitem(self, i, item)

    def __getitem__(self, i):
        if not self._typed:
            raise IndexError
        else:
            return _getitem(self, i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item):
        return _contains(self, item)

    def __delitem__(self, i):
        pass

    def insert(self, i, item):
        pass

    def count(self, item):
        return _count(self, item)


# XXX: should we have a better way to classmethod
@overload_method(TypeRef, 'empty_list')
def typedlist_empty(cls, item_type):
    if cls.instance_type is not ListType:
        return

    def impl(cls, item_type):
        return listobject.new_list(item_type)

    return impl


@box(types.ListType)
def box_lsttype(typ, val, c):
    context = c.context
    builder = c.builder

    # XXX deduplicate
    ctor = cgutils.create_struct_proxy(typ)
    lstruct = ctor(context, builder, value=val)
    # Returns the plain MemInfo
    boxed_meminfo = c.box(
        types.MemInfoPointer(types.voidptr),
        lstruct.meminfo,
    )

    modname = c.context.insert_const_string(
        c.builder.module, 'numba.typed.typedlist',
    )
    typedlist_mod = c.pyapi.import_module_noblock(modname)
    fmp_fn = c.pyapi.object_getattr_string(typedlist_mod, '_from_meminfo_ptr')

    lsttype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

    res = c.pyapi.call_function_objargs(fmp_fn, (boxed_meminfo, lsttype_obj))
    c.pyapi.decref(fmp_fn)
    c.pyapi.decref(typedlist_mod)
    c.pyapi.decref(boxed_meminfo)
    return res


@unbox(types.ListType)
def unbox_listtype(typ, val, c):
    context = c.context
    builder = c.builder

    miptr = c.pyapi.object_getattr_string(val, '_opaque')

    native = c.unbox(types.MemInfoPointer(types.voidptr), miptr)

    mi = native.value
    ctor = cgutils.create_struct_proxy(typ)
    lstruct = ctor(context, builder)

    data_pointer = context.nrt.meminfo_data(builder, mi)
    data_pointer = builder.bitcast(
        data_pointer,
        listobject.ll_list_type.as_pointer(),
    )

    lstruct.data = builder.load(data_pointer)
    lstruct.meminfo = mi

    lstobj = lstruct._getvalue()
    c.pyapi.decref(miptr)

    return NativeValue(lstobj)


#
# The following contains the logic for the type-inferred constructor
#


@type_callable(ListType)
def typedlist_call(context):
    """
    Defines typing logic for ``List()``.
    Produces List[undefined]
    """
    def typer():
        return types.ListType(types.undefined)
    return typer


@overload(numba_typeref_ctor)
def impl_numba_typeref_ctor(cls):
    """
    Defines ``List()``, the type-inferred version of the list ctor.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise ListType.

    See also: `redirect_type_ctor` in numba/target/bulitins.py
    """
    list_ty = cls.instance_type
    if not isinstance(list_ty, types.ListType):
        msg = "expecting a ListType but got {}".format(list_ty)
        return  # reject
    # Ensure the list is precisely typed.
    if not list_ty.is_precise():
        msg = "expecting a precise ListType but got {}".format(list_ty)
        raise errors.LoweringError(msg)

    item_type = types.TypeRef(list_ty.item_type)

    def impl(cls):
        # Simply call .empty_list with the item types from *cls*
        return List.empty_list(item_type)

    return impl
