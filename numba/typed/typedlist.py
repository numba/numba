
"""
Python wrapper that connects CPython interpreter to the Numba typed-list.

This is the code that is used when creating typed lists outside of a `@jit`
context and when returning a typed-list from a `@jit` decorated function. It
basically a Python class that has a Numba allocated typed-list under the hood
and uses `@jit` functions to access it. Since it inherits from MutableSequence
it should really quack like the CPython `list`.

"""
from numba.six import MutableSequence
from numba.types import ListType, TypeRef
from numba.targets.imputils import numba_typeref_ctor
from numba import listobject
from numba.dispatcher import Dispatcher
from numba import config
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
def _make_list(itemty, allocated=0):
    return listobject._as_meminfo(listobject.new_list(itemty,
                                                      allocated=allocated))


@njit
def _length(l):
    return len(l)


@njit
def _allocated(l):
    return l._allocated()


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


@njit
def _pop(l, i):
    return l.pop(i)


@njit
def _delitem(l, i):
    del l[i]


@njit
def _extend(l, iterable):
    return l.extend(iterable)


@njit
def _insert(l, i, item):
    l.insert(i, item)


@njit
def _remove(l, item):
    l.remove(item)


@njit
def _clear(l):
    l.clear()


@njit
def _reverse(l):
    l.reverse()


@njit
def _copy(l):
    return l.copy()


@njit
def _eq(t, o):
    return t == o


@njit
def _ne(t, o):
    return t != o


@njit
def _lt(t, o):
    return t < o


@njit
def _le(t, o):
    return t <= o


@njit
def _gt(t, o):
    return t > o


@njit
def _ge(t, o):
    return t >= o


@njit
def _index(l, item, start, end):
    return l.index(item, start, end)


@njit
def _sort(l, key, reverse):
    return l.sort(key, reverse)


def _from_meminfo_ptr(ptr, listtype):
    return List(meminfo=ptr, lsttype=listtype)


class List(MutableSequence):
    """A typed-list usable in Numba compiled functions.

    Implements the MutableSequence interface.
    """

    def __new__(cls, lsttype=None, meminfo=None, allocated=None):
        if config.DISABLE_JIT:
            return list.__new__(list)
        else:
            return object.__new__(cls)

    @classmethod
    def empty_list(cls, item_type, allocated=0):
        """Create a new empty List.

        Parameters
        ----------
        item_type: Numba type
            type of the list item.
        allocated: int
            number of items to pre-allocate
        """
        if config.DISABLE_JIT:
            return list()
        else:
            return cls(lsttype=ListType(item_type), allocated=allocated)

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
        allocated: int; keyword-only
            Used internally to pre-allocate space for items
        """
        if kwargs:
            self._list_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._list_type = None

    def _parse_arg(self, lsttype, meminfo=None, allocated=0):
        if not isinstance(lsttype, ListType):
            raise TypeError('*lsttype* must be a ListType')

        if meminfo is not None:
            opaque = meminfo
        else:
            opaque = _make_list(lsttype.item_type, allocated=allocated)
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

    def _allocated(self):
        if not self._typed:
            return 0
        else:
            return _allocated(self)

    def __eq__(self, other):
        return _eq(self, other)

    def __ne__(self, other):
        return _ne(self, other)

    def __lt__(self, other):
        return _lt(self, other)

    def __le__(self, other):
        return _le(self, other)

    def __gt__(self, other):
        return _gt(self, other)

    def __ge__(self, other):
        return _ge(self, other)

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
        _delitem(self, i)

    def insert(self, i, item):
        if not self._typed:
            self._initialise_list(item)
        _insert(self, i, item)

    def count(self, item):
        return _count(self, item)

    def pop(self, i=-1):
        return _pop(self, i)

    def extend(self, iterable):
        if not self._typed:
            # Need to get the first element of the iterable to initialise the
            # type of the list. FIXME: this may be a problem if the iterable
            # can not be sliced.
            self._initialise_list(iterable[0])
            self.append(iterable[0])
            return _extend(self, iterable[1:])
        return _extend(self, iterable)

    def remove(self, item):
        return _remove(self, item)

    def clear(self):
        return _clear(self)

    def reverse(self):
        return _reverse(self)

    def copy(self):
        return _copy(self)

    def index(self, item, start=None, stop=None):
        return _index(self, item, start, stop)

    def sort(self, key=None, reverse=False):
        """Sort the list inplace.

        See also ``list.sort()``
        """
        # If key is not already a dispatcher object, make it so
        if callable(key) and not isinstance(key, Dispatcher):
            key = njit(key)
        return _sort(self, key, reverse)

    def __str__(self):
        buf = []
        for x in self:
            buf.append("{}".format(x))
        return '[{0}]'.format(', '.join(buf))

    def __repr__(self):
        body = str(self)
        prefix = str(self._list_type)
        return "{prefix}({body})".format(prefix=prefix, body=body)


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
