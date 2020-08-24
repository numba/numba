"""
Methods for operating on a typed container payloads from CPython, used
predominantly by numba.typed.typeddict.Dict and numba.typed.typedlist.List.
These methods are in a separate module to prevent them from being imported at
numba.__init__ time, this triggering JIT, and also to benefit from in memory
caching.
"""
from numba import njit
from numba.typed import dictobject, listobject

# ------------------------------------------------------------------------------
# Typed container shared methods
# ------------------------------------------------------------------------------


@njit
def length(d):
    return len(d)


@njit
def setitem(d, key, value):
    d[key] = value


@njit
def getitem(d, key):
    return d[key]


@njit
def delitem(d, key):
    del d[key]


@njit
def contains(d, key):
    return key in d


@njit
def copy(d):
    return d.copy()


# ------------------------------------------------------------------------------
# Typed Dict methods
# ------------------------------------------------------------------------------

@njit
def _make_dict(keyty, valty):
    return dictobject._as_meminfo(dictobject.new_dict(keyty, valty))


@njit
def dict_get(d, key, default):
    return d.get(key, default)


@njit
def dict_setdefault(d, key, default):
    return d.setdefault(key, default)


@njit
def dict_iter(d):
    return list(d.keys())


@njit
def dict_popitem(d):
    return d.popitem()


# ------------------------------------------------------------------------------
# Typed List methods
# ------------------------------------------------------------------------------

@njit
def _make_list(itemty, allocated=listobject.DEFAULT_ALLOCATED):
    return listobject._as_meminfo(listobject.new_list(itemty,
                                                      allocated=allocated))


@njit
def list_allocated(l):
    return l._allocated()


@njit
def list_is_mutable(l):
    return l._is_mutable()


@njit
def list_make_mutable(l):
    return l._make_mutable()


@njit
def list_make_immutable(l):
    return l._make_immutable()


@njit
def list_append(l, item):
    l.append(item)


@njit
def list_count(l, item):
    return l.count(item)


@njit
def list_pop(l, i):
    return l.pop(i)


@njit
def list_extend(l, iterable):
    return l.extend(iterable)


@njit
def list_insert(l, i, item):
    l.insert(i, item)


@njit
def list_remove(l, item):
    l.remove(item)


@njit
def list_clear(l):
    l.clear()


@njit
def list_reverse(l):
    l.reverse()


@njit
def list_eq(t, o):
    return t == o


@njit
def list_ne(t, o):
    return t != o


@njit
def list_lt(t, o):
    return t < o


@njit
def list_le(t, o):
    return t <= o


@njit
def list_gt(t, o):
    return t > o


@njit
def list_ge(t, o):
    return t >= o


@njit
def list_index(l, item, start, end):
    return l.index(item, start, end)


@njit
def list_sort(l, key, reverse):
    return l.sort(key, reverse)
