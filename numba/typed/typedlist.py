"""
Python wrapper that connects CPython interpreter to the Numba typed-list.

This is the code that is used when creating typed lists outside of a `@jit`
context and when returning a typed-list from a `@jit` decorated function. It
basically a Python class that has a Numba allocated typed-list under the hood
and uses `@jit` functions to access it. Since it inherits from MutableSequence
it should really quack like the CPython `list`.

"""

from numba.core.types import ListType, TypeRef
from numba.core.imputils import numba_typeref_ctor
from numba.core import types, cgutils
from numba import njit
from numba.core.extending import (
    overload_method,
    overload,
    box,
    unbox,
    NativeValue,
    type_callable,
)
from numba.typed import api, listobject
from numba.typed.api import List
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
import sys

if sys.version_info >= (3, 8):
    T_co = pt.TypeVar('T_co', covariant=True)

    class _Sequence(pt.Protocol[T_co]):
        def __getitem__(self, i: int) -> T_co:
            ...

        def __len__(self) -> int:
            ...


DEFAULT_ALLOCATED = api.DEFAULT_ALLOCATED


@njit
def _make_list(itemty, allocated=DEFAULT_ALLOCATED):
    return listobject._as_meminfo(listobject.new_list(itemty,
                                                      allocated=allocated))


@njit
def _length(l):
    return len(l)


@njit
def _allocated(l):
    return l._allocated()


@njit
def _is_mutable(l):
    return l._is_mutable()


@njit
def _make_mutable(l):
    return l._make_mutable()


@njit
def _make_immutable(l):
    return l._make_immutable()


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


T = pt.TypeVar('T')
T_or_ListT = pt.Union[T, 'List[T]']


# XXX: should we have a better way to classmethod
@overload_method(TypeRef, 'empty_list')
def typedlist_empty(cls, item_type, allocated=DEFAULT_ALLOCATED):
    if cls.instance_type is not ListType:
        return

    def impl(cls, item_type, allocated=DEFAULT_ALLOCATED):
        return listobject.new_list(item_type, allocated=allocated)

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

    result_var = builder.alloca(c.pyapi.pyobj)
    builder.store(cgutils.get_null_value(c.pyapi.pyobj), result_var)

    with builder.if_then(cgutils.is_not_null(builder, lsttype_obj)):
        res = c.pyapi.call_function_objargs(
            fmp_fn, (boxed_meminfo, lsttype_obj),
        )
        c.pyapi.decref(fmp_fn)
        c.pyapi.decref(typedlist_mod)
        c.pyapi.decref(boxed_meminfo)
        builder.store(res, result_var)
    return builder.load(result_var)


@unbox(types.ListType)
def unbox_listtype(typ, val, c):
    context = c.context
    builder = c.builder

    # Check that `type(val) is Dict`
    list_type = c.pyapi.unserialize(c.pyapi.serialize_object(List))
    valtype = c.pyapi.object_type(val)
    same_type = builder.icmp_unsigned("==", valtype, list_type)

    with c.builder.if_else(same_type) as (then, orelse):
        with then:
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
            bb_unboxed = c.builder.basic_block

        with orelse:
            # Raise error on incorrect type
            c.pyapi.err_format(
                "PyExc_TypeError",
                "can't unbox a %S as a %S",
                valtype, list_type,
            )
            bb_else = c.builder.basic_block

    # Phi nodes to gather the output
    lstobj_res = c.builder.phi(lstobj.type)
    is_error_res = c.builder.phi(cgutils.bool_t)

    lstobj_res.add_incoming(lstobj, bb_unboxed)
    lstobj_res.add_incoming(lstobj.type(None), bb_else)

    is_error_res.add_incoming(cgutils.false_bit, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)

    # cleanup
    c.pyapi.decref(list_type)
    c.pyapi.decref(valtype)

    return NativeValue(lstobj_res, is_error=is_error_res)


#
# The following contains the logic for the type-inferred constructor
#

def _guess_dtype(iterable):
    """Guess the correct dtype of the iterable type. """
    if not isinstance(iterable, types.IterableType):
        raise TypingError(
            "List() argument must be iterable")
    # Special case for nested NumPy arrays.
    elif isinstance(iterable, types.Array) and iterable.ndim > 1:
        return iterable.copy(ndim=iterable.ndim - 1, layout='A')
    elif hasattr(iterable, "dtype"):
        return iterable.dtype
    elif hasattr(iterable, "yield_type"):
        return iterable.yield_type
    elif isinstance(iterable, types.UnicodeType):
        return iterable
    elif isinstance(iterable, types.DictType):
        return iterable.key_type
    else:
        # This should never happen, since the 'dtype' of any iterable
        # should have determined above.
        raise TypingError(
            "List() argument does not have a suitable dtype")


@type_callable(ListType)
def typedlist_call(context):
    """Defines typing logic for ``List()`` and ``List(iterable)``.

    If no argument is given, the returned typer types a new typed-list with an
    undefined item type. If a single argument is given it must be iterable with
    a guessable 'dtype'. In this case, the typer types a new typed-list with
    the type set to the 'dtype' of the iterable arg.

    Parameters
    ----------
    arg : single iterable (optional)
        The single optional argument.

    Returns
    -------
    typer : function
        A typer suitable to type constructor calls.

    Raises
    ------
    The returned typer raises a TypingError in case of unsuitable arguments.

    """

    class Typer(object):

        def attach_sig(self):
            from inspect import signature as mypysig

            def mytyper(iterable):
                pass
            self.pysig = mypysig(mytyper)

        def __call__(self, *args, **kwargs):
            if kwargs:
                raise TypingError(
                    "List() takes no keyword arguments"
                )
            elif args:
                if not 0 <= len(args) <= 1:
                    raise TypingError(
                        "List() expected at most 1 argument, got {}"
                        .format(len(args))
                    )
                rt = types.ListType(_guess_dtype(args[0]))
                self.attach_sig()
                return Signature(rt, args, None, pysig=self.pysig)
            else:
                item_type = types.undefined
                return types.ListType(item_type)

    return Typer()


@overload(numba_typeref_ctor)
def impl_numba_typeref_ctor(cls, *args):
    """Defines lowering for ``List()`` and ``List(iterable)``.

    This defines the lowering logic to instantiate either an empty typed-list
    or a typed-list initialised with values from a single iterable argument.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise ListType.
    args: tuple
        A tuple that contains a single iterable (optional)

    Returns
    -------
    impl : function
        An implementation suitable for lowering the constructor call.

    See also: `redirect_type_ctor` in numba/cpython/bulitins.py
    """
    list_ty = cls.instance_type
    if not isinstance(list_ty, types.ListType):
        return  # reject
    # Ensure the list is precisely typed.
    if not list_ty.is_precise():
        msg = "expecting a precise ListType but got {}".format(list_ty)
        raise LoweringError(msg)

    item_type = types.TypeRef(list_ty.item_type)
    if args:
        # special case 0d Numpy arrays
        if isinstance(args[0], types.Array) and args[0].ndim == 0:
            def impl(cls, *args):
                # Instatiate an empty list and populate it with the single
                # value from the array.
                r = List.empty_list(item_type)
                r.append(args[0].item())
                return r
        else:
            def impl(cls, *args):
                # Instatiate an empty list and populate it with values from the
                # iterable.
                r = List.empty_list(item_type)
                for i in args[0]:
                    r.append(i)
                return r
    else:
        def impl(cls, *args):
            # Simply call .empty_list with the item type from *cls*
            return List.empty_list(item_type)

    return impl
