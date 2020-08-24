"""
Python wrapper that connects CPython interpreter to the Numba typed-list.

This is the code that is used when creating typed lists outside of a `@jit`
context and when returning a typed-list from a `@jit` decorated function. It
basically a Python class that has a Numba allocated typed-list under the hood
and uses `@jit` functions to access it. Since it inherits from MutableSequence
it should really quack like the CPython `list`.

"""
from collections.abc import MutableSequence

from numba.core.types import ListType, TypeRef
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
    overload_method,
    overload,
    box,
    unbox,
    NativeValue,
    type_callable,
)
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature

DEFAULT_ALLOCATED = listobject.DEFAULT_ALLOCATED


def _from_meminfo_ptr(ptr, listtype):
    return List(meminfo=ptr, lsttype=listtype)


class List(MutableSequence):
    """A typed-list usable in Numba compiled functions.

    Implements the MutableSequence interface.
    """

    _legal_kwargs = ["lsttype", "meminfo", "allocated"]

    def __new__(cls,
                lsttype=None,
                meminfo=None,
                allocated=DEFAULT_ALLOCATED,
                **kwargs):
        if config.DISABLE_JIT:
            return list.__new__(list)
        else:
            return object.__new__(cls)

    @classmethod
    def empty_list(cls, item_type, allocated=DEFAULT_ALLOCATED):
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

    def __init__(self, *args, **kwargs):
        """
        For users, the constructor does not take any parameters.
        The keyword arguments are for internal use only.

        Parameters
        ----------
        args: iterable
            The iterable to intialize the list from
        lsttype : numba.core.types.ListType; keyword-only
            Used internally for the list type.
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        allocated: int; keyword-only
            Used internally to pre-allocate space for items
        """
        illegal_kwargs = any((kw not in self._legal_kwargs for kw in kwargs))
        if illegal_kwargs or args and kwargs:
            raise TypeError("List() takes no keyword arguments")

        from numba.typed.containermethods import (length, setitem, getitem,
                                                  delitem, contains, copy,
                                                  _make_list, list_allocated,
                                                  list_is_mutable,
                                                  list_make_mutable,
                                                  list_make_immutable,
                                                  list_append, list_count,
                                                  list_pop, list_extend,
                                                  list_insert, list_remove,
                                                  list_clear, list_reverse,
                                                  list_eq, list_ne, list_lt,
                                                  list_le, list_gt, list_ge,
                                                  list_index, list_sort)
        self._method_table = {'length': length,
                              'setitem': setitem,
                              'getitem': getitem,
                              'delitem': delitem,
                              'contains': contains,
                              'copy': copy,
                              'make_list': _make_list,
                              'list_allocated':list_allocated,
                              'list_is_mutable':list_is_mutable,
                              'list_make_mutable':list_make_mutable,
                              'list_make_immutable':list_make_immutable,
                              'list_append':list_append,
                              'list_count':list_count,
                              'list_pop':list_pop,
                              'list_extend':list_extend,
                              'list_insert':list_insert,
                              'list_remove':list_remove,
                              'list_clear':list_clear,
                              'list_reverse':list_reverse,
                              'list_eq':list_eq,
                              'list_ne':list_ne,
                              'list_lt':list_lt,
                              'list_le':list_le,
                              'list_gt':list_gt,
                              'list_ge':list_ge,
                              'list_index':list_index,
                              'list_sort':list_sort,
                              }

        if kwargs:
            self._list_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._list_type = None
            if args:
                if not 0 <= len(args) <= 1:
                    raise TypeError(
                        "List() expected at most 1 argument, got {}"
                        .format(len(args))
                    )
                iterable = args[0]
                # Special case Numpy scalars or anything that quacks like a
                # NumPy Array.
                if hasattr(iterable, "ndim") and iterable.ndim == 0:
                    self.append(iterable.item())
                else:
                    try:
                        iter(iterable)
                    except TypeError:
                        raise TypeError("List() argument must be iterable")
                    for i in args[0]:
                        self.append(i)

    def _parse_arg(self, lsttype, meminfo=None, allocated=DEFAULT_ALLOCATED):
        if not isinstance(lsttype, ListType):
            raise TypeError('*lsttype* must be a ListType')

        if meminfo is not None:
            opaque = meminfo
        else:
            opaque = self._method_table['make_list'](lsttype.item_type,
                                                     allocated=allocated)
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

    @property
    def _dtype(self):
        if not self._typed:
            raise RuntimeError("invalid operation on untyped list")
        return self._list_type.dtype

    def _initialise_list(self, item):
        lsttype = types.ListType(typeof(item))
        self._list_type, self._opaque = self._parse_arg(lsttype)

    def __len__(self):
        if not self._typed:
            return 0
        else:
            return self._method_table['length'](self)

    def _allocated(self):
        if not self._typed:
            return DEFAULT_ALLOCATED
        else:
            return self._method_table['list_allocated'](self)

    def _is_mutable(self):
        return self._method_table['list_is_mutable'](self)

    def _make_mutable(self):
        return self._method_table['list_make_mutable'](self)

    def _make_immutable(self):
        return self._method_table['list_make_immutable'](self)

    def __eq__(self, other):
        return self._method_table['list_eq'](self, other)

    def __ne__(self, other):
        return self._method_table['list_ne'](self, other)

    def __lt__(self, other):
        return self._method_table['list_lt'](self, other)

    def __le__(self, other):
        return self._method_table['list_le'](self, other)

    def __gt__(self, other):
        return self._method_table['list_gt'](self, other)

    def __ge__(self, other):
        return self._method_table['list_ge'](self, other)

    def append(self, item):
        if not self._typed:
            self._initialise_list(item)
        self._method_table['list_append'](self, item)

    def __setitem__(self, i, item):
        if not self._typed:
            self._initialise_list(item)
        self._method_table['setitem'](self, i, item)

    def __getitem__(self, i):
        if not self._typed:
            raise IndexError
        else:
            return self._method_table['getitem'](self, i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item):
        return self._method_table['contains'](self, item)

    def __delitem__(self, i):
        self._method_table['delitem'](self, i)

    def insert(self, i, item):
        if not self._typed:
            self._initialise_list(item)
        self._method_table['list_insert'](self, i, item)

    def count(self, item):
        return self._method_table['list_count'](self, item)

    def pop(self, i=-1):
        return self._method_table['list_pop'](self, i)

    def extend(self, iterable):
        # Empty iterable, do nothing
        if len(iterable) == 0:
            return self
        if not self._typed:
            # Need to get the first element of the iterable to initialise the
            # type of the list. FIXME: this may be a problem if the iterable
            # can not be sliced.
            self._initialise_list(iterable[0])
        return self._method_table['list_extend'](self, iterable)

    def remove(self, item):
        return self._method_table['list_remove'](self, item)

    def clear(self):
        return self._method_table['list_clear'](self)

    def reverse(self):
        return self._method_table['list_reverse'](self)

    def copy(self):
        return self._method_table['copy'](self)

    def index(self, item, start=None, stop=None):
        return self._method_table['list_index'](self, item, start, stop)

    def sort(self, key=None, reverse=False):
        """Sort the list inplace.

        See also ``list.sort()``
        """
        # If key is not already a dispatcher object, make it so
        if callable(key) and not isinstance(key, Dispatcher):
            # Import here so it doesn't trigger JIT initialisation on import
            # of this module.
            key = njit(key)
        return self._method_table['list_sort'](self, key, reverse)

    def __str__(self):
        buf = []
        for x in self:
            buf.append("{}".format(x))
        return '[{0}]'.format(', '.join(buf))

    def __repr__(self):
        body = str(self)
        prefix = str(self._list_type) if self._typed else "ListType[Undefined]"
        return "{prefix}({body})".format(prefix=prefix, body=body)


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
