from collections.abc import MutableMapping, MutableSequence
import typing as pt
from numba.core import config
from numba import typeof
from numba.core.types import ListType, DictType

T = pt.TypeVar('T')
T_or_ListT = pt.Union[T, 'List[T]']
Int_or_Slice = pt.Union["pt.SupportsIndex", slice]


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
            from numba.typed.typeddict import _make_dict
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
        dcttype = DictType(typeof(key), typeof(value))
        self._dict_type, self._opaque = self._parse_arg(dcttype)

    def __getitem__(self, key):
        if not self._typed:
            raise KeyError(key)
        else:
            from numba.typed.typeddict import _getitem
            return _getitem(self, key)

    def __setitem__(self, key, value):
        if not self._typed:
            self._initialise_dict(key, value)
        from numba.typed.typeddict import _setitem
        return _setitem(self, key, value)

    def __delitem__(self, key):
        if not self._typed:
            raise KeyError(key)
        from numba.typed.typeddict import _delitem
        _delitem(self, key)

    def __iter__(self):
        if not self._typed:
            return iter(())
        else:
            from numba.typed.typeddict import _iter
            return iter(_iter(self))

    def __len__(self):
        if not self._typed:
            return 0
        else:
            from numba.typed.typeddict import _length
            return _length(self)

    def __contains__(self, key):
        if len(self) == 0:
            return False
        else:
            from numba.typed.typeddict import _contains
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
        from numba.typed.typeddict import _get
        return _get(self, key, default)

    def setdefault(self, key, default=None):
        if not self._typed:
            if default is not None:
                self._initialise_dict(key, default)
        from numba.typed.typeddict import _setdefault
        return _setdefault(self, key, default)

    def popitem(self):
        if len(self) == 0:
            raise KeyError('dictionary is empty')
        from numba.typed.typeddict import _popitem
        return _popitem(self)

    def copy(self):
        from numba.typed.typeddict import _copy
        return _copy(self)


DEFAULT_ALLOCATED = 0


class List(MutableSequence, pt.Generic[T]):
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
            from numba.typed.typedlist import _make_list
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

    @property
    def _dtype(self):
        if not self._typed:
            raise RuntimeError("invalid operation on untyped list")
        return self._list_type.dtype

    def _initialise_list(self, item):
        lsttype = ListType(typeof(item))
        self._list_type, self._opaque = self._parse_arg(lsttype)

    def __len__(self) -> int:
        if not self._typed:
            return 0
        else:
            from numba.typed.typedlist import _length
            return _length(self)

    def _allocated(self):
        if not self._typed:
            return DEFAULT_ALLOCATED
        else:
            from numba.typed.typedlist import _allocated
            return _allocated(self)

    def _is_mutable(self):
        from numba.typed.typedlist import _is_mutable
        return _is_mutable(self)

    def _make_mutable(self):
        from numba.typed.typedlist import _make_mutable
        return _make_mutable(self)

    def _make_immutable(self):
        from numba.typed.typedlist import _make_immutable
        return _make_immutable(self)

    def __eq__(self, other):
        from numba.typed.typedlist import _eq
        return _eq(self, other)

    def __ne__(self, other):
        from numba.typed.typedlist import _ne
        return _ne(self, other)

    def __lt__(self, other):
        from numba.typed.typedlist import _lt
        return _lt(self, other)

    def __le__(self, other):
        from numba.typed.typedlist import _le
        return _le(self, other)

    def __gt__(self, other):
        from numba.typed.typedlist import _gt
        return _gt(self, other)

    def __ge__(self, other):
        from numba.typed.typedlist import _ge
        return _ge(self, other)

    def append(self, item: T) -> None:
        if not self._typed:
            self._initialise_list(item)
        from numba.typed.typedlist import _append
        _append(self, item)

    # noqa F811 comments required due to github.com/PyCQA/pyflakes/issues/592
    # noqa E704 required to follow overload style of using ... in the same line
    @pt.overload  # type: ignore[override]
    def __setitem__(self, i: int, o: T) -> None: ...  # noqa: F811, E704
    @pt.overload
    def __setitem__(self, s: slice, o: 'List[T]') -> None: ...  # noqa: F811, E704, E501

    def __setitem__(self, i: Int_or_Slice, item: T_or_ListT) -> None:  # noqa: F811, E501
        if not self._typed:
            self._initialise_list(item)
        from numba.typed.typedlist import _setitem
        _setitem(self, i, item)

    # noqa F811 comments required due to github.com/PyCQA/pyflakes/issues/592
    # noqa E704 required to follow overload style of using ... in the same line
    @pt.overload
    def __getitem__(self, i: int) -> T: ...  # noqa: F811, E704
    @pt.overload
    def __getitem__(self, i: slice) -> 'List[T]': ...  # noqa: F811, E704

    def __getitem__(self, i: Int_or_Slice) -> T_or_ListT:  # noqa: F811
        if not self._typed:
            raise IndexError
        else:
            from numba.typed.typedlist import _getitem
            return _getitem(self, i)

    def __iter__(self) -> pt.Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: T) -> bool:  # type: ignore[override]
        from numba.typed.typedlist import _contains
        return _contains(self, item)

    def __delitem__(self, i: Int_or_Slice) -> None:
        from numba.typed.typedlist import _delitem
        _delitem(self, i)

    def insert(self, i: int, item: T) -> None:
        if not self._typed:
            self._initialise_list(item)
        from numba.typed.typedlist import _insert
        _insert(self, i, item)

    def count(self, item: T) -> int:
        from numba.typed.typedlist import _count
        return _count(self, item)

    def pop(self, i: "pt.SupportsIndex" = -1) -> T:
        from numba.typed.typedlist import _pop
        return _pop(self, i)

    # type: ignore[override]
    def extend(self, iterable: "_Sequence[T]") -> None:
        # Empty iterable, do nothing
        if len(iterable) == 0:
            return None
        if not self._typed:
            # Need to get the first element of the iterable to initialise the
            # type of the list. FIXME: this may be a problem if the iterable
            # can not be sliced.
            self._initialise_list(iterable[0])
        from numba.typed.typedlist import _extend
        return _extend(self, iterable)

    def remove(self, item: T) -> None:
        from numba.typed.typedlist import _remove
        return _remove(self, item)

    def clear(self):
        from numba.typed.typedlist import _clear
        return _clear(self)

    def reverse(self):
        from numba.typed.typedlist import _reverse
        return _reverse(self)

    def copy(self):
        from numba.typed.typedlist import _copy
        return _copy(self)

    def index(self, item: T, start: pt.Optional[int] = None,
              stop: pt.Optional[int] = None) -> int:
        from numba.typed.typedlist import _index
        return _index(self, item, start, stop)

    def sort(self, key=None, reverse=False):
        """Sort the list inplace.

        See also ``list.sort()``
        """
        # If key is not already a dispatcher object, make it so
        from numba.core.dispatcher import Dispatcher
        from numba import njit
        if callable(key) and not isinstance(key, Dispatcher):
            key = njit(key)
        from numba.typed.typedlist import _sort
        return _sort(self, key, reverse)

    def __str__(self):
        buf = []
        for x in self:
            buf.append("{}".format(x))
        # Check whether the code was invoked from IPython shell
        try:
            get_ipython
            return '[{0}, ...]'.format(', '.join(buf[:1000]))
        except (NameError, IndexError):
            return '[{0}]'.format(', '.join(buf))

    def __repr__(self):
        body = str(self)
        prefix = str(self._list_type) if self._typed else "ListType[Undefined]"
        return "{prefix}({body})".format(prefix=prefix, body=body)
