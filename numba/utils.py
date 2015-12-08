from __future__ import print_function, division, absolute_import

import atexit
import collections
import functools
import io
import itertools
import os
import threading
import timeit
import math
import sys

import numpy

from .six import *
from numba.config import PYVERSION, MACHINE_BITS


IS_PY3 = PYVERSION >= (3, 0)

if IS_PY3:
    import builtins
    INT_TYPES = (int,)
    longint = int
    get_ident = threading.get_ident
    intern = sys.intern
    file_replace = os.replace
else:
    import thread
    import __builtin__ as builtins
    INT_TYPES = (int, long)
    longint = long
    get_ident = thread.get_ident
    intern = intern
    if sys.platform == 'win32':
        def file_replace(src, dest):
            # Best-effort emulation of os.replace()
            try:
                os.rename(src, dest)
            except OSError:
                os.unlink(dest)
                os.rename(src, dest)
    else:
        file_replace = os.rename

try:
    from inspect import signature as pysignature
except ImportError:
    try:
        from funcsigs import signature as pysignature
    except ImportError:
        raise ImportError("please install the 'funcsigs' package "
                          "('pip install funcsigs')")

try:
    from functools import singledispatch
except ImportError:
    try:
        from singledispatch import singledispatch
    except ImportError:
        raise ImportError("please install the 'singledispatch' package "
                          "('pip install singledispatch')")


# Mapping between operator module functions and the corresponding built-in
# operators.

operator_map = [
    # Binary
    ('add', 'iadd', '+'),
    ('sub', 'isub', '-'),
    ('mul', 'imul', '*'),
    ('floordiv', 'ifloordiv', '//'),
    ('truediv', 'itruediv', '/'),
    ('mod', 'imod', '%'),
    ('pow', 'ipow', '**'),
    ('and_', 'iand', '&'),
    ('or_', 'ior', '|'),
    ('xor', 'ixor', '^'),
    ('lshift', 'ilshift', '<<'),
    ('rshift', 'irshift', '>>'),
    ('eq', '', '=='),
    ('ne', '', '!='),
    ('lt', '', '<'),
    ('le', '', '<='),
    ('gt', '', '>'),
    ('ge', '', '>='),
    # Unary
    ('pos', '', '+'),
    ('neg', '', '-'),
    ('invert', '', '~'),
    ('not_', '', 'not'),
    ]

if not IS_PY3:
    operator_map.append(('div', 'idiv', '/?'))
if sys.version_info >= (3, 5):
    operator_map.append(('matmul', 'imatmul', '@'))

# Map of known in-place operators to their corresponding copying operators
inplace_map = dict((op + '=', op)
                   for (_bin, _inp, op) in operator_map
                   if _inp)


_shutting_down = False

def _at_shutdown():
    global _shutting_down
    _shutting_down = True

atexit.register(_at_shutdown)

def shutting_down(globals=globals):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """
    # At shutdown, the attribute may have been cleared or set to None.
    v = globals().get('_shutting_down')
    return v is True or v is None


class ConfigOptions(object):
    OPTIONS = {}

    def __init__(self):
        self._values = self.OPTIONS.copy()

    def set(self, name, value=True):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        self._values[name] = value

    def unset(self, name):
        self.set(name, False)

    def __getattr__(self, name):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        return self._values[name]

    def __repr__(self):
        return "Flags(%s)" % ', '.join('%s=%s' % (k, v)
                                       for k, v in self._values.items()
                                       if v is not False)

    def copy(self):
        copy = type(self)()
        copy._values = self._values.copy()
        return copy

    def __eq__(self, other):
        return isinstance(other, ConfigOptions) and other._values == self._values

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(sorted(self._values.items())))


class SortedMap(collections.Mapping):
    """Immutable
    """

    def __init__(self, seq):
        self._values = []
        self._index = {}
        for i, (k, v) in enumerate(sorted(seq)):
            self._index[k] = i
            self._values.append((k, v))

    def __getitem__(self, k):
        i = self._index[k]
        return self._values[i][1]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(k for k, v in self._values)


class SortedSet(collections.Set):
    def __init__(self, seq):
        self._set = set(seq)
        self._values = list(sorted(self._set))

    def __contains__(self, item):
        return item in self._set

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise AssertionError("key already in dictionary: %r" % (key,))
        super(UniqueDict, self).__setitem__(key, value)


class NonReentrantLock(object):
    """
    A lock class which explicitly forbids reentrancy.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def acquire(self):
        me = get_ident()
        if me == self._owner:
            raise RuntimeError("cannot re-acquire lock from same thread")
        self._lock.acquire()
        self._owner = me

    def release(self):
        if self._owner != get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._owner = None
        self._lock.release()

    def is_owned(self):
        """
        Whether the lock is owned by the current thread.
        """
        return self._owner == get_ident()

    __enter__ = acquire

    def __exit__(self, t, v, tb):
        self.release()


# Django's cached_property
# see https://docs.djangoproject.com/en/dev/ref/utils/#django.utils.functional.cached_property

class cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.

    Optional ``name`` argument allows you to make cached properties of other
    methods. (e.g.  url = cached_property(get_absolute_url, name='url') )
    """
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


def runonce(fn):
    @functools.wraps(fn)
    def inner():
        if not inner._ran:
            res = fn()
            inner._result = res
            inner._ran = True
        return inner._result

    inner._ran = False
    return inner


def bit_length(intval):
    """
    Return the number of bits necessary to represent integer `intval`.
    """
    assert isinstance(intval, INT_TYPES)
    if intval >= 0:
        return len(bin(intval)) - 2
    else:
        return len(bin(-intval - 1)) - 2


class BenchmarkResult(object):
    def __init__(self, func, records, loop):
        self.func = func
        self.loop = loop
        self.records = numpy.array(records) / loop
        self.best = numpy.min(self.records)

    def __repr__(self):
        name = getattr(self.func, "__name__", self.func)
        args = (name, self.loop, self.records.size, format_time(self.best))
        return "%20s: %10d loops, best of %d: %s per loop" % args


def format_time(tm):
    units = "s ms us ns ps".split()
    base = 1
    for unit in units[:-1]:
        if tm >= base:
            break
        base /= 1000
    else:
        unit = units[-1]
    return "%.1f%s" % (tm / base, unit)


def benchmark(func, maxsec=1):
    timer = timeit.Timer(func)
    number = 1
    result = timer.repeat(1, number)
    # Too fast to be measured
    while min(result) / number == 0:
        number *= 10
        result = timer.repeat(3, number)
    best = min(result) / number
    if best >= maxsec:
        return BenchmarkResult(func, result, number)
        # Scale it up to make it close the maximum time
    max_per_run_time = maxsec / 3 / number
    number = max(max_per_run_time / best / 3, 1)
    # Round to the next power of 10
    number = int(10 ** math.ceil(math.log10(number)))
    records = timer.repeat(3, number)
    return BenchmarkResult(func, records, number)


RANGE_ITER_OBJECTS = (builtins.range,)
if PYVERSION < (3, 0):
    RANGE_ITER_OBJECTS += (builtins.xrange,)
    try:
        from future.types.newrange import newrange
        RANGE_ITER_OBJECTS += (newrange,)
    except ImportError:
        pass


# Backported from Python 3.4: functools.total_ordering()

def _not_op(op, other):
    # "not a < b" handles "a >= b"
    # "not a <= b" handles "a > b"
    # "not a >= b" handles "a < b"
    # "not a > b" handles "a <= b"
    op_result = op(other)
    if op_result is NotImplemented:
        return NotImplemented
    return not op_result


def _op_or_eq(op, self, other):
    # "a < b or a == b" handles "a <= b"
    # "a > b or a == b" handles "a >= b"
    op_result = op(other)
    if op_result is NotImplemented:
        return NotImplemented
    return op_result or self == other


def _not_op_and_not_eq(op, self, other):
    # "not (a < b or a == b)" handles "a > b"
    # "not a < b and a != b" is equivalent
    # "not (a > b or a == b)" handles "a < b"
    # "not a > b and a != b" is equivalent
    op_result = op(other)
    if op_result is NotImplemented:
        return NotImplemented
    return not op_result and self != other


def _not_op_or_eq(op, self, other):
    # "not a <= b or a == b" handles "a >= b"
    # "not a >= b or a == b" handles "a <= b"
    op_result = op(other)
    if op_result is NotImplemented:
        return NotImplemented
    return not op_result or self == other


def _op_and_not_eq(op, self, other):
    # "a <= b and not a == b" handles "a < b"
    # "a >= b and not a == b" handles "a > b"
    op_result = op(other)
    if op_result is NotImplemented:
        return NotImplemented
    return op_result and self != other


def _is_inherited_from_object(cls, op):
    """
    Whether operator *op* on *cls* is inherited from the root object type.
    """
    if PYVERSION >= (3,):
        object_op = getattr(object, op)
        cls_op = getattr(cls, op)
        return object_op is cls_op
    else:
        # In 2.x, the inherited operator gets a new descriptor, so identity
        # doesn't work.  OTOH, dir() doesn't list methods inherited from
        # object (which it does in 3.x).
        return op not in dir(cls)


def total_ordering(cls):
    """Class decorator that fills in missing ordering methods"""
    convert = {
        '__lt__': [('__gt__',
                    lambda self, other: _not_op_and_not_eq(self.__lt__, self,
                                                           other)),
                   ('__le__',
                    lambda self, other: _op_or_eq(self.__lt__, self, other)),
                   ('__ge__', lambda self, other: _not_op(self.__lt__, other))],
        '__le__': [('__ge__',
                    lambda self, other: _not_op_or_eq(self.__le__, self,
                                                      other)),
                   ('__lt__',
                    lambda self, other: _op_and_not_eq(self.__le__, self,
                                                       other)),
                   ('__gt__', lambda self, other: _not_op(self.__le__, other))],
        '__gt__': [('__lt__',
                    lambda self, other: _not_op_and_not_eq(self.__gt__, self,
                                                           other)),
                   ('__ge__',
                    lambda self, other: _op_or_eq(self.__gt__, self, other)),
                   ('__le__', lambda self, other: _not_op(self.__gt__, other))],
        '__ge__': [('__le__',
                    lambda self, other: _not_op_or_eq(self.__ge__, self,
                                                      other)),
                   ('__gt__',
                    lambda self, other: _op_and_not_eq(self.__ge__, self,
                                                       other)),
                   ('__lt__', lambda self, other: _not_op(self.__ge__, other))]
    }
    # Find user-defined comparisons (not those inherited from object).
    roots = [op for op in convert if not _is_inherited_from_object(cls, op)]
    if not roots:
        raise ValueError(
            'must define at least one ordering operation: < > <= >=')
    root = max(roots)       # prefer __lt__ to __le__ to __gt__ to __ge__
    for opname, opfunc in convert[root]:
        if opname not in roots:
            opfunc.__name__ = opname
            opfunc.__doc__ = getattr(int, opname).__doc__
            setattr(cls, opname, opfunc)
    return cls


# Backported from Python 3.4: weakref.finalize()

from weakref import ref

class finalize:
    """Class for finalization of weakrefable objects

    finalize(obj, func, *args, **kwargs) returns a callable finalizer
    object which will be called when obj is garbage collected. The
    first time the finalizer is called it evaluates func(*arg, **kwargs)
    and returns the result. After this the finalizer is dead, and
    calling it just returns None.

    When the program exits any remaining finalizers for which the
    atexit attribute is true will be run in reverse order of creation.
    By default atexit is true.
    """

    # Finalizer objects don't have any state of their own.  They are
    # just used as keys to lookup _Info objects in the registry.  This
    # ensures that they cannot be part of a ref-cycle.

    __slots__ = ()
    _registry = {}
    _shutdown = False
    _index_iter = itertools.count()
    _dirty = False
    _registered_with_atexit = False

    class _Info:
        __slots__ = ("weakref", "func", "args", "kwargs", "atexit", "index")

    def __init__(self, obj, func, *args, **kwargs):
        if not self._registered_with_atexit:
            # We may register the exit function more than once because
            # of a thread race, but that is harmless
            import atexit
            atexit.register(self._exitfunc)
            finalize._registered_with_atexit = True
        info = self._Info()
        info.weakref = ref(obj, self)
        info.func = func
        info.args = args
        info.kwargs = kwargs or None
        info.atexit = True
        info.index = next(self._index_iter)
        self._registry[self] = info
        finalize._dirty = True

    def __call__(self, _=None):
        """If alive then mark as dead and return func(*args, **kwargs);
        otherwise return None"""
        info = self._registry.pop(self, None)
        if info and not self._shutdown:
            return info.func(*info.args, **(info.kwargs or {}))

    def detach(self):
        """If alive then mark as dead and return (obj, func, args, kwargs);
        otherwise return None"""
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is not None and self._registry.pop(self, None):
            return (obj, info.func, info.args, info.kwargs or {})

    def peek(self):
        """If alive then return (obj, func, args, kwargs);
        otherwise return None"""
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is not None:
            return (obj, info.func, info.args, info.kwargs or {})

    @property
    def alive(self):
        """Whether finalizer is alive"""
        return self in self._registry

    @property
    def atexit(self):
        """Whether finalizer should be called at exit"""
        info = self._registry.get(self)
        return bool(info) and info.atexit

    @atexit.setter
    def atexit(self, value):
        info = self._registry.get(self)
        if info:
            info.atexit = bool(value)

    def __repr__(self):
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is None:
            return '<%s object at %#x; dead>' % (type(self).__name__, id(self))
        else:
            return '<%s object at %#x; for %r at %#x>' % \
                (type(self).__name__, id(self), type(obj).__name__, id(obj))

    @classmethod
    def _select_for_exit(cls):
        # Return live finalizers marked for exit, oldest first
        L = [(f,i) for (f,i) in cls._registry.items() if i.atexit]
        L.sort(key=lambda item:item[1].index)
        return [f for (f,i) in L]

    @classmethod
    def _exitfunc(cls):
        # At shutdown invoke finalizers for which atexit is true.
        # This is called once all other non-daemonic threads have been
        # joined.
        reenable_gc = False
        try:
            if cls._registry:
                import gc
                if gc.isenabled():
                    reenable_gc = True
                    gc.disable()
                pending = None
                while True:
                    if pending is None or finalize._dirty:
                        pending = cls._select_for_exit()
                        finalize._dirty = False
                    if not pending:
                        break
                    f = pending.pop()
                    try:
                        # gc is disabled, so (assuming no daemonic
                        # threads) the following is the only line in
                        # this function which might trigger creation
                        # of a new finalizer
                        f()
                    except Exception:
                        sys.excepthook(*sys.exc_info())
                    assert f not in cls._registry
        finally:
            # prevent any more finalizers from executing during shutdown
            finalize._shutdown = True
            if reenable_gc:
                gc.enable()


try:
    from collections import OrderedDict
except ImportError:
    # Copied from http://code.activestate.com/recipes/576693/

    # Backport of OrderedDict() class that runs on Python 2.4, 2.5, 2.6, 2.7 and pypy.
    # Passes Python2.7's test suite and incorporates all the latest updates.

    try:
        from thread import get_ident as _get_ident
    except ImportError:
        from dummy_thread import get_ident as _get_ident

    try:
        from _abcoll import KeysView, ValuesView, ItemsView
    except ImportError:
        pass


    class OrderedDict(dict):
        'Dictionary that remembers insertion order'
        # An inherited dict maps keys to values.
        # The inherited dict provides __getitem__, __len__, __contains__, and get.
        # The remaining methods are order-aware.
        # Big-O running times for all methods are the same as for regular dictionaries.

        # The internal self.__map dictionary maps keys to links in a doubly linked list.
        # The circular doubly linked list starts and ends with a sentinel element.
        # The sentinel element never gets deleted (this simplifies the algorithm).
        # Each link is stored as a list of length three:  [PREV, NEXT, KEY].

        def __init__(self, *args, **kwds):
            '''Initialize an ordered dictionary.  Signature is the same as for
            regular dictionaries, but keyword arguments are not recommended
            because their insertion order is arbitrary.

            '''
            if len(args) > 1:
                raise TypeError('expected at most 1 arguments, got %d' % len(args))
            try:
                self.__root
            except AttributeError:
                self.__root = root = []                     # sentinel node
                root[:] = [root, root, None]
                self.__map = {}
            self.__update(*args, **kwds)

        def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
            'od.__setitem__(i, y) <==> od[i]=y'
            # Setting a new item creates a new link which goes at the end of the linked
            # list, and the inherited dictionary is updated with the new key/value pair.
            if key not in self:
                root = self.__root
                last = root[0]
                last[1] = root[0] = self.__map[key] = [last, root, key]
            dict_setitem(self, key, value)

        def __delitem__(self, key, dict_delitem=dict.__delitem__):
            'od.__delitem__(y) <==> del od[y]'
            # Deleting an existing item uses self.__map to find the link which is
            # then removed by updating the links in the predecessor and successor nodes.
            dict_delitem(self, key)
            link_prev, link_next, key = self.__map.pop(key)
            link_prev[1] = link_next
            link_next[0] = link_prev

        def __iter__(self):
            'od.__iter__() <==> iter(od)'
            root = self.__root
            curr = root[1]
            while curr is not root:
                yield curr[2]
                curr = curr[1]

        def __reversed__(self):
            'od.__reversed__() <==> reversed(od)'
            root = self.__root
            curr = root[0]
            while curr is not root:
                yield curr[2]
                curr = curr[0]

        def clear(self):
            'od.clear() -> None.  Remove all items from od.'
            try:
                for node in self.__map.itervalues():
                    del node[:]
                root = self.__root
                root[:] = [root, root, None]
                self.__map.clear()
            except AttributeError:
                pass
            dict.clear(self)

        def popitem(self, last=True):
            '''od.popitem() -> (k, v), return and remove a (key, value) pair.
            Pairs are returned in LIFO order if last is true or FIFO order if false.

            '''
            if not self:
                raise KeyError('dictionary is empty')
            root = self.__root
            if last:
                link = root[0]
                link_prev = link[0]
                link_prev[1] = root
                root[0] = link_prev
            else:
                link = root[1]
                link_next = link[1]
                root[1] = link_next
                link_next[0] = root
            key = link[2]
            del self.__map[key]
            value = dict.pop(self, key)
            return key, value

        # -- the following methods do not depend on the internal structure --

        def keys(self):
            'od.keys() -> list of keys in od'
            return list(self)

        def values(self):
            'od.values() -> list of values in od'
            return [self[key] for key in self]

        def items(self):
            'od.items() -> list of (key, value) pairs in od'
            return [(key, self[key]) for key in self]

        def iterkeys(self):
            'od.iterkeys() -> an iterator over the keys in od'
            return iter(self)

        def itervalues(self):
            'od.itervalues -> an iterator over the values in od'
            for k in self:
                yield self[k]

        def iteritems(self):
            'od.iteritems -> an iterator over the (key, value) items in od'
            for k in self:
                yield (k, self[k])

        def update(*args, **kwds):
            '''od.update(E, **F) -> None.  Update od from dict/iterable E and F.

            If E is a dict instance, does:           for k in E: od[k] = E[k]
            If E has a .keys() method, does:         for k in E.keys(): od[k] = E[k]
            Or if E is an iterable of items, does:   for k, v in E: od[k] = v
            In either case, this is followed by:     for k, v in F.items(): od[k] = v

            '''
            if len(args) > 2:
                raise TypeError('update() takes at most 2 positional '
                                'arguments (%d given)' % (len(args),))
            elif not args:
                raise TypeError('update() takes at least 1 argument (0 given)')
            self = args[0]
            # Make progressively weaker assumptions about "other"
            other = ()
            if len(args) == 2:
                other = args[1]
            if isinstance(other, dict):
                for key in other:
                    self[key] = other[key]
            elif hasattr(other, 'keys'):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
            for key, value in kwds.items():
                self[key] = value

        __update = update  # let subclasses override update without breaking __init__

        __marker = object()

        def pop(self, key, default=__marker):
            '''od.pop(k[,d]) -> v, remove specified key and return the corresponding value.
            If key is not found, d is returned if given, otherwise KeyError is raised.

            '''
            if key in self:
                result = self[key]
                del self[key]
                return result
            if default is self.__marker:
                raise KeyError(key)
            return default

        def setdefault(self, key, default=None):
            'od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od'
            if key in self:
                return self[key]
            self[key] = default
            return default

        def __repr__(self, _repr_running={}):
            'od.__repr__() <==> repr(od)'
            call_key = id(self), _get_ident()
            if call_key in _repr_running:
                return '...'
            _repr_running[call_key] = 1
            try:
                if not self:
                    return '%s()' % (self.__class__.__name__,)
                return '%s(%r)' % (self.__class__.__name__, self.items())
            finally:
                del _repr_running[call_key]

        def __reduce__(self):
            'Return state information for pickling'
            items = [[k, self[k]] for k in self]
            inst_dict = vars(self).copy()
            for k in vars(OrderedDict()):
                inst_dict.pop(k, None)
            if inst_dict:
                return (self.__class__, (items,), inst_dict)
            return self.__class__, (items,)

        def copy(self):
            'od.copy() -> a shallow copy of od'
            return self.__class__(self)

        @classmethod
        def fromkeys(cls, iterable, value=None):
            '''OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S
            and values equal to v (which defaults to None).

            '''
            d = cls()
            for key in iterable:
                d[key] = value
            return d

        def __eq__(self, other):
            '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
            while comparison to a regular mapping is order-insensitive.

            '''
            if isinstance(other, OrderedDict):
                return len(self)==len(other) and self.items() == other.items()
            return dict.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

        # -- the following methods are only used in Python 2.7 --

        def viewkeys(self):
            "od.viewkeys() -> a set-like object providing a view on od's keys"
            return KeysView(self)

        def viewvalues(self):
            "od.viewvalues() -> an object providing a view on od's values"
            return ValuesView(self)

        def viewitems(self):
            "od.viewitems() -> a set-like object providing a view on od's items"
            return ItemsView(self)
