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
import traceback
from types import ModuleType
import numpy as np

from .six import *
try:
    # preferred over pure-python StringIO due to threadsafety
    # note: parallel write to StringIO could cause data to go missing
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from numba.config import PYVERSION, MACHINE_BITS


IS_PY3 = PYVERSION >= (3, 0)

if IS_PY3:
    import builtins
    INT_TYPES = (int,)
    longint = int
    get_ident = threading.get_ident
    intern = sys.intern
    file_replace = os.replace
    asbyteint = int

    def erase_traceback(exc_value):
        """
        Erase the traceback and hanging locals from the given exception instance.
        """
        if exc_value.__traceback__ is not None:
            traceback.clear_frames(exc_value.__traceback__)
        return exc_value.with_traceback(None)

else:
    import thread
    import __builtin__ as builtins
    INT_TYPES = (int, long)
    longint = long
    get_ident = thread.get_ident
    intern = intern
    def asbyteint(x):
        # convert 1-char str into int
        return ord(x)

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

    def erase_traceback(exc_value):
        """
        Erase the traceback and hanging locals from the given exception instance.
        """
        return exc_value

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
        import singledispatch
    except ImportError:
        raise ImportError("please install the 'singledispatch' package "
                          "('pip install singledispatch')")
    else:
        # Hotfix for https://bitbucket.org/ambv/singledispatch/issues/8/inconsistent-hierarchy-with-enum
        def _c3_merge(sequences):
            """Merges MROs in *sequences* to a single MRO using the C3 algorithm.

            Adapted from http://www.python.org/download/releases/2.3/mro/.

            """
            result = []
            while True:
                sequences = [s for s in sequences if s]   # purge empty sequences
                if not sequences:
                    return result
                for s1 in sequences:   # find merge candidates among seq heads
                    candidate = s1[0]
                    for s2 in sequences:
                        if candidate in s2[1:]:
                            candidate = None
                            break      # reject the current head, it appears later
                    else:
                        break
                if candidate is None:
                    raise RuntimeError("Inconsistent hierarchy")
                result.append(candidate)
                # remove the chosen candidate
                for seq in sequences:
                    if seq[0] == candidate:
                        del seq[0]

        singledispatch._c3_merge = _c3_merge

        singledispatch = singledispatch.singledispatch


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
    # This one has its args reversed!
    ('contains', '', 'in'),
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

    def _check_attr(self, name):
        if name not in self.OPTIONS:
            raise AttributeError("Invalid flag: %s" % name)

    def __getattr__(self, name):
        self._check_attr(name)
        return self._values[name]

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(ConfigOptions, self).__setattr__(name, value)
        else:
            self._check_attr(name)
            self._values[name] = value

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


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise AssertionError("key already in dictionary: %r" % (key,))
        super(UniqueDict, self).__setitem__(key, value)


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


def stream_list(lst):
    """
    Given a list, return an infinite iterator of iterators.
    Each iterator iterates over the list from the last seen point up to
    the current end-of-list.

    In effect, each iterator will give the newly appended elements from the
    previous iterator instantiation time.
    """
    def sublist_iterator(start, stop):
        return iter(lst[start:stop])

    start = 0
    while True:
        stop = len(lst)
        yield sublist_iterator(start, stop)
        start = stop


class BenchmarkResult(object):
    def __init__(self, func, records, loop):
        self.func = func
        self.loop = loop
        self.records = np.array(records) / loop
        self.best = np.min(self.records)

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


def logger_hasHandlers(logger):
    # Backport from python3.5 logging implementation of `.hasHandlers()`
    c = logger
    rv = False
    while c:
        if c.handlers:
            rv = True
            break
        if not c.propagate:
            break
        else:
            c = c.parent
    return rv


# A dummy module for dynamically-generated functions
_dynamic_modname = '<dynamic>'
_dynamic_module = ModuleType(_dynamic_modname)
_dynamic_module.__builtins__ = moves.builtins


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
            atexit.register(_at_shutdown)
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


# dummy invocation to force _at_shutdown() to be registered
finalize(lambda: None, lambda: None)
assert finalize._registered_with_atexit
