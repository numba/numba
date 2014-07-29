from __future__ import print_function, division, absolute_import

import collections
import functools
import io
import timeit
import math
import sys
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

import numpy

from numba.config import PYVERSION, MACHINE_BITS


INT_TYPES = (int,)
if PYVERSION < (3, 0):
    INT_TYPES += (long,)


class ConfigOptions(object):
    OPTIONS = ()

    def __init__(self):
        self._enabled = set()

    def set(self, name):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        self._enabled.add(name)

    def unset(self, name):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        self._enabled.discard(name)

    def __getattr__(self, name):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        return name in self._enabled

    def __repr__(self):
        return "Flags(%s)" % ', '.join(str(x) for x in self._enabled)

    def copy(self):
        copy = type(self)()
        copy._enabled = set(self._enabled)
        return copy

    def __eq__(self, other):
        return isinstance(other, ConfigOptions) and other._enabled == self._enabled

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(sorted(self._enabled)))


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
        assert key not in self
        super(UniqueDict, self).__setitem__(key, value)


# def cache(fn):
#     @functools.wraps(fn)
#     def cached_func(self, *args, **kws):
#         if self in cached_func.cache:
#             return cached_func.cache[self]
#         ret = fn(self, *args, **kws)
#         cached_func.cache[self] = ret
#         return ret
#     cached_func.cache = {}
#     def invalidate(self):
#         if self in cached_func.cache:
#             del cached_func.cache[self]
#     cached_func.invalidate = invalidate
#
#     return cached_func


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
    return len(bin(abs(intval))) - 2


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


# Other common python2/3 adaptors
# Copied from Blaze which borrowed from six

IS_PY3 = PYVERSION >= (3, 0)

if IS_PY3:
    def dict_iteritems(d):
        return d.items().__iter__()

    def dict_itervalues(d):
        return d.values().__iter__()

    def dict_values(d):
        return list(d.values())

    def dict_keys(d):
        return list(d.keys())

    def iter_next(it):
        return it.__next__()

    def func_globals(f):
        return f.__globals__

    longint = int

    unicode = str

    StringIO = io.StringIO

else:
    from cStringIO import StringIO

    def dict_iteritems(d):
        return d.iteritems()

    def dict_itervalues(d):
        return d.itervalues()

    def dict_values(d):
        return d.values()

    def dict_keys(d):
        return d.keys()

    def iter_next(it):
        return it.next()

    def func_globals(f):
        return f.func_globals

    longint = long

    unicode = unicode


# Backported from Python 3.4

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
