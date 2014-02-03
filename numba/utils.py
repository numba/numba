from __future__ import print_function, division, absolute_import
import collections
import functools
import timeit
import math
import numpy
from numba.config import PYVERSION


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
    assert isinstance(intval, int)
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
    result = timer.repeat(3, number)
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

    def longint(v):
        return int(v)

else:
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

    def longint(v):
        return long(v)