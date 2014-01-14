from __future__ import print_function, division, absolute_import
import collections
import functools
from timeit import default_timer as timer
import numpy


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
        return (k for k, v in self._values)


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
    def __init__(self, func, records):
        self.func = func
        self.records = records
        self.mean = numpy.mean(self.records)
        self.best = numpy.min(self.records)
        self.worst = numpy.max(self.records)

    def __repr__(self):
        name = getattr(self.func, "__name__", self.func)
        args = (name, format_time(self.mean), format_time(self.best),
                format_time(self.worst), len(self.records))
        return "%20s | mean %7s | best %7s | worst %7s | repeat %d" % args


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


def benchmark(func, maxsec=.1, maxct=1000000):
    total = 0
    records = []

    while True:
        ts = timer()
        func()
        te = timer()
        dur = te - ts
        records.append(dur)
        total += dur
        if total > maxsec:
            break
        if len(records) >= maxct:
            break

    return BenchmarkResult(func, records)
