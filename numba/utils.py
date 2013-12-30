import collections
import functools


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


def cache(fn):
    @functools.wraps(fn)
    def cached_func(self, *args, **kws):
        if self in cached_func.cache:
            return cached_func.cache[self]
        ret = fn(self, *args, **kws)
        cached_func.cache[self] = ret
        return ret
    cached_func.cache = {}
    def invalidate(self):
        if self in cached_func.cache:
            del cached_func.cache[self]
    cached_func.invalidate = invalidate

    return cached_func


def runonce(fn):
    @functools.wraps(fn)
    def inner():
        if not inner._ran:
            fn()
        inner._ran = True
    inner._ran = False
    return inner



def bit_length(intval):
    assert isinstance(intval, int)
    return len(bin(abs(intval))) - 2
