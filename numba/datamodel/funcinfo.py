from __future__ import print_function, absolute_import

from collections import deque


class FunctionInfo(object):
    def __init__(self, dmm, fe_ret, fe_args):
        self._dmm = dmm
        self._fe_ret = fe_ret
        self._fe_args = fe_args
        self._nargs = len(fe_args)
        self._dm_ret = self._dmm.lookup(fe_ret)
        self._dm_args = [self._dmm.lookup(ty) for ty in fe_args]
        argtys = [bt.get_argument_type() for bt in self._dm_args]
        if len(argtys):
            self._be_args, self._posmap = zip(*_flatten(argtys))
        else:
            self._be_args = self._posmap = ()
        self._be_ret = self._dm_ret.get_return_type()

    def as_arguments(self, builder, values):
        if len(values) != self._nargs:
            raise TypeError("invalid number of args")

        if not values:
            return ()

        args = [dm.as_argument(builder, val)
                for dm, val in zip(self._dm_args, values)]

        args, _ = zip(*_flatten(args))
        return args

    def from_arguments(self, builder, args):
        if len(args) != len(self._posmap):
            raise TypeError("invalid number of args")

        if not args:
            return ()

        valtree = _unflatten(self._posmap, args)
        values = [dm.from_argument(builder, val)
                  for dm, val in zip(self._dm_args, valtree)]

        return values

    @property
    def argument_types(self):
        return tuple(self._be_args)

    @property
    def return_type(self):
        return self._be_ret


def _unflatten(posmap, flatiter):
    poss = deque(posmap)
    vals = deque(flatiter)

    res = []
    while poss:
        assert len(poss) == len(vals)
        cur = poss.popleft()
        ptr = res
        for loc in cur[:-1]:
            if loc >= len(ptr):
                ptr.append([])
            ptr = ptr[loc]

        assert len(ptr) == cur[-1]
        ptr.append(vals.popleft())

    return res


def _flatten(iterable, indices=(0,)):
    """
    Flatten nested iterable of (tuple, list) with position information
    """
    for i in iterable:
        if isinstance(i, (tuple, list)):
            inner = indices + (0,)
            for j, k in _flatten(i, indices=inner):
                yield j, k
        else:
            yield i, indices
        indices = indices[:-1] + (indices[-1] + 1,)

