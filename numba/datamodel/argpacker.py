from __future__ import print_function, absolute_import

from collections import deque


class ArgPacker(object):
    """
    Compute the position for each high-level typed argument.
    It flattens every composite argument into primitive types.
    It maintains a position map for unflattening the arguments.

    Since struct (esp. nested struct) have specific ABI requirements (e.g.
    alignemnt, pointer address-space, ...) in different architecture (e.g.
    OpenCL, CUDA), flattening composite argument types simplifes the call
    setup from the Python side.  Functions are receiving simple primitive
    types and there are only a handful of these.
    """
    def __init__(self, dmm, fe_args):
        self._dmm = dmm
        self._fe_args = fe_args
        self._nargs = len(fe_args)
        self._dm_args = [self._dmm.lookup(ty) for ty in fe_args]
        argtys = [bt.get_argument_type() for bt in self._dm_args]
        if argtys:
            self._be_args, self._posmap = zip(*_flatten(argtys, mark_empty=True))
        else:
            self._be_args = self._posmap = ()

    def as_arguments(self, builder, values):
        """Flatten all argument values
        """
        if len(values) != self._nargs:
            raise TypeError("invalid number of args")

        if not values:
            return ()

        args = [dm.as_argument(builder, val)
                for dm, val in zip(self._dm_args, values)]

        flattened = list(_flatten(args))
        if flattened:
            args, _ = zip(*flattened)
        else:
            args = ()
        return args

    def from_arguments(self, builder, args):
        """Unflatten all argument values
        """

        valtree = _unflatten(self._posmap, args)
        values = [dm.from_argument(builder, val)
                  for dm, val in zip(self._dm_args, valtree)]

        return values

    def assign_names(self, args, names):
        """Assign names for each flattened argument values.
        """

        valtree = _unflatten(self._posmap, args)
        for aval, aname in zip(valtree, names):
            self._assign_names(aval, aname)

    def _assign_names(self, val_or_nested, name, depth=()):
        if isinstance(val_or_nested, (tuple, list)):
            for pos, aval in enumerate(val_or_nested):
                self._assign_names(aval, name, depth=depth + (pos,))
        else:
            postfix = '.'.join(map(str, depth))
            parts = [name, postfix]
            val_or_nested.name = '.'.join(filter(bool, parts))

    @property
    def argument_types(self):
        """Return a list of LLVM types that are results of flattening
        composite types.
        """
        return tuple(ty for ty in self._be_args if ty != ())


def _unflatten(posmap, flatiter):
    """Rebuild a nested tuple structure
    """
    poss = deque(posmap)
    vals = deque(flatiter)

    res = []
    while poss:
        cur = poss.popleft()
        ptr = res
        for loc in cur[:-1]:
            if loc >= len(ptr):
                ptr.append([])
            ptr = ptr[loc]

        if cur == ():
            ptr.append(())
        else:
            assert len(ptr) == cur[-1]
            ptr.append(vals.popleft())

    assert not vals

    return res


def _flatten(iterable, mark_empty=False):
    """
    Flatten nested iterable of (tuple, list) with position information
    """
    def rec(iterable, indices):
        for i in iterable:
            if isinstance(i, (tuple, list)):
                if len(i) > 0:
                    inner = indices + (0,)
                    for j, k in rec(i, indices=inner):
                        yield j, k
                elif mark_empty:
                    yield i, ()
            else:
                yield i, indices
            indices = indices[:-1] + (indices[-1] + 1,)
    return rec(iterable, indices=(0,))

