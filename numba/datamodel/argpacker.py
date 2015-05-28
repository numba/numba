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
        self._unflatten_code = _build_unflatten_code(argtys)
        self._be_args = list(_flatten(argtys))

    def as_arguments(self, builder, values):
        """Flatten all argument values
        """
        if len(values) != self._nargs:
            raise TypeError("invalid number of args")

        if not values:
            return ()

        args = [dm.as_argument(builder, val)
                for dm, val in zip(self._dm_args, values)]

        args = tuple(_flatten(args))
        return args

    def from_arguments(self, builder, args):
        """Unflatten all argument values
        """

        valtree = _unflatten(self._unflatten_code, args)
        values = [dm.from_argument(builder, val)
                  for dm, val in zip(self._dm_args, valtree)]

        return values

    def assign_names(self, args, names):
        """Assign names for each flattened argument values.
        """

        valtree = _unflatten(self._unflatten_code, args)
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


def _flatten(iterable):
    """
    Flatten nested iterable of (tuple, list).
    """
    def rec(iterable):
        for i in iterable:
            if isinstance(i, (tuple, list)):
                for j in rec(i):
                    yield j
            else:
                yield i
    return rec(iterable)


_PUSH_LIST = 1
_APPEND_NEXT_VALUE = 2
_APPEND_EMPTY_TUPLE = 3
_POP = 4


def _build_unflatten_code(iterable):
    """Build an unflatten opcode sequence for the given *iterable* structure
    (an iterable of nested sequences).
    """
    code = []
    def rec(iterable):
        for i in iterable:
            if isinstance(i, (tuple, list)):
                if len(i) > 0:
                    code.append(_PUSH_LIST)
                    rec(i)
                    code.append(_POP)
                else:
                    code.append(_APPEND_EMPTY_TUPLE)
            else:
                code.append(_APPEND_NEXT_VALUE)

    rec(iterable)
    return code


def _unflatten(code, flatiter):
    """Rebuild a nested tuple structure using the given opcode sequence.
    """
    vals = deque(flatiter)

    res = []
    cur = res
    stack = []
    for op in code:
        if op is _PUSH_LIST:
            stack.append(cur)
            cur.append([])
            cur = cur[-1]
        elif op is _APPEND_NEXT_VALUE:
            cur.append(vals.popleft())
        elif op is _APPEND_EMPTY_TUPLE:
            cur.append(())
        elif op is _POP:
            cur = stack.pop()

    assert not stack, stack

    return res
