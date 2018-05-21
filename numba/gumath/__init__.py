import logging
import inspect
import functools

from .ndtypes import Function
from .gufunc import GuFunc


def jit_xnd(fn_or_types=None, *, types=tuple(), **kwargs):
    if not callable(fn_or_types):
        return functools.partial(jit_xnd, types=fn_or_types, **kwargs)
    fn = fn_or_types
    g = GuFunc(fn, **kwargs)
    if isinstance(types, str) or isinstance(types, Function):
        types = [types]
    for t in types:
        if isinstance(t, str):
            t = Function.from_ndt(t)
        g.add(t)
    if types:
        return g.func
    return g

