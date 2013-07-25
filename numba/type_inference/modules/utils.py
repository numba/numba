# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import functools
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from numba import error
from numba.type_inference.module_type_inference import (register,
                                                        register_inferer,
                                                        register_unbound,
                                                        register_value)


def expect_n_args(node, name, nargs):
    if not isinstance(nargs, tuple):
        nargs = (nargs,)

    if len(node.args) not in nargs:
        expected = " or ".join(map(str, nargs))
        raise error.NumbaError(
            node, "builtin %s expects %s arguments" % (name,
                                                       expected))

def register_with_argchecking(nargs, can_handle_deferred_types=False):
    if nargs is not None and not isinstance(nargs, tuple):
        nargs = (nargs,)

    def decorator(func, value=None):
        @functools.wraps(func)
        def infer(context, node, *args):
            if nargs is not None:
                expect_n_args(node, name, nargs)
                need_nones = max(nargs) - len(args)
                args += (None,) * need_nones

            return func(context, node, *args)

        if value is None:
            name = infer.__name__.strip("_")
            if name == 'datetime':
                import datetime
                value = datetime.datetime
            else:
                value = getattr(builtins, name)
        else:
            name = getattr(value, "__name__", "<unknown>")

        register_value(value, infer, pass_in_types=False, pass_in_callnode=True,
                       can_handle_deferred_types=can_handle_deferred_types)

        return func # wrapper

    return decorator
