"""
Stub for early user testing
"""
from __future__ import print_function, division, absolute_import
import warnings
from numba import types, compiler, dispatcher


def autojit(*args, **kws):
    warnings.warn("autojit is deprecated, use jit instead", DeprecationWarning)
    return jit(*args, **kws)


def jit(*args, **kws):
    if isinstance(args[0], (str, tuple, types.Prototype)):
        [sig] = args

        sig = sig if not isinstance(sig, str) else _parse_signature(sig)

        return _jit(sig, kws)
    else:
        [pyfunc] = args
        disp = dispatcher.Overloaded(py_func=pyfunc)
        return disp


def _parse_signature(signature_str):
    # Just eval signature_str using the types submodules as globals
    return eval(signature_str, {}, types.__dict__)

def _jit(sig, kws):
    def wrapper(func):
        disp = dispatcher.Overloaded(py_func=func)
        disp.jit(sig, **kws)
        return disp

    return wrapper


def _read_flags(flags, kws):
    if kws.pop('nopython', False) == False:
        flags.set("enable_pyobject")

    if kws.pop("forceobj", False) == True:
        flags.set("force_pyobject")

    if kws:
        # Unread options?
        raise NameError("Unrecognized options: %s" % k.keys())
