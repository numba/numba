from __future__ import print_function, absolute_import, division
import math
import warnings

from numba.targets.imputils import Registry
from numba import types
from numba.itanium_mangler import mangle
from .hsaimpl import _declare_function

registry = Registry()
lower = registry.lower

# -----------------------------------------------------------------------------

_unary_b_f = types.int32(types.float32)
_unary_b_d = types.int32(types.float64)
_unary_f_f = types.float32(types.float32)
_unary_d_d = types.float64(types.float64)
_binary_f_ff = types.float32(types.float32, types.float32)
_binary_d_dd = types.float64(types.float64, types.float64)

function_descriptors = {
    'isnan': (_unary_b_f, _unary_b_d),
    'isinf': (_unary_b_f, _unary_b_d),

    'ceil': (_unary_f_f, _unary_d_d),
    'floor': (_unary_f_f, _unary_d_d),

    'fabs': (_unary_f_f, _unary_d_d),

    'sqrt': (_unary_f_f, _unary_d_d),
    'exp': (_unary_f_f, _unary_d_d),
    'expm1': (_unary_f_f, _unary_d_d),
    'log': (_unary_f_f, _unary_d_d),
    'log10': (_unary_f_f, _unary_d_d),
    'log1p': (_unary_f_f, _unary_d_d),

    'sin': (_unary_f_f, _unary_d_d),
    'cos': (_unary_f_f, _unary_d_d),
    'tan': (_unary_f_f, _unary_d_d),
    'asin': (_unary_f_f, _unary_d_d),
    'acos': (_unary_f_f, _unary_d_d),
    'atan': (_unary_f_f, _unary_d_d),
    'sinh': (_unary_f_f, _unary_d_d),
    'cosh': (_unary_f_f, _unary_d_d),
    'tanh': (_unary_f_f, _unary_d_d),
    'asinh': (_unary_f_f, _unary_d_d),
    'acosh': (_unary_f_f, _unary_d_d),
    'atanh': (_unary_f_f, _unary_d_d),

    'copysign': (_binary_f_ff, _binary_d_dd),
    'atan2': (_binary_f_ff, _binary_d_dd),
    'pow': (_binary_f_ff, _binary_d_dd),
    'fmod': (_binary_f_ff, _binary_d_dd),

    'erf': (_unary_f_f, _unary_d_d),
    'erfc': (_unary_f_f, _unary_d_d),
    'gamma': (_unary_f_f, _unary_d_d),
    'lgamma': (_unary_f_f, _unary_d_d),

    # unsupported functions listed in the math module documentation:
    # frexp, ldexp, trunc, modf, factorial, fsum
}


# some functions may be named differently by the underlying math
# library as oposed to the Python name.
_lib_counterpart = {
    'gamma': 'tgamma'
}


def _mk_fn_decl(name, decl_sig):
    sym = _lib_counterpart.get(name, name)

    def core(context, builder, sig, args):
        fn = _declare_function(context, builder, sym, decl_sig, decl_sig.args,
                               mangler=mangle)
        res = builder.call(fn, args)
        return context.cast(builder, res, decl_sig.return_type, sig.return_type)

    core.__name__ = name
    return core


_supported = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh',
              'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'isnan', 'isinf',
              'ceil', 'floor', 'fabs', 'sqrt', 'exp', 'expm1', 'log',
              'log10', 'log1p', 'copysign', 'pow', 'fmod', 'erf', 'erfc',
              'gamma', 'lgamma',
              ]

for name in _supported:
    sigs = function_descriptors.get(name)
    if sigs is None:
        warnings.warn("HSA - failed to register '{0}'".format(name))
        continue

    try:
        # only symbols present in the math module
        key = getattr(math, name)
    except AttributeError:
        continue

    for sig in sigs:
        fn = _mk_fn_decl(name, sig)
        lower(key, *sig.args)(fn)
