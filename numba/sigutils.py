from __future__ import print_function, division, absolute_import

from numba import types, typing


def is_signature(sig):
    return isinstance(sig, (str, tuple, types.Prototype))


def normalize_signature(sig):
    if isinstance(sig, str):
        return normalize_signature(parse_signature(sig))
    elif isinstance(sig, tuple):
        return sig, None
    elif isinstance(sig, types.Prototype):
        return sig.args, sig.return_type
    elif isinstance(sig, typing.Signature):
        return sig.args, sig.return_type
    else:
        raise TypeError(type(sig))


def parse_signature(signature_str):
    # Just eval signature_str using the types submodules as globals
    return eval(signature_str, {}, types.__dict__)

