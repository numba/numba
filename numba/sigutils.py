from __future__ import print_function, division, absolute_import

from numba import types, typing


def is_signature(sig):
    """
    Return whether *sig* is a valid signature specification (for user-facing
    APIs).
    """
    return isinstance(sig, (str, tuple, typing.Signature))


def normalize_signature(sig):
    """
    From *sig* (a signature specification), return a ``(return_type, args)``
    tuple, where ``args`` itself is a tuple of types, and ``return_type``
    can be None if not specified.
    """
    if isinstance(sig, str):
        return normalize_signature(parse_signature(sig))
    elif isinstance(sig, tuple):
        return sig, None
    elif isinstance(sig, typing.Signature):
        return sig.args, sig.return_type
    else:
        raise TypeError(type(sig))


def parse_signature(signature_str):
    # Just eval signature_str using the types submodules as globals
    return eval(signature_str, {}, types.__dict__)

