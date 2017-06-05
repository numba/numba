from __future__ import absolute_import, print_function

from numba import config, types, sigutils
from .base import register_class_type, ClassBuilder, JitMethod


def jitclass(spec):
    """
    A decorator for creating a jitclass.

    **arguments**:

    - spec:
        Specifies the types of each field on this class.
        Must be a dictionary or a sequence.
        With a dictionary, use collections.OrderedDict for stable ordering.
        With a sequence, it must contain 2-tuples of (fieldname, fieldtype).

    **returns**:

    A callable that takes a class object, which will be compiled.
    """
    def wrap(cls):
        if config.DISABLE_JIT:
            return cls
        else:
            return register_class_type(cls, spec, types.ClassType, ClassBuilder)

    return wrap


def jitmethod(signatures=None, **kwargs):
    """
    A decorator for methods in jitclasses.

    It allows user to specify the function signature and other compiler options.
    User can provide the function signature(s) similar to the ``numba.jit``.
    Note, the type for ``self`` (the first argument of the method) should be
    omitted, so that the signature starts with the 2nd argument.  This decorator
    accept the same keyword arguments as ``numba.jit``.  They are propagated
    to ``numba.jit`` under-the-hood.
    """
    if sigutils.is_signature(signatures):
        signatures = [signatures]

    if signatures is None:
        signatures = []

    def wrap(fn):
        norm_sigs = [sigutils.normalize_signature(s) for s in signatures]
        return JitMethod(fn, norm_sigs, **kwargs)

    return wrap

