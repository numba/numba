from __future__ import absolute_import, print_function

from numba import config, types
from .base import register_class_type, ClassBuilder


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
