from __future__ import absolute_import, print_function
from numba import types
from .base import register_class_type, ClassBuilder
from .immutable import ImmutableClassBuilder


def jitclass(spec, immutable=False):
    """
    A decorator for creating a jitclass.

    Args
    ----
    spec:
        Specifies the types of each field on this class.
        Must be a dictionary or a sequence.
        With a dictionary, use collections.OrderedDict for stable ordering.
        With a sequence, it must contain 2-tuples of (fieldname, fieldtype).

    immutable:
        Set to True to create an immutable jitclass that is usable in GPU
        context as a pass-by-value data structure.
    """
    def wrap(cls):
        args = ((types.ImmutableClassType, ImmutableClassBuilder)
                if immutable
                else (types.ClassType, ClassBuilder))
        return register_class_type(cls, spec, *args)

    return wrap
