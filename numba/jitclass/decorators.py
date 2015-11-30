from __future__ import absolute_import, print_function
from numba import types
from .base import register_class_type, ClassBuilder
from .immutable import ImmutableClassBuilder


def jitclass(spec, immutable=False):
    def wrap(cls):
        args = ((types.ImmutableClassType, ImmutableClassBuilder)
                if immutable
                else (types.ClassType, ClassBuilder))
        return register_class_type(cls, spec, *args)

    return wrap
