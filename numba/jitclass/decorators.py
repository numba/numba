from __future__ import absolute_import, print_function
from numba import types
from .base import register_class_type, ClassBuilder
from .immutable import ImmutableClassBuilder


def jitclass(spec, immutable=False):
    if not callable(spec):
        specfn = lambda *args, **kwargs: spec
    else:
        specfn = spec

    def wrap(cls):
        args = ((types.ImmutableClassType, ImmutableClassBuilder)
                if immutable
                else (types.ClassType, ClassBuilder))
        return register_class_type(cls, specfn, *args)

    return wrap
