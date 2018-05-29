from __future__ import absolute_import, print_function

from numba import types
from .templates import AttributeTemplate, infer_getattr


@infer_getattr
class LowLevelCallableAttribute(AttributeTemplate):
    key = types.LowLevelCallable

    def resolve_function(self, llc):
        return llc[1]
