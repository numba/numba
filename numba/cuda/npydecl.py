from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, MacroTemplate,
                                    signature, Registry)
import numpy
from numba.typing.npydecl import NdEmpty

registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global


@intrinsic_attr
class NumpyModuleAttribute(AttributeTemplate):
    key = types.Module(numpy)

    def resolve_empty(self, mod):
        return types.Function(NdEmpty)


intrinsic_global(numpy.empty, types.Function(NdEmpty))
intrinsic_global(numpy, types.Module(numpy))
