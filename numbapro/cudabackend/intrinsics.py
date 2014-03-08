from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, MacroTemplate,
                                    signature)
from numbapro import cuda

INTR_FUNCS = []
INTR_ATTRS = []
INTR_GLOBALS = []

def intrinsic(template):
    if issubclass(template, AttributeTemplate):
        INTR_ATTRS.append(template)
    else:
        INTR_FUNCS.append(template)
    return template


def intrinsic_global(v, t):
    INTR_GLOBALS.append((v, t))

# -----------------------------------------------------------------------------

class Cuda_grid(MacroTemplate):
    key = cuda.grid

    def expand(self, args, kws):
        assert not kws
        print(args)
        raise NotImplementedError


@intrinsic
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    def resolve_grid(self, mod):
        return types.Macro(Cuda_grid)


intrinsic_global(cuda, types.Module(cuda))
intrinsic_global(cuda.grid, types.Function(Cuda_grid))
