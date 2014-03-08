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


class Cuda_threadIdx_x(MacroTemplate):
    key = cuda.threadIdx.x


class Cuda_threadIdx_y(MacroTemplate):
    key = cuda.threadIdx.y


class Cuda_threadIdx_z(MacroTemplate):
    key = cuda.threadIdx.z


@intrinsic
class Cuda_threadIdx(AttributeTemplate):
    key = types.Module(cuda.threadIdx)

    def resolve_x(self, mod):
        return types.Macro(Cuda_threadIdx_x)

    def resolve_y(self, mod):
        return types.Macro(Cuda_threadIdx_y)

    def resolve_z(self, mod):
        return types.Macro(Cuda_threadIdx_z)


@intrinsic
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    def resolve_grid(self, mod):
        return types.Macro(Cuda_grid)

    def resolve_threadIdx(self, mod):
        return types.Module(cuda.threadIdx)


intrinsic_global(cuda, types.Module(cuda))
intrinsic_global(cuda.grid, types.Function(Cuda_grid))
intrinsic_global(cuda.threadIdx, types.Module(cuda.threadIdx))
