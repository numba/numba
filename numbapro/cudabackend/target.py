from __future__ import print_function, absolute_import

from numba import typing
from numba.targets.base import BaseContext
from numba.targets.options import TargetOptions
from .import intrinsics

# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.Context):
    def __init__(self):
        super(CUDATypingContext, self).__init__()
        # Load CUDA intrinsics
        for ftcls in intrinsics.INTR_FUNCS:
            self.insert_function(ftcls(self))
        for ftcls in intrinsics.INTR_ATTRS:
            self.insert_attributes(ftcls(self))
        for gv, gty in intrinsics.INTR_GLOBALS:
            self.insert_global(gv, gty)


# -----------------------------------------------------------------------------
# Implementation

class CUDATargetContext(BaseContext):
    def init(self):
        pass

    def get_executable(self, func, fndesc):
        raise NotImplementedError


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}
