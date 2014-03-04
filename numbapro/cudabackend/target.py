from __future__ import print_function, absolute_import

from numba import typing
from numba.targets.base import BaseContext
from numba.targets.options import TargetOptions
from . import cudadecl
from . import cudaimpl


# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.Context):
    def __init__(self):
        super(CUDATypingContext, self).__init__()
        # Load CUDA intrinsics
        for ftcls in cudadecl.INTR_FUNCS:
            self.insert_function(ftcls(self))
        for ftcls in cudadecl.INTR_ATTRS:
            self.insert_attributes(ftcls(self))
        for gv, gty in cudadecl.INTR_GLOBALS:
            self.insert_global(gv, gty)


# -----------------------------------------------------------------------------
# Implementation

class CUDATargetContext(BaseContext):
    def init(self):
        self.insert_func_defn(cudaimpl.FUNCTIONS)

    def get_executable(self, func, fndesc):
        print(func)
        raise NotImplementedError


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}
