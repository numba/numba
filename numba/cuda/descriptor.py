from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CUDATargetOptions(TargetOptions):
    OPTIONS = {}


class CUDATarget(TargetDescriptor):
    def __init__(self):
        self.options = CUDATargetOptions
        # The typing and target contexts are initialized only when needed -
        # this prevents an attempt to load CUDA libraries at import time on
        # systems that might not have them present.
        self._typingctx = None
        self._targetctx = None

    @property
    def typingctx(self):
        if self._typingctx is None:
            self._typingctx = CUDATypingContext()
        return self._typingctx

    @property
    def targetctx(self):
        if self._targetctx is None:
            self._targetctx = CUDATargetContext(self._typingctx)
        return self._targetctx


cuda_target = CUDATarget()
