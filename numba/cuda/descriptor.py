from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}


class CUDATargetDesc(TargetDescriptor):
    options = CPUTargetOptions
    typingctx = CUDATypingContext()
    targetctx = CUDATargetContext(typingctx)
