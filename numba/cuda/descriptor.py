from __future__ import print_function, division, absolute_import
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}


class CUDATargetDesc(TargetDescriptor):
    options = CPUTargetOptions
    typingctx = CUDATypingContext()
    targetctx = CUDATargetContext(typingctx)
