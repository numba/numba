from __future__ import print_function, division, absolute_import
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from .target import OCLTargetContext, OCLTypingContext


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}


class OCLTargetDesc(TargetDescriptor):
    options = CPUTargetOptions
    typingctx = OCLTypingContext()
    targetctx = OCLTargetContext(typingctx)
