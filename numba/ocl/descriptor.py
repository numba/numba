from __future__ import print_function, division, absolute_import
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from .target import OCLTargetContext, OCLTypingContext


class OCLTargetOptions(TargetOptions):
    OPTIONS = {}


class OCLTargetDesc(TargetDescriptor):
    options = OCLTargetOptions
    typingctx = OCLTypingContext()
    targetctx = OCLTargetContext(typingctx)
