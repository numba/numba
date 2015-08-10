from __future__ import print_function, division, absolute_import
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from .target import HSATargetContext, HSATypingContext


class HSATargetOptions(TargetOptions):
    OPTIONS = {}


class HSATargetDesc(TargetDescriptor):
    options = HSATargetOptions
    typingctx = HSATypingContext()
    targetctx = HSATargetContext(typingctx)
