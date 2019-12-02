from __future__ import print_function, division, absolute_import
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from .target import OneAPITargetContext, OneAPITypingContext


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}


class OneAPITargetDesc(TargetDescriptor):
    options = CPUTargetOptions
    typingctx = OneAPITypingContext()
    targetctx = OneAPITargetContext(typingctx)
