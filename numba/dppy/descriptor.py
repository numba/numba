from __future__ import print_function, division, absolute_import
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from .target import DPPyTargetContext, DPPyTypingContext


class CPUTargetOptions(TargetOptions):
    OPTIONS = {}


class DPPyTargetDesc(TargetDescriptor):
    options = CPUTargetOptions
    typingctx = DPPyTypingContext()
    targetctx = DPPyTargetContext(typingctx)
