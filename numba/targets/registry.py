from __future__ import print_function, division, absolute_import
from numba import utils, typing
from numba.targets import cpu
from numba.targets.descriptors import TargetDescriptor
from numba import dispatcher

# -----------------------------------------------------------------------------
# Default CPU target descriptors


class CPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    typing_context = typing.Context()
    target_context = cpu.CPUContext(typing_context)


class CPUOverloaded(dispatcher.Overloaded):
    targetdescr = CPUTarget()


target_registry = utils.UniqueDict()
target_registry['cpu'] = CPUOverloaded
