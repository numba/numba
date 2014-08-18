from __future__ import print_function, division, absolute_import

from . import cpu
from .descriptors import TargetDescriptor
from .. import dispatcher, utils, typing

# -----------------------------------------------------------------------------
# Default CPU target descriptors


class CPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    typing_context = typing.Context()
    target_context = cpu.CPUContext(typing_context)


class CPUOverloaded(dispatcher.Overloaded):
    targetdescr = CPUTarget()


class TargetRegistry(utils.UniqueDict):
    """
    Attributes
    ----------
    ondemand:

        A dictionary of target-name -> function, where function is executed
        the first time a target is used.  It is used for deferred
        initialization for some targets (e.g. gpu).
    """
    def __init__(self, *args, **kws):
        super(TargetRegistry, self).__init__(*args, **kws)
        self.ondemand = utils.UniqueDict()

    def __getitem__(self, item):
        if item in self.ondemand:
            self[item] = self.ondemand[item]()
            del self.ondemand[item]
        return super(TargetRegistry, self).__getitem__(item)


target_registry = TargetRegistry()
target_registry['cpu'] = CPUOverloaded
