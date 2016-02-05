from __future__ import print_function, division, absolute_import

from . import cpu
from .descriptors import TargetDescriptor
from .. import dispatcher, utils, typing

# -----------------------------------------------------------------------------
# Default CPU target descriptors


class CPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions

    @utils.cached_property
    def target_context(self):
        return cpu.CPUContext(self.typing_context)

    @utils.cached_property
    def typing_context(self):
        return typing.Context()


# The global CPU target
cpu_target = CPUTarget()


class CPUDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target


class TargetRegistry(utils.UniqueDict):
    """
    A registry of API implementations for various backends.

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


dispatcher_registry = TargetRegistry()
dispatcher_registry['cpu'] = CPUDispatcher
