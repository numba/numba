"""
After a function is jitted it becomes an CPUOverloaded object which defines the target (machine) the function is being built for as well as the typing context. 

The target context defines the machine/OS/available functions and operators for the function being compiled. The target_context is of type cpu.CPUContext (numba/targets/cpu.py) which creates the LLVM IR Module and LLVM EngineBuilder, loads functions from the C math and numpy libraries, and operators into the execution environment. 

The typing context defines the type system for numba. numba/typing directory contains a number of files that build the type system. 

"""

from __future__ import print_function, division, absolute_import
from numba import utils, typing
from numba.targets import cpu
from numba.targets.descriptors import TargetDescriptor
from numba import dispatcher

# -----------------------------------------------------------------------------
# Default CPU target descriptors


class CPUTarget(TargetDescriptor):
    """
    Sets the options, and typing and target context for a given CPU
    """
    options = cpu.CPUTargetOptions
    typing_context = typing.Context()
    target_context = cpu.CPUContext(typing_context)


class CPUOverloaded(dispatcher.Overloaded):
    """
    Jitted functions are CPUOverloaded objects. A wrapper to CPUTarget.
    """
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
