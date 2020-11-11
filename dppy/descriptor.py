from __future__ import print_function, division, absolute_import
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions

from numba.core import dispatcher, utils, typing
from .target import DPPyTargetContext, DPPyTypingContext

from numba.core.cpu import CPUTargetOptions


class DPPyTarget(TargetDescriptor):
    options = CPUTargetOptions
    #typingctx = DPPyTypingContext()
    #targetctx = DPPyTargetContext(typingctx)

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPyTargetContext(self.typing_context)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DPPyTypingContext()

    @property
    def target_context(self):
        """
        The target context for DPPy targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for DPPy targets.
        """
        return self._toplevel_typing_context



# The global DPPy target
dppy_target = DPPyTarget()
