from __future__ import print_function, division, absolute_import
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions

from numba.core import dispatcher, utils, typing
from .target import DPPLTargetContext, DPPLTypingContext

from numba.core.cpu import CPUTargetOptions


class DPPLTarget(TargetDescriptor):
    options = CPUTargetOptions
    #typingctx = DPPLTypingContext()
    #targetctx = DPPLTargetContext(typingctx)

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPLTargetContext(self.typing_context)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DPPLTypingContext()

    @property
    def target_context(self):
        """
        The target context for DPPL targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for DPPL targets.
        """
        return self._toplevel_typing_context



# The global DPPL target
dppl_target = DPPLTarget()
