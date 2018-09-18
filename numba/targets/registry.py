from __future__ import print_function, division, absolute_import

import contextlib

from . import cpu
from .descriptors import TargetDescriptor
from .. import dispatcher, utils, typing

# -----------------------------------------------------------------------------
# Default CPU target descriptors

class _NestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        old_nested = self._typing_context, self._target_context
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            self._typing_context, self._target_context = old_nested


class CPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    _nested = _NestedContext()

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return cpu.CPUContext(self.typing_context)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return typing.Context()

    @property
    def target_context(self):
        """
        The target context for CPU targets.
        """
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    def nested_context(self, typing_context, target_context):
        """
        A context manager temporarily replacing the contexts with the
        given ones, for the current thread of execution.
        """
        return self._nested.nested(typing_context, target_context)


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
