from abc import ABC, abstractmethod
from numba.core.registry import TargetRegistry, CPUDispatcher

from threading import local as tls

_active_context = tls()
_active_context_default = 'cpu'


class HardwareRegistry(TargetRegistry):

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            msg = "No target is registered against '{}', known targets:\n{}"
            known = '\n'.join([f"{k: <{10}} -> {v}"
                               for k, v in hardware_registry.items()])
            raise ValueError(msg.format(item, known)) from None


hardware_registry = HardwareRegistry()


class hardware_target(object):
    def __init__(self, name):
        self._orig_target = getattr(_active_context, 'target',
                                    _active_context_default)
        self.target = name

    def __enter__(self):
        _active_context.target = self.target

    def __exit__(self, ty, val, tb):
        _active_context.target = self._orig_target


def current_target():
    """Returns the current target
    """
    return getattr(_active_context, 'target', _active_context_default)


def get_local_target(context):
    """
    Gets the local target from the call stack if available and the TLS
    override if not.
    """
    # TODO: Should this logic be reversed to prefer TLS override?
    if len(context.callstack._stack) > 0:
        target = context.callstack[0].target
    else:
        target = hardware_registry.get(current_target(), None)
    if target is None:
        msg = ("The hardware target found is not registered."
               "Given target was {}.")
        raise ValueError(msg.format(target))
    else:
        return target


def resolve_target_str(target_str):
    """Resolves a target specified as a string to its Target class."""
    return hardware_registry[target_str]


def resolve_dispatcher_from_str(target_str):
    """Returns the dispatcher associated with a target hardware string"""
    target_hw = resolve_target_str(target_str)
    return dispatcher_registry[target_hw]


class JitDecorator(ABC):

    @abstractmethod
    def __call__(self):
        return NotImplemented


class Target(ABC):
    """ Implements a hardware/pseudo-hardware target """

    @classmethod
    def inherits_from(cls, other):
        """Returns True if this target inherits from 'other' False otherwise"""
        return issubclass(cls, other)


class Generic(Target):
    """Mark the hardware target as generic, i.e. suitable for compilation on
    any target. All hardware must inherit from this.
    """


class CPU(Generic):
    """Mark the hardware target as CPU.
    """


class GPU(Generic):
    """Mark the hardware target as GPU, i.e. suitable for compilation on a GPU
    target.
    """


class CUDA(GPU):
    """Mark the hardware target as CUDA.
    """


class ROCm(GPU):
    """Mark the hardware target as ROCm.
    """


class NPyUfunc(Target):
    """Mark the hardware target as a ufunc
    """


hardware_registry['generic'] = Generic
hardware_registry['CPU'] = CPU
hardware_registry['cpu'] = CPU
hardware_registry['GPU'] = GPU
hardware_registry['gpu'] = GPU
hardware_registry['CUDA'] = CUDA
hardware_registry['cuda'] = CUDA
hardware_registry['ROCm'] = ROCm
hardware_registry['npyufunc'] = NPyUfunc

dispatcher_registry = TargetRegistry(key_type=Target)
dispatcher_registry[hardware_registry['cpu']] = CPUDispatcher
