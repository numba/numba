from abc import ABC, abstractmethod
from numba.core.registry import TargetRegistry, CPUDispatcher

from threading import local as tls

_active_context = tls()
_active_context_default = 'cpu'


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
    return getattr(_active_context, 'target', _active_context_default)


hardware_registry = TargetRegistry()


class JitDecorator(ABC):

    @abstractmethod
    def __call__(self):
        return NotImplemented


class Target(ABC):
    """ Implements a hardware/pseudo-hardware target """


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
