"""
Hints to wrap Kernel arguments to indicate how to manage host-device
memory transfers before & after the kernel call.
"""
import abc

from numba import config
from six import add_metaclass
from numba.typing.typeof import typeof, Purpose


@add_metaclass(abc.ABCMeta)
class ArgHint:
    def __init__(self, value):
        self.value = value

    @abc.abstractmethod
    def to_device(self, retr, stream=0):
        """
        :param stream: a stream to use when copying data
        :param retr:
            a list of clean-up work to do after the kernel's been run.
            Append 0-arg lambdas to it!
        :return: a value (usually an `DeviceNDArray`) to be passed to
            the kernel
        """
        pass

    @property
    def _numba_type_(self):
        return typeof(self.value, Purpose.argument)


class In(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, _ = auto_device(
            self.value,
            stream=stream)
        return devary


class Out(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, conv = auto_device(
            self.value,
            copy=False,
            stream=stream)
        if conv:
            retr.append(lambda: devary.copy_to_host(self.value, stream=stream))
        return devary


class InOut(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, conv = auto_device(
            self.value,
            stream=stream)
        if conv:
            retr.append(lambda: devary.copy_to_host(self.value, stream=stream))
        return devary


# whitelist the values that we recognise, as the config strings come from
# the process' environment!
WHITELISTED_ARGHINTS = dict(
    In=In,
    Out=Out,
    InOut=InOut)


def wrap_arg(value):
    if isinstance(value, ArgHint):
        return value
    else:
        return WHITELISTED_ARGHINTS[config.CUDA_DEFAULT_ARGHINT](value)


__all__ = [
    'In',
    'Out',
    'InOut',

    'ArgHint',
    'wrap_arg',
]
