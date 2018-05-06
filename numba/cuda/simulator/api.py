'''
Contains CUDA API functions
'''
from __future__ import absolute_import

from contextlib import contextmanager
from .cudadrv.devices import require_context, reset, gpus
from .kernel import FakeCUDAKernel
from numba.typing import Signature
from warnings import warn
from ..args import In, Out, InOut


def select_device(dev=0):
    assert dev == 0, 'Only a single device supported by the simulator'


class stream(object):
    '''
    The stream API is supported in the simulator - however, all execution
    occurs synchronously, so synchronization requires no operation.
    '''
    @contextmanager
    def auto_synchronize(self):
        yield

    def synchronize(self):
        pass


def synchronize():
    pass

def close():
    gpus.closed = True


def declare_device(*args, **kwargs):
    pass


def detect():
    print('Found 1 CUDA devices')
    print('id %d    %20s %40s' % (0, 'SIMULATOR', '[SUPPORTED]'))
    print('%40s: 5.2' % 'compute capability')


def list_devices():
    return gpus


# Events

class Event(object):
    '''
    The simulator supports the event API, but they do not record timing info,
    and all simulation is synchronous. Execution time is not recorded.
    '''
    def record(self, stream=0):
        pass

    def wait(self, stream=0):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, event):
        warn('Simulator timings are bogus')
        return 0.0

event = Event


def jit(fn_or_sig=None, device=False, debug=False, argtypes=None, inline=False, restype=None,
        fastmath=False, link=None):
    if link is not None:
        raise NotImplementedError('Cannot link PTX in the simulator')
    # Check for first argument specifying types - in that case the
    # decorator is not being passed a function
    if fn_or_sig is None or isinstance(fn_or_sig, (str, tuple, Signature)):
        def jitwrapper(fn):
            return FakeCUDAKernel(fn,
                                  device=device,
                                  fastmath=fastmath)
        return jitwrapper
    return FakeCUDAKernel(fn_or_sig, device=device)

autojit = jit


@contextmanager
def defer_cleanup():
    # No effect for simulator
    yield
