'''
Contains CUDA API functions
'''

# Imports here bring together parts of the API from other modules, so some of
# them appear unused.
from contextlib import contextmanager
from .cudadrv.devices import require_context, reset, gpus  # noqa: F401
from .kernel import FakeCUDAKernel
from numba.core.typing import Signature
from warnings import warn
from ..args import In, Out, InOut  # noqa: F401

import inspect
import logging

# The logger name is not strictly accurate, but makes the simulator more
# accurately reflect the hardware target
_logger = logging.getLogger('numba.cuda.decorators')


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


def jit(func_or_sig=None, device=False, debug=False, argtypes=None,
        inline=False, restype=None, fastmath=False, link=None,
        boundscheck=None,
        ):
    # Here for API compatibility
    if boundscheck:
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if link is not None:
        raise NotImplementedError('Cannot link PTX in the simulator')
    # Check for first argument specifying types - in that case the
    # decorator is not being passed a function
    if func_or_sig is None or isinstance(func_or_sig, (str, tuple, Signature)):
        def jitwrapper(fn):
            return FakeCUDAKernel(fn,
                                  device=device,
                                  fastmath=fastmath)
        return jitwrapper
    return FakeCUDAKernel(func_or_sig, device=device)


def jit_module(**kwargs):
    """ Automatically ``jit``-wraps functions defined in a Python module. By
    default, wrapped functions are treated as device functions rather than
    kernels - pass ``device=False`` to treat functions as kernels.

    Note that ``jit_module`` should be called following the declaration of all
    functions to be jitted. This function may be called multiple times within
    a module with different options, and any new function declarations since
    the previous ``jit_module`` call will be wrapped with the options provided
    to the current call to ``jit_module``.

    Note that only functions which are defined in the module ``jit_module`` is
    called from are considered for automatic jit-wrapping.  See the Numba
    documentation for more information about what can/cannot be jitted.

    :param kwargs: Keyword arguments to pass to ``jit`` such as ``device``
                   or ``opt``.

    """
    if 'device' not in kwargs:
        kwargs['device'] = True

    # Get the module jit_module is being called from
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    # Replace functions in module with jit-wrapped versions
    for name, obj in module.__dict__.items():
        if inspect.isfunction(obj) and inspect.getmodule(obj) == module:
            _logger.debug("Auto decorating function {} from module {} with jit "
                          "and options: {}".format(obj, module.__name__,
                                                   kwargs))
            module.__dict__[name] = jit(obj, **kwargs)


@contextmanager
def defer_cleanup():
    # No effect for simulator
    yield
