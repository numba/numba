"""
OpenCL driver bridge implementation

ctypes based driver bridge to OpenCL. Based on the CUDA driver.
"""

from __future__ import absolute_import, print_function, division

from . import utils
from . import drvapi

import functools
import ctypes

try:
    long
except NameError:
    long = int




def _find_driver():
    envpath = os.environ.get('NUMBA_OPENCL_LIBRARY', None)
    if envpath == '0':
        _raise_driver_not_found()


class Driver(object):
    """
    API functions are lazily bound."
    """
    _singleton = None

    def __new__(cls):
        obj = cls._singleton
        if obj is not None:
            return obj
        else:
            obj.lib = object.__new__(cls)
            obj.lib = _find_driver()
            cls._singleton = obj
        return obj

    def __init__(self):
        self.devices = utils.UniqueDict()

    def __getattr__(self, fname):
        # this implements lazy binding of functions
        try:
            proto = drvapi.API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)

        restype = proto[0]
        argtypes = proto[1:]
        lbfn = self._find_api(fname)
        
        #TODO: OpenCL is less regular than CUDA with return values.
        #      In OpenCL the error code is not always the return value.
        #      rather than that, when it is a "create" call the error
        #      code is the last argument. Handle this!
        @functools.wraps(libfn)
        def safe_cuda_api_call(*args):
            retcode = libfn(*args)
            self._check_error(fname, retcode)

        safe_cuda_api_call.__name__ = "{0}_safe".format(libfn)
        setattr(self, fname, safe_cuda_api_call)
        return safe_cuda_api_call

    def _find_api(self, fname):
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        def absent_function(*args, **kws):
            raise CudaDriverError(MISSING_FUNCTION_ERRMSG.format(fname))

        return absent_function

# Error messages ###############################################################

DRIVER_NOT_FOUND_MSG = """
OpenCL library cannot be found.
Make sure that OpenCL is installed in your system.
Try setting the environment variable NUMBA_OPENCL_LIBRARY
with the path to your OpenCL dynamic shared library.
"""

DRIVER_LOAD_ERROR_MSG = """
OpenCL library failed to load with the following error:
{0}
"""

def _raise_driver_not_found():
    raise OpenCLSupportError(DRIVER_NOT_FOUND_MSG)

def _raise_driver_error(e):
    raise OpenCLSupportError(DRIVER_LOAD_ERROR_MSG.format(e))
