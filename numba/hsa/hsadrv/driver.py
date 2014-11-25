"""
HSA driver bridge implementation
"""

from __future__ import absolute_import, print_function, division
import sys
import os
import traceback
import ctypes
import weakref
import functools
import copy
import warnings
import struct
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
                    c_void_p, c_float)
import contextlib
from collections import namedtuple
from numba import utils, servicelib, mviewbuf
from .error import HsaSupportError, HsaDriverError, HsaApiError, HsaWarning
from .drvapi import API_PROTOTYPES
from . import enums, drvapi
from numba import config
from numba.utils import longint as long

def _find_driver():
    envpath = os.environ.get('NUMBA_HSA_DRIVER', None)
    if envpath == '0':
        # Force fail
        _raise_driver_not_found()

    # Determine DLL type
    if struct.calcsize('P') != 8 or sys.platform == 'win32' or sys.platform == 'darwin':
        _raise_platform_not_supported()
    else:
        # Assume to be *nix like and 64 bit
        dlloader = ctypes.CDLL
        dldir = ['/usr/lib', '/usr/lib64']
        dlname = 'libhsa-runtime64.so'

    if envpath is not None:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            raise ValueError("NUMBA_HSA_DRIVER %s is not a valid path" %
                             envpath)
        if not os.path.isfile(envpath):
            raise ValueError("NUMBA_HSA_DRIVER %s is not a valid file "
                             "path.  Note it must be a filepath of the .so/"
                             ".dll/.dylib or the driver" % envpath)
        candidates = [envpath]
    else:
        # First search for the name in the default library path.
        # If that is not found, try the specific path.
        candidates = [dlname] + [os.path.join(x, dlname) for x in dldir]

    # Load the driver; Collect driver error information
    path_not_exist = []
    driver_load_error = []

    for path in candidates:
        try:
            dll = dlloader(path)
        except OSError as e:
            # Problem opening the DLL
            path_not_exist.append(not os.path.isfile(path))
            driver_load_error.append(e)
        else:
            return dll

    # Problem loading driver
    if all(path_not_exist):
        _raise_driver_not_found()
    else:
        errmsg = '\n'.join(str(e) for e in driver_load_error)
        _raise_driver_error(errmsg)


PLATFORM_NOT_SUPPORTED_ERROR = """
HSA is not currently ussported in this platform ({0}).
"""
def _raise_platform_not_supported():
    raise HsaSupportError(PLATFORM_NOT_SUPPORTED_ERROR.format(sys.platform))


DRIVER_NOT_FOUND_MSG = """
The HSA runtime library cannot be found.  

If you are sure that the HSA is installed, try setting environment
variable NUMBA_HSA_DRIVER with the file path of the HSA runtime shared
library.
"""
def _raise_driver_not_found():
    raise HsaSupportError(DRIVER_NOT_FOUND_MSG)


DRIVER_LOAD_ERROR_MSG = """
A HSA runtime library was found, but failed to load with error:
%s
"""

def _raise_driver_error(e):
    raise HsaSupportError(DRIVER_LOAD_ERROR_MSG % e)


def _build_reverse_error_warn_maps():
    err_map = utils.UniqueDict()
    warn_map = utils.UniqueDict()

    for name in [name for name in dir(enums) if name.startswith('HSA_')]:
        code = getattr(enums, name)
        if 'STATUS_ERROR' in name:
            err_map[code] = name
        elif 'STATUS_INFO' in name:
            warn_map[code] = name
        else:
            pass # should we warn here?
    return err_map, warn_map

ERROR_MAP, WARN_MAP = _build_reverse_error_warn_maps()

def _check_error(fname, retcode):
    def _check_error(self, fname, retcode):
        if retcode != enums.HSA_STATUS_SUCCESS:
            if retcode >= enums.HSA_STATUS_ERROR:
                errname = ERROR_MAP.get(retcode, "UNKNOWN_HSA_ERROR")
                msg = "Call to %s results in %s" % (fname, errname)
                raise HsaApiError(retcode, msg)
            else:
                warn_name = WARN_MAP.get(retcode, "UNKNOWN_HSA_INFO")
                msg = "Call to {0} returned {1}".format(fname, warn_name)
                warnings.warn(msg, HsaWarning)

MISSING_FUNCTION_ERRMSG = """driver missing function: %s.
"""

# The Driver ###########################################################

class Driver(object):
    """
    Driver API functions are lazily bound.
    """
    _singleton = None

    def __new__(cls):
        obj = cls._singleton
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            cls._singleton = obj
        return obj

    def __init__(self):
        try:
            if config.DISABLE_HSA:
                raise HsaSupportError("HSA disabled by user")
            self.lib = _find_driver()
        except HsaSupportError as e:
            self.is_initialized = True
            self.initialization_error = e

    def __del__(self):
        if self.is_initialized and self_initialization_error is None:
            self.hsa_shut_down()

    def initialize(self):
        self.is_initialized = True
        try:
            self.hsa_init()
            print ("hsa has been initialized!!!!")
        except CudaAPIError as e:
            self.initialization_error = e
            raise HsaSupportError("Error at driver init: \n%s:" % e)

    @property
    def is_available(self):
        if not self.is_initialized:
            self.initialize()
        return self.initialization_error is None

    def __getattr__(self, fname):
        # First request of a driver API function
        try:
            proto = API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)
        restype = proto[0]
        argtypes = proto[1:]

        # Initialize driver
        if not self.is_initialized:
            self.initialize()

        if self.initialization_error is not None:
            raise HsaSupportError("Error at driver init: \n%s:" %
                                  self.initialization_error)

        # Find function in driver library
        libfn = self._find_api(fname)
        libfn.restype = restype
        libfn.argtypes = argtypes

        @functools.wraps(libfn)
        def safe_cuda_api_call(*args):
            retcode = libfn(*args)
            self._check_error(fname, retcode)

        setattr(self, fname, safe_cuda_api_call)
        return safe_cuda_api_call

    def _find_api(self, fname):
        # Try version 2
        try:
            return getattr(self.lib, fname + "_v2")
        except AttributeError:
            pass

        # Try regular
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        # Not found.
        # Delay missing function error to use
        def absent_function(*args, **kws):
            raise CudaDriverError(MISSING_FUNCTION_ERRMSG % fname)

        setattr(self, fname, absent_function)
        return absent_function

    '''
    def get_device(self, devnum=0):
        dev = self.devices.get(devnum)
        if dev is None:
            dev = Device(devnum)
            self.devices[devnum] = dev
        return weakref.proxy(dev)

    def get_device_count(self):
        count = c_int()
        self.cuDeviceGetCount(byref(count))
        return count.value

    def list_devices(self):
        """Returns a list of active devices
        """
        return list(self.devices.values())

    def reset(self):
        """Reset all devices
        """
        for dev in self.devices.values():
            dev.reset()
    '''

driver = Driver()

