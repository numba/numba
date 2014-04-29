"""
OpenCL driver bridge implementation

ctypes based driver bridge to OpenCL. Based on the CUDA driver.
"""

from __future__ import absolute_import, print_function, division

from ... import utils
from . import drvapi, enums
from .types import *

import weakref
import functools
import ctypes
import os
import sys

try:
    long
except NameError:
    long = int

def _ctypes_func_wraps(model):
    def inner(func):
        wrapped = functools.wraps(model)(func)
        wrapped.restype = model.restype
        wrapped.argtypes = model.argtypes
        return wrapped
    return inner


def _find_driver():
    envpath = os.environ.get('NUMBA_OPENCL_LIBRARY', None)
    if envpath == '0':
        _raise_driver_not_found()


    if envpath is not None:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            _raise_bad_env_path(envpath)

        if not os.path.isfile(envpath):
            _raise_bad_env_path(envpath)


    if sys.platform == 'win32':
        dll_loader = ctypes.WinDLL
    else:
        dll_loader = ctypes.CDLL

    dll_path = envpath or ctypes.util.find_library("OpenCL")

    if dll_path is None:
        _raise_driver_not_found()

    try:
        dll = dll_loader(dll_path)
    except OSError as e:
        # a bit of logic to better diagnose the problem
        if envpath:
            # came from environment variable
            _raise_bad_env_path(envpath, str(e))

        _raise_driver_error()
      
    return dll


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
            obj = object.__new__(cls)
            obj.lib = _find_driver()
            cls._singleton = obj
        return obj

    def __init__(self):
        count = cl_uint()
        self.clGetPlatformIDs(4096, None, ctypes.byref(count))
        platforms = (ctypes.c_void_p * count.value) ()
        self.clGetPlatformIDs(count, platforms, None)
        self.platforms = [Platform(x, self) for x in platforms]

    def __getattr__(self, fname):
        # this implements lazy binding of functions
        try:
            proto = drvapi.API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)

        libfn = self._find_api(fname)
        libfn.restype = proto[0]
        libfn.argtypes = proto[1:-1]
        error_code_idx = proto[-1]
        if error_code_idx is None:
            return libfn
        elif error_code_idx == 0:
            @_ctypes_func_wraps(libfn)
            def safe_ocl_api_call(*args):
                retcode = libfn(*args)
                if retcode != enums.CL_SUCCESS:
                    _raise_opencl_error(fname, retcode)
            safe_ocl_api_call.restype = None
            return safe_ocl_api_call
        elif error_code_idx == -1:
            @_ctypes_func_wraps(libfn)
            def safe_ocl_api_call(*args):
                retcode = cl_int()
                new_args = args.copy()
                new_args.append(ctypes.byref(retcode))
                if retcode != enums.CL_SUCCESS:
                    _raise_opencl_error(fname, retcode)

            safe_ocl_api_call.argtypes = safe_ocl_api_call.argtypes[:-1]
            return safe_ocl_api_call
        else:
            _raise_opencl_driver_error("Invalid prototype for '{0}'.", fname)


    def _find_api(self, fname):
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        def absent_function(*args, **kws):
            raise _raise_opencl_driver_error("Function '{0}' not found.", fname)

        return absent_function


# Platform class ###############################################################
class Platform(object):
    """
    The Platform represents possible different implementations of OpenCL in a
    host.
    """
    def __init__(self, platform_id, driver):
        def get_info(param_name):
            sz = ctypes.c_size_t()
            driver.clGetPlatformInfo(platform_id, param_name, 0, None, ctypes.byref(sz))
            ret_val = (ctypes.c_char * sz.value)()
            driver.clGetPlatformInfo(platform_id, param_name, sz, ctypes.byref(ret_val), None)
            return ret_val.value

        self.profile = get_info(enums.CL_PLATFORM_PROFILE)
        self.version = get_info(enums.CL_PLATFORM_VERSION)
        self.name = get_info(enums.CL_PLATFORM_NAME)
        self.vendor = get_info(enums.CL_PLATFORM_VENDOR)
        self.extensions = get_info(enums.CL_PLATFORM_EXTENSIONS).split()

    def __repr__(self):
        return "<OpenCL Platform name:{0} vendor:{1} profile:{2} version:{3}>".format(self.name, self.vendor, self.profile, self.version)


# Exception classes ############################################################
class OpenCLSupportError(Exception):
    pass

class OpenCLDriverError(Exception):
    pass

class OpenCLAPIError(Exception):
    pass

# Error messages ###############################################################

DRIVER_NOT_FOUND_MSG = """
OpenCL library cannot be found.
Make sure that OpenCL is installed in your system.
Try setting the environment variable NUMBA_OPENCL_LIBRARY
with the path to your OpenCL shared library.
"""
def _raise_driver_not_found():
    raise OpenCLSupportError(DRIVER_NOT_FOUND_MSG)


DRIVER_LOAD_ERRMSG = """
OpenCL library failed to load with the following error:
{0}
"""
def _raise_driver_error(e):
    raise OpenCLSupportError(DRIVER_LOAD_ERRMSG.format(e))

BAD_ENV_PATH_ERRMSG = """
NUMBA_OPENCL_LIBRARY is set to '{0}' which is not a valid path to a
dynamic link library for your system.
"""
def _raise_bad_env_path(path, extra=None):
    error_message=BAD_ENV_PATH_ERRMSG.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)


def _raise_opencl_driver_error(msg, function):
    e = OpenCLDriverError(msg.format(function))
    e.fname = function
    raise e

def _raise_opencl_error(fname, errcode):
    e = OpenCLAPIError("OpenCL Error when calling '{0}': {1}".format(fname, errcode))
    e.fname = fname
    e.code = errcode
    raise e


# The Driver ###################################################################
driver = Driver()
