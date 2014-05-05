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
import ctypes.util
import os
import sys

try:
    long
except NameError:
    long = int


_ctypes_array_metaclass = type(cl_int*1)

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
        pass

    def _populate_platforms(self):
        count = cl_uint()
        self.clGetPlatformIDs(0, None, ctypes.byref(count))
        platforms = (ctypes.c_void_p * count.value) ()
        self.clGetPlatformIDs(count, platforms, None)
        self._platforms = [Platform(x) for x in platforms]


    @property
    def platforms(self):
        return self._platforms[:]

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
                new_args = args + (ctypes.byref(retcode),)
                rv = libfn(*new_args)
                if retcode.value != enums.CL_SUCCESS:
                    _raise_opencl_error(fname, retcode)
                return rv

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


    def create_context(self, platform=None, devices=None):
        """
        Create an opencl context. By default, it will create the context in the
        first platform, using all the devices in that platform.
        """
        if platform is None:
            platform = self.platforms[0]

        if devices is None:
            devices = platform.devices # all of the platform devices is a good default?

        _properties = (cl_context_properties*3)(enums.CL_CONTEXT_PLATFORM, platform.id, 0)
        _devices = (cl_device_id*len(devices))(*[dev.id for dev in devices])
        ctxt = driver.clCreateContext(_properties, len(devices), _devices, None, None)
        rv = Context(ctxt)
        driver.clReleaseContext(ctxt)
        return Context(ctxt)


class OpenCLWrapper(object):
    """
    A base class for OpenCL wrapper objects.
    Identity will be based on their OpenCL id.
    subclasses must implement _retain and _release methods appropriate for their id
    """
    def __init__(self, id):
        self.id = id
        self._retain()

    def __del__(self):
        self._release()

    def __eq__(self, other):
        return (self.__class__ == other.__class__) and (self.id == other.id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)


# Platform class ###############################################################
class Platform(object):
    """
    The Platform represents possible different implementations of OpenCL in a
    host.
    """
    def __init__(self, platform_id):
        def get_info(param_name):
            sz = ctypes.c_size_t()
            try:
                driver.clGetPlatformInfo(platform_id, param_name, 0, None, ctypes.byref(sz))
            except OpenCLAPIError:
                return None
            ret_val = (ctypes.c_char * sz.value)()
            driver.clGetPlatformInfo(platform_id, param_name, sz, ctypes.byref(ret_val), None)
            return ret_val.value

        self.id = platform_id
        self.profile = get_info(enums.CL_PLATFORM_PROFILE)
        self.version = get_info(enums.CL_PLATFORM_VERSION)
        self.name = get_info(enums.CL_PLATFORM_NAME)
        self.vendor = get_info(enums.CL_PLATFORM_VENDOR)
        self.extensions = get_info(enums.CL_PLATFORM_EXTENSIONS).split()

        device_count = cl_uint()
        driver.clGetDeviceIDs(platform_id, enums.CL_DEVICE_TYPE_ALL, 0, None, ctypes.byref(device_count))
        devices = (cl_device_id * device_count.value)()
        driver.clGetDeviceIDs(platform_id, enums.CL_DEVICE_TYPE_ALL, device_count, devices, None)
        self._devices = [Device(x, self, driver) for x in devices]

    @property
    def devices(self):
        return self._devices[:]

    def __repr__(self):
        return "<OpenCL Platform name:{0} vendor:{1} profile:{2} version:{3}>".format(self.name, self.vendor, self.profile, self.version)


# Device class #################################################################
class Device(object):
    """
    The device represents a computing device.
    """
    def __init__(self, device_id, platform_id, driver):
        def get_string_info(param_name):
            sz = ctypes.c_size_t()
            try:
                driver.clGetDeviceInfo(device_id, param_name, 0, None, ctypes.byref(sz))
            except OpenCLAPIError:
                return None
            ret_val = (ctypes.c_char * sz.value)()
            driver.clGetDeviceInfo(device_id, param_name, sz, ret_val, None)
            return ret_val.value

        def get_info(param_name, param_type, count = 1):
            ret_val = (param_type * count)()
            try:
                driver.clGetDeviceInfo(device_id, param_name, ctypes.sizeof(param_type), ret_val, None)
            except OpenCLAPIError:
                return None
            if count == 1:
                return ret_val[0]
            else:
                return [x.value for x in ret_val]
    
        self.platform_id = platform_id
        self.id = device_id
        self.name = get_string_info(enums.CL_DEVICE_NAME)
        self.profile = get_string_info(enums.CL_DEVICE_PROFILE)
        self.type = get_info(enums.CL_DEVICE_TYPE, cl_device_type)
        self.vendor = get_string_info(enums.CL_DEVICE_VENDOR)
        self.vendor_id = get_info(enums.CL_DEVICE_VENDOR_ID, cl_uint)
        self.version = get_string_info(enums.CL_DEVICE_VERSION)
        self.driver_version = get_string_info(enums.CL_DRIVER_VERSION)
        self.address_bits = get_info(enums.CL_DEVICE_ADDRESS_BITS, cl_uint)
        self.available = get_info(enums.CL_DEVICE_AVAILABLE, cl_bool)
        self.compiler_available = get_info(enums.CL_DEVICE_COMPILER_AVAILABLE, cl_bool)
        self.double_fp_config = get_info(enums.CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config)
        self.endian_little = get_info(enums.CL_DEVICE_ENDIAN_LITTLE, cl_bool)
        self.error_correction_support = get_info(enums.CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool)
        self.execution_capabilities = get_info(enums.CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities)
        self.extensions = get_string_info(enums.CL_DEVICE_EXTENSIONS).split()
        self.global_mem_cache_size = get_info(enums.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong)
        self.global_mem_cache_type = get_info(enums.CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type)
        self.global_mem_cacheline_size = get_info(enums.CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint)
        self.global_mem_size = get_info(enums.CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong)
        self.half_fp_config = get_info(enums.CL_DEVICE_HALF_FP_CONFIG, cl_device_fp_config)
        self.image_support = get_info(enums.CL_DEVICE_IMAGE_SUPPORT, cl_bool)
        self.image2d_max_height = get_info(enums.CL_DEVICE_IMAGE2D_MAX_HEIGHT, ctypes.c_size_t)
        self.image2d_max_width = get_info(enums.CL_DEVICE_IMAGE2D_MAX_WIDTH, ctypes.c_size_t)
        self.image3d_max_depth = get_info(enums.CL_DEVICE_IMAGE3D_MAX_DEPTH, ctypes.c_size_t)
        self.image3d_max_height = get_info(enums.CL_DEVICE_IMAGE3D_MAX_HEIGHT, ctypes.c_size_t)
        self.image3d_max_width = get_info(enums.CL_DEVICE_IMAGE3D_MAX_WIDTH, ctypes.c_size_t)
        self.local_mem_size = get_info(enums.CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong)
        self.local_mem_type = get_info(enums.CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type)
        self.max_clock_frequency = get_info(enums.CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint)
        self.max_compute_units = get_info(enums.CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint)
        self.max_constant_args = get_info(enums.CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint)
        self.max_constant_buffer_size = get_info(enums.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong)
        self.max_mem_alloc_size = get_info(enums.CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong)
        self.max_parameter_size = get_info(enums.CL_DEVICE_MAX_PARAMETER_SIZE, ctypes.c_size_t)
        self.max_read_image_args = get_info(enums.CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint)
        self.max_samplers = get_info(enums.CL_DEVICE_MAX_SAMPLERS, cl_uint)
        self.max_work_group_size = get_info(enums.CL_DEVICE_MAX_WORK_GROUP_SIZE, ctypes.c_size_t)
        self.max_work_item_dimensions = get_info(enums.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint)
        self.max_work_item_sizes = get_info(enums.CL_DEVICE_MAX_WORK_ITEM_SIZES, ctypes.c_size_t, count=self.max_work_item_dimensions)
        self.max_write_image_args = get_info(enums.CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint)
        self.mem_base_addr_align = get_info(enums.CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint)
        self.min_data_type_align_size = get_info(enums.CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, cl_uint)
        self.preferred_vector_width_char = get_info(enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint)
        self.preferred_vector_width_short = get_info(enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint)
        self.preferred_vector_width_int = get_info(enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint)
        self.preferred_vector_width_long = get_info(enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint)
        self.preferred_vector_width_float = get_info(enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint)
        self.preferred_vector_width_double = get_info(enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint)
        self.profiling_timer_resolution = get_info(enums.CL_DEVICE_PROFILING_TIMER_RESOLUTION, ctypes.c_size_t)
        self.queue_properties = get_info(enums.CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties)
        self.single_fp_config = get_info(enums.CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config)

    @property
    def type_str(self):
        types = []
        if self.type & enums.CL_DEVICE_TYPE_CPU:
            types.append('CPU')
        if self.type & enums.CL_DEVICE_TYPE_GPU:
            types.append('GPU')
        if self.type & enums.CL_DEVICE_TYPE_ACCELERATOR:
            types.append('ACCELERATOR')
        if self.type & enums.CL_DEVICE_TYPE_CUSTOM:
            types.append('CUSTOM')
        return ' + '.join(types)

    def __repr__(self):
        return "<OpenCL device id:{3} name:{0} type:{1} profile:{2}>".format(self.name, self.type_str, self.profile, self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Device) and (self.id == other.id)

    def __ne__(self, other):
        return not (self == other)

    

# Context class ################################################################
class Context(OpenCLWrapper):
    """
    An OpenCL context resource.

    The Context is the major workhorse in resource management. Memobjects, queues,
    events, programs are aggregated in a Context.
    It acts as a factory of all those resources.
    """
    def _retain(self):
        driver.clRetainContext(self.id)

    def _release(self):
        driver.clReleaseContext(self.id)

    def create_buffer(self, size_in_bytes, flags=enums.CL_MEM_READ_WRITE, host_ptr=None):
        mem = driver.clCreateBuffer(self.id, flags, size_in_bytes, host_ptr)
        return Memory(mem)

    def create_program_from_source(self, source):
        source = ctypes.create_string_buffer(source)
        ptr = ctypes.c_char_p(ctypes.addressof(source))
        program = driver.clCreateProgramWithSource(self.id, 1, ctypes.byref(ptr), None)
        return Program(program)

    def create_command_queue(self, device, out_of_order=False, profiling=False):
        flags = 0
        if out_of_order:
            flags |= enums.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        if profiling:
            flags |= enums.CL_QUEUE_PROFILING_ENABLE

        return CommandQueue(driver.clCreateCommandQueue(self.id, device.id, flags))

    
# Memory class #################################################################
class Memory(OpenCLWrapper):
    """
    An OpenCL memory object (cl_mem)

    This is a light-weight proxy object. It will retain the id on creation and
    release on its destruction. This means having one live Memory object will
    prevent the cl_mem object from being destroyed. It is possible to have
    many Memory objects for the same cl_mem object. This should not be an
    issue.
    Two different Memory objects pointing to the same cl_mem will be evaluated
    equal and will share a common hash.
    """
    def _retain(self):
        driver.clRetainMemObject(self.id)

    def _release(self):
        driver.clReleaseMemObject(self.id)


# CommandQueue class ###########################################################
class CommandQueue(OpenCLWrapper):
    """
    An OpenCL command queue. Any OpenCL operation on OpenCL objects such as
    memory, program and kernel objects are performed using a command-queue.
    A queue is an ordered set of operations (commands). The order in the queue
    acts as a means of synchronization. Operations in different queues are
    independent.
    """
    def _retain(self):
        driver.clRetainCommandQueue(self.id)

    def _release(self):
        driver.clReleaseCommandQueue(self.id)

    def flush(self):
        driver.clFlush(self.id)

    def finish(self):
        driver.clFinish(self.id)

    def enqueue_task(self, kernel, wait_list=None, wants_event=False):
        """
        Enqueue a task for execution.
        - wait_list: a list of events to wait for
        - wants_event: if True, return an event for synchronization with this command.
        """
        if wait_list is not None:
            # must be a list of Events
            num_events_in_wait_list = len(wait_list)
            event_wait_list = (cl_event * num_events_in_wait_list)(*[e.id for e in wait_list])
        else:
            num_events_in_wait_list = 0
            event_wait_list = None

        event = (cl_event * 1)() if wants_event else None

        driver.clEnqueueTask(self.id, kernel.id, num_events_in_wait_list, event_wait_list, event)
        if wants_event:
            return event[0]

    def enqueue_nd_range_kernel(self, kernel, nd, global_work_size, local_work_size,
                                wait_list=None, wants_event = False):
        """
        enqueue a n-dimensional range kernel
        - nd : number of dimensions
        - global_work_size : a list/tuple of nd elements with the global work shape
        - local_work_size : a list/tuple of nd elements with the local work shape
        - wait_list: a list of events to wait for
        - wants_event: if True, return an event for synchronization with this command
        """
        if wait_list is not None:
            num_events_in_wait_list = len(wait_list)
            event_wait_list = (cl_event * num_events_in_wait_list)(*[e.id for e in wait_list])
        else:
            num_events_in_wait_list = 0
            event_wait_list = None

        global_ws = (ctypes.c_size_t*nd)(*global_work_size)
        local_ws = (ctypes.c_size_t*nd)(*local_work_size)
        event = (cl_event*1)() if wants_event else None
        driver.clEnqueueNDRangeKernel(self.id, kernel.id, nd, None, global_ws, local_ws,
                                      num_events_in_wait_list, event_wait_list, event)

        if wants_event:
            return event[0]

    def enqueue_read_buffer(self, buff, offset, bc, dst_ptr,
                            blocking=True, wait_list=None, wants_event=False):
        if wait_list is not None:
            # must be a list of Events
            num_events_in_wait_list = len(wait_list)
            event_wait_list = (cl_event * num_events_in_wait_list)(*[e.id for e in wait_list])
        else:
            num_events_in_wait_list = 0
            event_wait_list = None

        event = (cl_event * 1)() if wants_event else None

        driver.clEnqueueReadBuffer(self.id, buff.id, blocking, offset, bc, dst_ptr,
                                   num_events_in_wait_list, event_wait_list, event)

        if wants_event:
            return Event(event[0])


    def enqueue_write_buffer(self, buff, offset, bc, src_ptr,
                             blocking=True, wait_list=None, event=False, wants_event=False):
        if wait_list is not None:
            num_events_in_wait_list = len(wait_list)
            event_wait_list = (cl_event * num_events_in_wait_list)(*[e.id for e in wait_list])
        else:
            num_events_in_wait_list = 0
            event_wait_list = None

        event = (cl_event * 1)() if wants_event else None
        driver.clEnqueueWriteBuffer(self.id, buff.id, blocking, offset, bc, src_ptr,
                                    num_events_in_wait_list, event_wait_list, event)

        if wants_event:
            return Event(event[0])


# Program class ################################################################
class Program(OpenCLWrapper):
    """
    An OpenCL program consists of a set of kernels identified by the __kernel
    qualifier in the program source.
    """
    def _retain(self):
        driver.clRetainProgram(self.id)

    def _release(self):
        driver.clReleaseProgram(self.id)

    def build(self, devices=None, options=None):
        if options is not None:
            options = ctypes.create_string_buffer(options)

        if devices is not None:
            num_devices = len(devices)
            devices = (cl_device_id * num_devices)(*[dev.id for dev in devices])
        else:
            num_devices = 0
        driver.clBuildProgram(self.id, num_devices, devices, options, None, None)

    def create_kernel(self, name):
        name = ctypes.create_string_buffer(name)
        return Kernel(driver.clCreateKernel(self.id, name))


# Kernel class #################################################################
class Kernel(OpenCLWrapper):
    def _retain(self):
        driver.clRetainKernel(self.id)

    def _release(self):
        driver.clReleaseKernel(self.id)

    def set_arg_raw(self, arg_number, ptr, size_in_bytes):
        driver.clSetKernelArg(self.id, arg_number, size_in_bytes, ptr)

    def set_arg(self, arg_number, value):
        if isinstance(value, (Memory,)):
            arg_value = ctypes.byref(cl_mem(value.id))
            arg_size = ctypes.sizeof(cl_mem)
        elif isinstance(type(value), _ctypes_array_metaclass):
            arg_value = value
            arg_size = ctypes.sizeof(arg_value)
        elif isinstance(value, int):
            arg_value = (cl_int *1)(value)
            arg_size = ctypes.sizeof(arg_value)
        else:
            arg_value = None
            arg_size = 0

        self.set_arg_raw(arg_number, arg_value, arg_size)


    def get_work_group_size_for_device(self, device):
        sz = (ctypes.c_size_t * 1)()
        driver.clGetKernelWorkGroupInfo(self.id, device.id, enums.CL_KERNEL_WORK_GROUP_SIZE, ctypes.sizeof(sz), sz, None)
        return sz[0]


# Event class ##################################################################
class Event(OpenCLWrapper):
    def _retain(self):
        driver.clRetainEvent(self.id)

    def _release(self):
        driver.clReleaseEvent(self.id)


# Exception classes ############################################################
class OpenCLSupportError(Exception):
    """The underlying OpenCL library does not support a feature. This includes
    OpenCL availability itself.
    """
    pass

class OpenCLDriverError(Exception):
    """A problem with the OpenCL Python driver code"""
    pass

class OpenCLAPIError(Exception):
    """Incorrect usage of OpenCL API"""
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

def _raise_opencl_driver_error(msg, *args):
    e = OpenCLDriverError(msg.format(*args))
    e.fname = function
    raise e

def _raise_opencl_error(fname, errcode):
    e = OpenCLAPIError("OpenCL Error when calling '{0}': {1}".format(fname, errcode))
    e.fname = fname
    e.code = errcode
    raise e

NUMBA_USAGE_ERRMSG = """
Incorrect use of numba Python OpenCL driver:
{0}

This may error may be caused by access to reserved parts of the driver.
"""

def _raise_usage_error(msg):
    """
    An error caused by a bad usage pattern in the code or by the user accessing
    internals he should not be messing around
    """
    e = OpenCLAPIError(NUMBA_USAGE_ERRMSG.format(msg))


def _raise_unimplemented_error():
    _raise_opencl_driver_error("unimplemented")

# The Driver ###################################################################
driver = Driver()
driver._populate_platforms()
