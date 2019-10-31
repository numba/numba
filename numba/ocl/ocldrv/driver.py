"""
OpenCL driver bridge implementation

ctypes based driver bridge to OpenCL. Based on the CUDA driver.
Unlike CUDA, OpenCL has platforms, which take responsibilities that
for CUDA are embeeded on the driver, but in reality belong to the platform
"""

from __future__ import absolute_import, print_function, division

from numba import utils, mviewbuf
from . import drvapi, enums
from .types import *

from functools import partial
import contextlib
import weakref
import functools
import ctypes as ct
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
    envpath = os.environ.get('NUMBA_OPENCL_DRIVER', None)
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
        dll_loader = ct.WinDLL
    else:
        dll_loader = ct.CDLL

    dll_path = envpath or ct.util.find_library("OpenCL")

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


def _event_list_to_ctypes(event_list):
    assert(event_list)
    count = len(event_list)
    events = (cl_event * count)(*[e.id for e in event_list])
    return count, events

# The Driver ###################################################################

class Driver(object):
    """
    API functions are lazily bound.
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

    @property
    def is_available(self):
        return self.lib is not None

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
            retval = libfn
        elif error_code_idx == 0:
            @_ctypes_func_wraps(libfn)
            def safe_ocl_api_call(*args):
                retcode = libfn(*args)
                if retcode != enums.CL_SUCCESS:
                    _raise_opencl_error(fname, retcode)
            safe_ocl_api_call.restype = None
            retval =  safe_ocl_api_call
        elif error_code_idx == -1:
            @_ctypes_func_wraps(libfn)
            def safe_ocl_api_call(*args):
                retcode = cl_int()
                new_args = args + (ct.byref(retcode),)
                rv = libfn(*new_args)
                if retcode.value != enums.CL_SUCCESS:
                    _raise_opencl_error(fname, retcode.value)
                return rv

            safe_ocl_api_call.argtypes = safe_ocl_api_call.argtypes[:-1]
            retval =  safe_ocl_api_call
        else:
            _raise_opencl_driver_error("Invalid prototype for '{0}'.", fname)
        setattr(self, fname, retval)
        return retval


    def _find_api(self, fname):
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        def absent_function(*args, **kws):
            raise _raise_opencl_driver_error("Function '{0}' not found.", fname)

        return absent_function


    def wait_for_events(self, *events):
        if events:
            self.clWaitForEvents(*_event_list_to_ctypes(events))

    @property
    def platforms(self):
        count = cl_uint()
        self.clGetPlatformIDs(0, None, ct.byref(count))
        platforms = (cl_platform_id * count.value)()
        self.clGetPlatformIDs(count, platforms, None)
        return [Platform(x, False) for x in platforms]

    @property
    def default_platform(self):
        return self.platforms[0]

    @property
    def default_device(self):
        return self.default_platform.default_device

# The Driver ###################################################################
driver = Driver()


# OpenCLWrapper ################################################################
def _get_string_info(func, attr_enum, self):
    sz = ct.c_size_t()
    func(self.id, attr_enum, 0, None, ct.byref(sz))
    ret_val = (ct.c_char*sz.value)()
    func(self.id, attr_enum, sz, ret_val, None)
    return ret_val.value


def _get_info(param_type, func, attr_enum, self):
    ret_val = (param_type * 1)()
    func(self.id, attr_enum, ct.sizeof(ret_val), ret_val, None)
    return ret_val[0]

def _get_array_info(param_type, func, attr_enum, self):
    sz = ct.c_size_t()
    func(self.id, attr_enum, 0, None, ct.byref(sz))
    count = sz.value // ct.sizeof(param_type)
    ret_val = (param_type*count)()
    func(self.id, attr_enum, sz, ret_val, None)
    return list(ret_val)


class OpenCLWrapper(object):
    """
    A base class for OpenCL wrapper objects.
    Identity will be based on their OpenCL id.
    subclasses must implement _retain and _release methods appropriate for their id
    """
    def __init__(self, id, retain=True):
        self.id = id
        if retain:
            self._retain()

    def __del__(self):
        self._release()

    def __eq__(self, other):
        return (self.__class__ == other.__class__) and (self.id == other.id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)


    # add getters on an "as needed" basis
    # note that the map is based on the underlying ctype and not the
    # cl_whatever that may be used in _cl_properties.
    # c_char_p assumes that it is a string.
    _getter_by_type = {
        ct.c_char_p: _get_string_info,
        ct.c_void_p: partial(_get_info, ct.c_void_p),
        ct.c_uint64: partial(_get_info, ct.c_uint64),
        ct.c_uint32: partial(_get_info, ct.c_uint32),
        ct.c_int32: partial(_get_info, ct.c_int32),
        ct.POINTER(ct.c_size_t): partial(_get_array_info, ct.c_size_t),
        ct.POINTER(ct.c_void_p): partial(_get_array_info, ct.c_void_p),
        ct.POINTER(ct.c_int64): partial(_get_array_info, ct.c_int64),
    }

    @classmethod
    def _define_cl_properties(cls):
        """
        define cl properties for this class.
        """
        func = getattr(driver, cls._cl_info_function)
        for p in cls._cl_properties.items():
            getter = cls._getter_by_type[p[1][-1]]
            getter = partial(getter, func, p[1][0])
            setattr(cls, p[0], property(getter))

# Platform class ###############################################################
class Platform(OpenCLWrapper):
    """
    The Platform represents possible different implementations of OpenCL in a
    host.
    """
    _cl_info_function = "clGetPlatformInfo"
    _cl_properties = {
        'profile': (enums.CL_PLATFORM_PROFILE, ct.c_char_p),
        'version': (enums.CL_PLATFORM_VERSION, ct.c_char_p),
        'name': (enums.CL_PLATFORM_NAME, ct.c_char_p),
        'vendor': (enums.CL_PLATFORM_VENDOR, ct.c_char_p),
        'extensions': (enums.CL_PLATFORM_EXTENSIONS, ct.c_char_p)
    }

    def __repr__(self):
        return "<OpenCL Platform name:{0} vendor:{1} profile:{2} version:{3}>".format(self.name, self.vendor, self.profile, self.version)

    def _retain(self):
        # It looks like there is no retain/release for platforms
        pass

    def _release(self):
        # It looks like there is no retain/release for platforms
        pass

    def get_device_count(self, type_=enums.CL_DEVICE_TYPE_ALL):
        count = cl_uint()
        driver.clGetDeviceIDs(self.id, type_, 0, None, ct.byref(count))
        return count.value

    def get_devices(self, type_=enums.CL_DEVICE_TYPE_ALL):
        device_count = cl_uint()
        try:
            driver.clGetDeviceIDs(self.id, type_, 0, None, ct.byref(device_count))
        except OpenCLAPIError as e:
            if e.code == enums.CL_DEVICE_NOT_FOUND:
                return []
            else:
                raise
        devices = (cl_device_id * device_count.value)()
        driver.clGetDeviceIDs(self.id, type_, device_count, devices, None)
        return [Device(x) for x in devices]

    @property
    def cpu_devices(self):
        return self.get_devices(enums.CL_DEVICE_TYPE_CPU)

    @property
    def gpu_devices(self):
        return self.get_devices(enums.CL_DEVICE_TYPE_GPU)

    @property
    def accelerator_devices(self):
        return self.get_devices(enums.CL_DEVICE_TYPE_ACCELERATOR)

    @property
    def all_devices(self):
        return self.get_devices()

    @property
    def default_device(self):
        return self.get_devices(enums.CL_DEVICE_TYPE_DEFAULT)[0]

    def create_context(self, devices=None):
        """
        Create an opencl context using the given or all devices
        """
        if devices is None:
            devices = self.all_devices

        _properties = (cl_context_properties*3)(enums.CL_CONTEXT_PLATFORM, self.id, 0)
        _devices = (cl_device_id*len(devices))(*[dev.id for dev in devices])
        return Context(driver.clCreateContext(_properties, len(devices), _devices, None, None),
                       False)

Platform._define_cl_properties()


# Device class #################################################################
class Device(OpenCLWrapper):
    """
    The device represents a computing device.
    """
    _cl_info_function = "clGetDeviceInfo"
    _cl_properties = {
        "_platform_id":                  (enums.CL_DEVICE_PLATFORM, cl_platform_id),
        "name":                          (enums.CL_DEVICE_NAME, ct.c_char_p),
        "profile":                       (enums.CL_DEVICE_PROFILE, ct.c_char_p),
        "type":                          (enums.CL_DEVICE_TYPE, cl_device_type),
        "vendor":                        (enums.CL_DEVICE_VENDOR, ct.c_char_p),
        "vendor_id":                     (enums.CL_DEVICE_VENDOR_ID, cl_uint),
        "version":                       (enums.CL_DEVICE_VERSION, ct.c_char_p),
        "driver_version":                (enums.CL_DRIVER_VERSION, ct.c_char_p),
        "address_bits":                  (enums.CL_DEVICE_ADDRESS_BITS, cl_uint),
        "available":                     (enums.CL_DEVICE_AVAILABLE, cl_bool),
        "compiler_available":            (enums.CL_DEVICE_COMPILER_AVAILABLE, cl_bool),
        "double_fp_config":              (enums.CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config),
        "endian_little":                 (enums.CL_DEVICE_ENDIAN_LITTLE, cl_bool),
        "error_correction_support":      (enums.CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool),
        "execution_capabilities":        (enums.CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities),
        "extensions":                    (enums.CL_DEVICE_EXTENSIONS, ct.c_char_p),
        "global_mem_cache_size":         (enums.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong),
        "global_mem_cache_type":         (enums.CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type),
        "global_mem_cacheline_size":     (enums.CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint),
        "global_mem_size":               (enums.CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong),
        "half_fp_config":                (enums.CL_DEVICE_HALF_FP_CONFIG, cl_device_fp_config),
        "image_support":                 (enums.CL_DEVICE_IMAGE_SUPPORT, cl_bool),
        "image2d_max_height":            (enums.CL_DEVICE_IMAGE2D_MAX_HEIGHT, ct.c_size_t),
        "image2d_max_width":             (enums.CL_DEVICE_IMAGE2D_MAX_WIDTH, ct.c_size_t),
        "image3d_max_depth":             (enums.CL_DEVICE_IMAGE3D_MAX_DEPTH, ct.c_size_t),
        "image3d_max_height":            (enums.CL_DEVICE_IMAGE3D_MAX_HEIGHT, ct.c_size_t),
        "image3d_max_width":             (enums.CL_DEVICE_IMAGE3D_MAX_WIDTH, ct.c_size_t),
        "local_mem_size":                (enums.CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong),
        "local_mem_type":                (enums.CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type),
        "max_clock_frequency":           (enums.CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint),
        "max_compute_units":             (enums.CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint),
        "max_constant_args":             (enums.CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint),
        "max_constant_buffer_size":      (enums.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong),
        "max_mem_alloc_size":            (enums.CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong),
        "max_parameter_size":            (enums.CL_DEVICE_MAX_PARAMETER_SIZE, ct.c_size_t),
        "max_read_image_args":           (enums.CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint),
        "max_samplers":                  (enums.CL_DEVICE_MAX_SAMPLERS, cl_uint),
        "max_work_group_size":           (enums.CL_DEVICE_MAX_WORK_GROUP_SIZE, ct.c_size_t),
        "max_work_item_dimensions":      (enums.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint),
        "max_work_item_sizes":           (enums.CL_DEVICE_MAX_WORK_ITEM_SIZES, ct.POINTER(ct.c_size_t)),
        "max_write_image_args":          (enums.CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint),
        "mem_base_addr_align":           (enums.CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint),
        "min_data_type_align_size":      (enums.CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, cl_uint),
        "preferred_vector_width_char":   (enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint),
        "preferred_vector_width_short":  (enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint),
        "preferred_vector_width_int":    (enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint),
        "preferred_vector_width_long":   (enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint),
        "preferred_vector_width_float":  (enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint),
        "preferred_vector_width_double": (enums.CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint),
        "profiling_timer_resolution":    (enums.CL_DEVICE_PROFILING_TIMER_RESOLUTION, ct.c_size_t),
        "queue_properties":              (enums.CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties),
        "reference_count":               (enums.CL_DEVICE_REFERENCE_COUNT, cl_uint),
        "single_fp_config":              (enums.CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config),
    }

    @property
    def opencl_version(self):
        ver = self.version # CL_DEVICE_VERSION looks like "OpenCL X.Y ..."
        opencl, x, y = ver[0:6], int(ver[7]), int(ver[9])
        assert(opencl=='OpenCL' and x in {1,2} and y in {0,1,2})
        return (x,y)

    @property
    def platform(self):
        return Platform(self._platform_id)

    @property
    def type_str(self):
        t = self.type
        types = []
        if t & enums.CL_DEVICE_TYPE_CPU:
            types.append('CPU')
        if t & enums.CL_DEVICE_TYPE_GPU:
            types.append('GPU')
        if t & enums.CL_DEVICE_TYPE_ACCELERATOR:
            types.append('ACCELERATOR')
        if t & enums.CL_DEVICE_TYPE_CUSTOM:
            types.append('CUSTOM')
        return ' + '.join(types)

    def _retain(self):
        #driver.clRetainDevice(self.id)
        pass # only Subdevices retain

    def _release(self):
        #driver.clReleaseDevice(self.id)
        pass  # only Subdevices release

    def __repr__(self):
        return "<OpenCL device id:{3} name:{0} type:{1} profile:{2}>".format(self.name, self.type_str, self.profile, hex(self.id))

    def reset(self):
        return True # TODO

Device._define_cl_properties()

# Context class ################################################################
class Context(OpenCLWrapper):
    """
    An OpenCL context resource.

    The Context is the major workhorse in resource management. Memobjects, queues,
    events, programs are aggregated in a Context.
    It acts as a factory of all those resources.
    """
    _cl_info_function = "clGetContextInfo"
    _cl_properties = {
        "reference_count": (enums.CL_CONTEXT_REFERENCE_COUNT, cl_uint),
        "_device_ids": (enums.CL_CONTEXT_DEVICES, ct.POINTER(cl_device_id)),
        "_properties": (enums.CL_CONTEXT_PROPERTIES, ct.POINTER(cl_context_properties)),
    }

    @property
    def devices(self):
        return [Device(_id) for _id in self._device_ids]

    @property
    def platform(self):
        """may return None in certain cases, which means 'left to implementation'"""
        p = self._properties
        for i in range(0,len(p),2):
            if p[i] == enums.CL_CONTEXT_PLATFORM:
                return Platform(p[i+1])
        return None

    def _retain(self):
        driver.clRetainContext(self.id)

    def _release(self):
        driver.clReleaseContext(self.id)

    def create_buffer(self, size_in_bytes, host_ptr=None, flags=enums.CL_MEM_READ_WRITE):
        mem = MemObject(driver.clCreateBuffer(self.id, flags, size_in_bytes, host_ptr), False)
        mem.host_ptr = None # @@
        return mem

    def create_buffer_and_copy(self, size_in_bytes, host_ptr, flags=enums.CL_MEM_READ_WRITE):
        flags = flags | enums.CL_MEM_COPY_HOST_PTR
        mem = MemObject(driver.clCreateBuffer(self.id, flags, size_in_bytes, host_ptr), False)
        mem.host_ptr = None # @@
        return mem

    def create_pointer(self, size, align, flags=enums.CL_MEM_READ_WRITE):
        ptr = driver.clSVMAlloc(self.id, flags, size, align)
        flags = flags | enums.CL_MEM_USE_HOST_PTR
        mem = MemObject(driver.clCreateBuffer(self.id, flags, size, ptr), False, ptr, size)
        mem.host_ptr = ptr # @@ fixes AMD OpenCL bug, with CL_MEM_HOST_PTR returning NULL
        return mem

    def create_program_from_source(self, source):
        source = ct.create_string_buffer(source)
        ptr = ct.c_char_p(ct.addressof(source))
        program = driver.clCreateProgramWithSource(self.id, 1, ct.byref(ptr), None)
        return Program(program, False)

    def create_program_from_binary(self, binary):
        devs = self._device_ids
        numdevs = len(devs)
        bin_lens = [len(binary)] * numdevs
        binary = ct.create_string_buffer(binary, len(binary))
        binary_ptrs = [ct.addressof(binary)] * numdevs
        binaries = (ct.c_void_p * numdevs)(*binary_ptrs)
        device_list = (cl_device_id * numdevs)(*devs)
        lengths = (ct.c_size_t * numdevs)(*bin_lens)
        program = driver.clCreateProgramWithBinary(self.id, numdevs, device_list,lengths, binaries, None)
        return Program(program)

    def create_program_from_il(self, binary):
        interlang = ct.c_char_p(binary)
        program = driver.clCreateProgramWithIL(self.id, interlang, len(binary))
        return Program(program)

    def create_command_queue(self, device, out_of_order=False, profiling=False):
        flags = 0
        if out_of_order:
            flags |= enums.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        if profiling:
            flags |= enums.CL_QUEUE_PROFILING_ENABLE

        return CommandQueue(driver.clCreateCommandQueue(self.id, device.id, flags), False)


Context._define_cl_properties()

# Memory class #################################################################
class MemObject(OpenCLWrapper):
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
    __ocl_memory__ = True

    _cl_info_function = "clGetMemObjectInfo"
    _cl_properties = {
        "type": (enums.CL_MEM_TYPE, cl_mem_object_type),
        "flags": (enums.CL_MEM_FLAGS, cl_mem_flags),
        "_size": (enums.CL_MEM_SIZE, ct.c_size_t),
        #"host_ptr": (enums.CL_MEM_HOST_PTR, ct.c_void_p), # @@ AMD bug
        "map_count": (enums.CL_MEM_MAP_COUNT, cl_uint),
        "reference_count": (enums.CL_MEM_REFERENCE_COUNT, cl_uint),
        "_context_id": (enums.CL_MEM_CONTEXT, cl_context),
        "svm": (enums.CL_MEM_USES_SVM_POINTER, cl_bool)
    }

    def __init__(self, id, retain=True, ptr=None, size=None):
        super(MemObject,self).__init__(id,retain)
        assert((ptr and size) or (not ptr and not size))
        if ptr:
            self.ptr = ptr
            self._view_size = size

    @property
    def size(self):
        return self._view_size if self.svm else self._size

    @property
    def context(self):
        return Context(self._context_id)

    @property
    def device_ctypes_pointer(self):
        """ does not return the device pointer but the full cl MemObject,
        which owns the pointer that is given to clSetKernelArgSVMPointer
        """
        return self # @

    def create_region(self, offset, size):
        """
        create a sub-buffer from this memobject at offset 'offset' with size 'size'.
        note that the source memobject can not be a sub-buffer object itself.
        Note that in order to pass pointers in OpenCL it has to be done through a memobject,
        as there is no "device pointers". This allows creating memobjects that reside inside
        of other memobjects, which is important to handle subarrays, for example.
        Look into OpenCL clCreateSubBuffer to see important notes about the use of overlapping
        regions.
        """
        params = (ct.c_size_t * 2)(offset, size)
        region = enums.CL_BUFFER_CREATE_TYPE_REGION
        return MemObject(driver.clCreateSubBuffer(self.id, 0, region, params), False)

    def _retain(self):
        driver.clRetainMemObject(self.id)

    def _release(self):
        svm = self.svm
        refc = self.reference_count
        ptr = self.host_ptr
        ctx = self._context_id
        driver.clReleaseMemObject(self.id)
        if svm and refc == 0:
            driver.clSVMFree(ptr,ctx)

    def view(self, start, stop=None):
        pointer = self.host_ptr + start
        if stop is None:
            size = self.size - start
        else:
            size = stop - start
        assert size > 0, "zero or negative memory size"
        #return MemObject(driver.clCreateBuffer(self._context_id, self.flags, size, pointer))
        mem = MemObject(self.id,True,pointer,size)
        mem.host_ptr = self.host_ptr  # @@ fixes AMD OpenCL bug, with CL_MEM_HOST_PTR returning NULL
        return mem

MemObject._define_cl_properties()

# CommandQueue class ###########################################################
class CommandQueue(OpenCLWrapper):
    """
    An OpenCL command queue. Any OpenCL operation on OpenCL objects such as
    memory, program and kernel objects are performed using a command-queue.
    A queue is an ordered set of operations (commands). The order in the queue
    acts as a means of synchronization. Operations in different queues are
    independent.
    """
    _cl_info_function = "clGetCommandQueueInfo"
    _cl_properties = {
        "_context_id": (enums.CL_QUEUE_CONTEXT, cl_context),
        "_device_id": (enums.CL_QUEUE_DEVICE, cl_device_id),
        "reference_count": (enums.CL_QUEUE_REFERENCE_COUNT, cl_uint),
        "properties": (enums.CL_QUEUE_PROPERTIES, cl_command_queue_properties),
    }

    @property
    def context(self):
        return Context(self._context_id)

    @property
    def device(self):
        return Device(self._device_id)

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
            return Event(event[0], False)

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

        global_ws = (ct.c_size_t*nd)(*global_work_size)
        local_ws = ((ct.c_size_t*nd)(*local_work_size)
                    if local_work_size is not None else local_work_size)
        event = (cl_event*1)() if wants_event else None
        driver.clEnqueueNDRangeKernel(self.id, kernel.id, nd, None, global_ws, local_ws,
                                      num_events_in_wait_list, event_wait_list, event)

        if wants_event:
            return Event(event[0], False)

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
            return Event(event[0], False)


    def enqueue_write_buffer(self, buff, offset, bc, src_ptr,
                             blocking=True, wait_list=None, wants_event=False):
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
            return Event(event[0], False)

    def enqueue_copy_buffer(self, src_buff, dst_buff, src_offset, dst_offset, bc,
                            wait_list=None, wants_event=False):
        if wait_list is not None:
            num_events_in_wait_list = len(wait_list)
            event_wait_list = (cl_event * num_events_in_wait_list)(*[e.id for e in wait_list])
        else:
            num_events_in_wait_list = 0
            event_wait_list = None

        event = (cl_event * 1)() if wants_event else None
        driver.clEnqueueCopyBuffer(self.id, src_buff.id, dst_buff.id, src_offset, dst_offset, bc,
                               num_events_in_wait_list, event_wait_list, event)

        if wants_event:
            return Event(event[0], False)

    @contextlib.contextmanager
    def auto_synchronize(self):
        '''
        A context manager that waits for all commands in this stream to execute
        and commits any pending memory transfers upon exiting the context.
        '''
        yield self
        self.finish()

CommandQueue._define_cl_properties()


# Program class ################################################################
class Program(OpenCLWrapper):
    """
    An OpenCL program consists of a set of kernels identified by the __kernel
    qualifier in the program source.
    """
    _cl_info_function = "clGetProgramInfo"
    _cl_properties = {
        "reference_count": (enums.CL_PROGRAM_REFERENCE_COUNT, cl_uint),
        "_context_id": (enums.CL_PROGRAM_CONTEXT, cl_context),
        "_device_ids": (enums.CL_PROGRAM_DEVICES, ct.POINTER(cl_device_id)),
        "source": (enums.CL_PROGRAM_SOURCE, ct.c_char_p),
        "_kernel_names": (enums.CL_PROGRAM_KERNEL_NAMES, ct.c_char_p),
        "_binary_sizes": (enums.CL_PROGRAM_BINARY_SIZES, ct.POINTER(ct.c_size_t))
        #note: access for binaries needs a special pattern not handled by
        #      autoregister of cl properties
    }

    @property
    def context(self):
        return Context(self._context_id)

    @property
    def devices(self):
        return [Device(d) for d in self._device_ids]

    @property
    def kernel_names(self):
        return self._kernel_names.decode().split(';')

    @property
    def binaries(self):
        sizes = self._binary_sizes
        results = [(ct.c_byte*sz)() for sz in sizes]
        arg = (ct.POINTER(ct.c_byte)*len(results))(*results)
        driver.clGetProgramInfo(self.id, enums.CL_PROGRAM_BINARIES, ct.sizeof(arg), arg, None)
        return results

    def _retain(self):
        driver.clRetainProgram(self.id)

    def _release(self):
        driver.clReleaseProgram(self.id)

    def build(self, devices=None, options=None):
        if options is not None:
            options = ct.create_string_buffer(options)

        if devices is not None:
            num_devices = len(devices)
            devices = (cl_device_id * num_devices)(*[dev.id for dev in devices])
        else:
            num_devices = 0
        try:
            driver.clBuildProgram(self.id, num_devices, devices, options, None, None)
        except OpenCLAPIError as e:
            self.build_info(devices)

    def build_info(self, devices=None):
        if devices is None:
            devices = self.devices
        for dev in devices:
            status = cl_build_status()
            driver.clGetProgramBuildInfo(self.id, dev.id, enums.CL_PROGRAM_BUILD_STATUS, ct.sizeof(cl_build_status), ct.byref(status), None)
            log_size = ct.c_size_t()
            driver.clGetProgramBuildInfo(self.id, dev.id, enums.CL_PROGRAM_BUILD_LOG, 0, None, ct.byref(log_size))
            log = ct.create_string_buffer(log_size.value)
            driver.clGetProgramBuildInfo(self.id, dev.id, enums.CL_PROGRAM_BUILD_LOG, log_size, log, None)
            err_str = opencl_strerror(status.value) + ": " + log.value
            _raise_opencl_driver_error(err_str)

    def create_kernel(self, name, args=None):
        name = ct.create_string_buffer(name)
        kern = Kernel(driver.clCreateKernel(self.id, name), False)
        if args is not None:
            kern.set_args(args)
        return kern

Program._define_cl_properties()

# Kernel class #################################################################
class Kernel(OpenCLWrapper):
    """
    An OpenCL kernel is an entry point to a program, potentially with its args
    bound. Argument binding is done through this kernel object.
    """
    _cl_info_function = "clGetKernelInfo"
    _cl_properties = {
        "function_name": (enums.CL_KERNEL_FUNCTION_NAME, ct.c_char_p),
        "num_args": (enums.CL_KERNEL_NUM_ARGS, cl_uint),
        "_context_id": (enums.CL_KERNEL_CONTEXT, cl_context),
        "_program_id": (enums.CL_KERNEL_PROGRAM, cl_program),
        "reference_count": (enums.CL_KERNEL_REFERENCE_COUNT, cl_uint),
    }

    @property
    def context(self):
        return Context(self._context_id)

    @property
    def program(self):
        return Program(self._program_id)

    def _retain(self):
        driver.clRetainKernel(self.id)

    def _release(self):
        driver.clReleaseKernel(self.id)

    def set_arg_raw(self, arg_number, ptr, size_in_bytes):
        driver.clSetKernelArg(self.id, arg_number, size_in_bytes, ptr)

    def set_arg(self, arg_number, value):
        if isinstance(value, (MemObject,)) and value.svm:
            driver.clSetKernelArgSVMPointer(self.id, arg_number, value.ptr)
            return
        elif isinstance(value, (MemObject,)):
            arg_value = ct.byref(cl_mem(value.id))
            arg_size = ct.sizeof(cl_mem)
        elif isinstance(type(value), _ctypes_array_metaclass):
            arg_value = value
            arg_size = ct.sizeof(arg_value)
        elif isinstance(value, int):
            arg_value = (cl_int *1)(value)
            arg_size = ct.sizeof(arg_value)
        else:
            # Assume ctypes
            arg_value = ct.addressof(value)
            arg_size = ct.sizeof(value)

        self.set_arg_raw(arg_number, arg_value, arg_size)

    def set_args(self, args):
        for idx, val in enumerate(args):
            self.set_arg(idx, val)

    def get_work_group_size_for_device(self, device):
        sz = (ct.c_size_t * 1)()
        driver.clGetKernelWorkGroupInfo(self.id, device.id, enums.CL_KERNEL_WORK_GROUP_SIZE, ct.sizeof(sz), sz, None)
        return sz[0]

Kernel._define_cl_properties()

# Event class ##################################################################
class Event(OpenCLWrapper):
    _cl_info_function = "clGetEventInfo"
    _cl_properties = {
        '_context_id': (enums.CL_EVENT_CONTEXT, cl_event),
        '_command_queue_id': (enums.CL_EVENT_COMMAND_QUEUE, cl_command_queue),
        'command_type': (enums.CL_EVENT_COMMAND_TYPE, cl_command_type),
        'execution_status': (enums.CL_EVENT_COMMAND_EXECUTION_STATUS, cl_int),
        'reference_count': (enums.CL_EVENT_REFERENCE_COUNT, cl_uint),
    }

    @property
    def context(self):
        return Context(self._context_id)

    @property
    def command_queue(self):
        return CommandQueue(self._command_queue_id)

    def _retain(self):
        driver.clRetainEvent(self.id)

    def _release(self):
        driver.clReleaseEvent(self.id)

    def wait(self):
        driver.wait_for_events(self)

Event._define_cl_properties()

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
    #e.fname = function
    raise e

def _raise_opencl_error(fname, errcode):
    e = OpenCLAPIError("OpenCL Error '{0}': {1}".format(fname, opencl_strerror(errcode)))
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

# error name formatting
_cl_error_dict = {}
_cl_errors = """
CL_SUCCESS
CL_DEVICE_NOT_FOUND
CL_DEVICE_NOT_AVAILABLE
CL_COMPILER_NOT_AVAILABLE
CL_MEM_OBJECT_ALLOCATION_FAILURE
CL_OUT_OF_RESOURCES
CL_OUT_OF_HOST_MEMORY
CL_PROFILING_INFO_NOT_AVAILABLE
CL_MEM_COPY_OVERLAP
CL_IMAGE_FORMAT_MISMATCH
CL_IMAGE_FORMAT_NOT_SUPPORTED
CL_BUILD_PROGRAM_FAILURE
CL_MAP_FAILURE
CL_MISALIGNED_SUB_BUFFER_OFFSET
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
CL_COMPILE_PROGRAM_FAILURE
CL_LINKER_NOT_AVAILABLE
CL_LINK_PROGRAM_FAILURE
CL_DEVICE_PARTITION_FAILED
CL_KERNEL_ARG_INFO_NOT_AVAILABLE
CL_INVALID_VALUE
CL_INVALID_DEVICE_TYPE
CL_INVALID_PLATFORM
CL_INVALID_DEVICE
CL_INVALID_CONTEXT
CL_INVALID_QUEUE_PROPERTIES
CL_INVALID_COMMAND_QUEUE
CL_INVALID_HOST_PTR
CL_INVALID_MEM_OBJECT
CL_INVALID_IMAGE_FORMAT_DESCRIPTOR
CL_INVALID_IMAGE_SIZE
CL_INVALID_SAMPLER
CL_INVALID_BINARY
CL_INVALID_BUILD_OPTIONS
CL_INVALID_PROGRAM
CL_INVALID_PROGRAM_EXECUTABLE
CL_INVALID_KERNEL_NAME
CL_INVALID_KERNEL_DEFINITION
CL_INVALID_KERNEL
CL_INVALID_ARG_INDEX
CL_INVALID_ARG_VALUE
CL_INVALID_ARG_SIZE
CL_INVALID_KERNEL_ARGS
CL_INVALID_WORK_DIMENSION
CL_INVALID_WORK_GROUP_SIZE
CL_INVALID_WORK_ITEM_SIZE
CL_INVALID_GLOBAL_OFFSET
CL_INVALID_EVENT_WAIT_LIST
CL_INVALID_EVENT
CL_INVALID_OPERATION
CL_INVALID_GL_OBJECT
CL_INVALID_BUFFER_SIZE
CL_INVALID_MIP_LEVEL
CL_INVALID_GLOBAL_WORK_SIZE
CL_INVALID_PROPERTY
CL_INVALID_IMAGE_DESCRIPTOR
CL_INVALID_COMPILER_OPTIONS
CL_INVALID_LINKER_OPTIONS
CL_INVALID_DEVICE_PARTITION_COUNT
""".split()

for i in _cl_errors:
    _cl_error_dict[getattr(enums,i)] = i

def opencl_strerror(code):
    try:
        return _cl_error_dict[code]
    except KeyError:
        return "Unknown OpenCL error (code={0})".format(code)


# -----------------------------------------------------------------------------

#from . import devices

def _device_pointer_attr(devmem, attr, osize, odata):
    """Query attribute on the device pointer
    """
    driver.clGetMemObjectInfo(devmem.id, attr, osize, ct.byref(odata), None)


def device_pointer_type(devmem):
    """Query the device pointer type: host, device, array, unified?
    """
    ptrtype = c_int(0)
    _device_pointer_attr(devmem, enums.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                         ptrtype)
    map = {
        enums.CU_MEMORYTYPE_HOST: 'host',
        enums.CU_MEMORYTYPE_DEVICE: 'device',
        enums.CU_MEMORYTYPE_ARRAY: 'array',
        enums.CU_MEMORYTYPE_UNIFIED: 'unified',
    }
    return map[ptrtype.value]


def device_extents(devmem):
    """Find the extents (half open begin and end pointer) of the underlying
    device memory allocation.

    NOTE: it always returns the extents of the allocation but the extents
    of the device memory view that can be a subsection of the entire allocation.
    """
    return devmem.ptr, devmem.ptr + devmem.size


def device_memory_size(devmem):
    """Check the memory size of the device memory.
    The result is cached in the device memory object.
    It may query the driver for the memory size of the device memory allocation.
    """
    sz = getattr(devmem, '_ocl_memsize_', None)
    if sz is None:
        s, e = device_extents(devmem)
        sz = e - s
        devmem._ocl_memsize_ = sz
    assert sz > 0, "zero length array"
    return sz


def host_pointer(obj):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    if isinstance(obj, (int, long)):
        return obj

    forcewritable = isinstance(obj, np.void)
    return mviewbuf.memoryview_get_buffer(obj, forcewritable)


def host_memory_extents(obj):
    "Returns (start, end) the start and end pointer of the array (half open)."
    return mviewbuf.memoryview_get_extents(obj)


def memory_size_from_info(shape, strides, itemsize):
    """et the byte size of a contiguous memory buffer given the shape, strides
    and itemsize.
    """
    assert len(shape) == len(strides), "# dim mismatch"
    ndim = len(shape)
    s, e = mviewbuf.memoryview_get_extents_info(shape, strides, ndim, itemsize)
    return e - s


def host_memory_size(obj):
    "Get the size of the memory"
    s, e = host_memory_extents(obj)
    assert e >= s, "memory extend of negative size"
    return e - s


def device_pointer(obj):
    "Get the device pointer as an integer"
    # cl_mem is already an integer, because of ctypes' c_void_p to int autocast
    return device_ctypes_pointer(obj)


def device_ctypes_pointer(obj):
    "Get the ctypes object for the device pointer"
    if obj is None:
        return c_void_p(0)
    require_device_memory(obj)
    return obj.device_ctypes_pointer


def is_device_memory(obj):
    return getattr(obj, '__ocl_memory__', False)


def require_device_memory(obj):
    """A sentry for methods that accept CUDA memory object.
    """
    if not is_device_memory(obj):
        raise Exception("Not an OpenCL memory object.")


def device_memory_depends(devmem, *objs):
    """Add dependencies to the device memory.

    Mainly used for creating structures that points to other device memory,
    so that the referees are not GC and released.
    """
    depset = getattr(devmem, "_depends_", [])
    depset.extend(objs)


def host_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    queue = stream #devices.get_queue(stream)
    dev_mem = dst.gpu_data
    offset = dev_mem.ptr - dev_mem.host_ptr if dev_mem.svm else 0
    queue.enqueue_write_buffer(dst.gpu_data, offset, size, src.ctypes.data)


def device_to_host(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    queue = stream #devices.get_queue(stream)
    dev_mem = src.gpu_data
    offset = dev_mem.ptr - dev_mem.host_ptr if dev_mem.svm else 0
    queue.enqueue_read_buffer(src.gpu_data, offset, size, dst.ctypes.data)


def device_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    varargs = []

    if stream:
        assert isinstance(stream, Stream)
        fn = driver.cuMemcpyDtoDAsync
        varargs.append(stream.handle)
    else:
        fn = driver.cuMemcpyDtoD

    fn(device_pointer(dst), device_pointer(src), size, *varargs)


def device_memset(dst, val, size, stream=0):
    """Memset on the device.
    If stream is not zero, asynchronous mode is used.

    dst: device memory
    val: byte value to be written
    size: number of byte to be written
    stream: a CUDA stream
    """
    varargs = []

    if stream:
        assert isinstance(stream, Stream)
        fn = driver.cuMemsetD8Async
        varargs.append(stream.handle)
    else:
        fn = driver.cuMemsetD8

    fn(device_pointer(dst), val, size, *varargs)

