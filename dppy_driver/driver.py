from __future__ import absolute_import, division, print_function

from . import _numba_dppy_bindings
from cffi import FFI
from numpy import ndarray
from numba import ctypes_support as ctypes

ffi = FFI()

##########################################################################
# Exception classes
##########################################################################


class DPPyDriverError(Exception):
    """A problem encountered inside the libdpglue Python driver code
    """
    pass


class DeviceNotFoundError(Exception):
    """The requested type of device is not available
    """
    pass


class UnsupportedTypeError(Exception):
    """When expecting either a DeviceArray or numpy.ndarray object
    """
    pass


##########################################################################
# Helper functions
##########################################################################


def _raise_driver_error(fname, errcode):
    e = DPPyDriverError("DP_GLUE_FAILURE encountered")
    e.fname = fname
    e.code = errcode
    raise e


def _raise_device_not_found_error(fname):
    e = DeviceNotFoundError("This type of device not available on the system")
    e.fname = fname
    raise e


def _raise_unsupported_type_error(fname):
    e = UnsupportedTypeError("Type needs to be DeviceArray or numpy.ndarray")
    e.fname = fname
    raise e


def _raise_unsupported_kernel_arg_error(fname):
    e = (UnsupportedTypeError("Type needs to be DeviceArray or a supported "
                              "ctypes type "
                              "(refer _is_supported_ctypes_raw_obj)"))
    e.fname = fname
    raise e


def _is_supported_ctypes_raw_obj(obj):
    return isinstance(obj, (ctypes.c_ssize_t,
                            ctypes.c_double,
                            ctypes.c_float,
                            ctypes.c_uint8,
                            ctypes.c_size_t))

##########################################################################
# DeviceArray class
##########################################################################


class DeviceArray:

    _buffObj  = None
    _ndarray  = None
    _buffSize = None
    _dataPtr  = None

    def __init__(self, env_ptr, arr):

        # We only support device buffers for ndarray and ctypes (for basic
        # types like int, etc)
        if not isinstance(arr, ndarray):
            _raise_unsupported_type_error("DeviceArray constructor")

        # create a dp_buffer_t object
        self._buffObj  = _numba_dppy_bindings.ffi.new("buffer_t *")
        self._ndarray  = arr
        self._buffSize = arr.itemsize * arr.size
        self._dataPtr  = ffi.cast("void *", arr.ctypes.data)
        retval = (_numba_dppy_bindings
                  .lib
                  .create_dp_rw_mem_buffer(env_ptr,
                                           self._buffSize,
                                           self._buffObj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("create_dp_rw_mem_buffer", -1)

    def __del__(self):
        retval = (_numba_dppy_bindings
                  .lib
                  .destroy_dp_rw_mem_buffer(self._buffObj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("destroy_dp_rw_mem_buffer", -1)

    def get_buffer_obj(self):
        return self._buffObj

    def get_buffer_size(self):
        return self._buffSize

    def get_buffer_ptr(self):
        return self.get_buffer_obj()[0].buffer_ptr

    def get_data_ptr(self):
        return self._dataPtr

##########################################################################
# Program class
##########################################################################


class Program():

    def __init__(self, device_env, spirv_module):
        self._prog_t_obj = _numba_dppy_bindings.ffi.new("program_t *")
        retval = (_numba_dppy_bindings
                  .lib
                  .create_dp_program_from_spirv(
                      device_env.get_env_ptr(),
                      spirv_module,
                      len(spirv_module),
                      self._prog_t_obj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error(
                "create_dp_program_from_spirv", -1)

        retval = (_numba_dppy_bindings
                  .lib
                  .build_dp_program(
                      device_env.get_env_ptr(),
                      self._prog_t_obj[0]))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error(
                "build_dp_program", -1)

    def __del__(self):
        retval = (_numba_dppy_bindings
                  .lib
                  .destroy_dp_program(self._prog_t_obj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("destroy_dp_program", -1)

    def get_prog_t_obj(self):
        return self._prog_t_obj[0]


##########################################################################
# Kernel class
##########################################################################


class Kernel():

    def __init__(self, device_env, prog_t_obj, kernel_name):
        self._kernel_t_obj = _numba_dppy_bindings.ffi.new("kernel_t *")
        retval = (_numba_dppy_bindings
                  .lib
                  .create_dp_kernel(
                      device_env.get_env_ptr(),
                      prog_t_obj.get_prog_t_obj(),
                      kernel_name.encode(),
                      self._kernel_t_obj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("create_dp_kernel", -1)

    def __del__(self):
        retval = (_numba_dppy_bindings
                  .lib
                  .destroy_dp_kernel(self._kernel_t_obj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("destroy_dp_kernel", -1)

    def get_kernel_t_obj(self):
        return self._kernel_t_obj[0]

    def dump(self):
        retval = self._kernel_t_obj.dump_fn(self._kernel_t_obj)
        if retval == -1:
            _raise_driver_error("kernel dump_fn", -1)

##########################################################################
# KernelArg class
##########################################################################


class KernelArg():

    def __init__(self, arg, void_p_arg=False):
        self.arg = arg
        self.kernel_arg_t = _numba_dppy_bindings.ffi.new("kernel_arg_t *")
        if void_p_arg is True:
            self.ptr_to_arg_p = ffi.new("void **")
            self.ptr_to_arg_p[0] = ffi.cast("void *", 0)
            retval = (_numba_dppy_bindings
                      .lib
                      .create_dp_kernel_arg(
                          self.ptr_to_arg_p,
                          ffi.sizeof(self.ptr_to_arg_p),
                          self.kernel_arg_t))
            if(retval):
                _raise_driver_error("create_dp_kernel_arg", -1)
        else:
            if isinstance(arg, DeviceArray):
                self.ptr_to_arg_p = ffi.new("void **")
                self.ptr_to_arg_p[0] = arg.get_buffer_obj()[0].buffer_ptr
                retval = (_numba_dppy_bindings
                          .lib
                          .create_dp_kernel_arg(
                              self.ptr_to_arg_p,
                              arg.get_buffer_obj()[0].sizeof_buffer_ptr,
                              self.kernel_arg_t))
                if(retval):
                    _raise_driver_error("create_dp_kernel_arg", -1)
            else:
                # it has to be of type ctypes
                if getattr(arg, '__module__', None) == "ctypes":
                    self.ptr_to_arg_p = ffi.cast("void *", ctypes.addressof(arg))
                    retval = (_numba_dppy_bindings
                              .lib
                              .create_dp_kernel_arg(
                                  self.ptr_to_arg_p,
                                  ctypes.sizeof(arg),
                                  self.kernel_arg_t))
                    if(retval):
                        _raise_driver_error("create_dp_kernel_arg", -1)
                else:
                    print(type(arg))
                    _raise_unsupported_kernel_arg_error("KernelArg init")

    def __del__(self):
        retval = (_numba_dppy_bindings
                  .lib
                  .destroy_dp_kernel_arg(self.kernel_arg_t))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("destroy_dp_kernel_arg", -1)

    def get_kernel_arg_obj(self):
        return self.kernel_arg_t[0]


##########################################################################
# DeviceEnv class
##########################################################################


class DeviceEnv():

    def __init__(self, env_t_obj):
        self._env_ptr = env_t_obj

    def __del__(self):
        pass

    def retain_context(self):
        retval = (_numba_dppy_bindings
                  .lib
                  .retain_dp_context(self._env_ptr.context))
        if(retval == -1):
            _raise_driver_error("retain_dp_context", -1)

        return (self._env_ptr.context)

    def release_context(self):
        retval = (_numba_dppy_bindings
                  .lib
                  .release_dp_context(self._env_ptr.context))
        if retval == -1:
            _raise_driver_error("release_dp_context", -1)

    def copy_array_to_device(self, array):
        if isinstance(array, DeviceArray):
            retval = (_numba_dppy_bindings
                      .lib
                      .write_dp_mem_buffer_to_device(
                          self._env_ptr,
                          array.get_buffer_obj()[0],
                          True,
                          0,
                          array.get_buffer_size(),
                          array.get_data_ptr()))
            if retval == -1:
                print("Error Code  : ", retval)
                _raise_driver_error("write_dp_mem_buffer_to_device", -1)
            return array
        elif isinstance(array, ndarray) or getattr(array, '__module__', None) == "ctypes":
            dArr = DeviceArray(self._env_ptr, array)
            retval = (_numba_dppy_bindings
                      .lib
                      .write_dp_mem_buffer_to_device(
                          self._env_ptr,
                          dArr.get_buffer_obj()[0],
                          True,
                          0,
                          dArr.get_buffer_size(),
                          dArr.get_data_ptr()))
            if retval == -1:
                print("Error Code  : ", retval)
                _raise_driver_error("write_dp_mem_buffer_to_device", -1)
            return dArr
        else:
            _raise_unsupported_type_error("copy_array_to_device")

    def copy_array_from_device(self, array):
        if not isinstance(array, DeviceArray):
            _raise_unsupported_type_error("copy_array_to_device")
        retval = (_numba_dppy_bindings
                  .lib
                  .read_dp_mem_buffer_from_device(
                      self._env_ptr,
                      array.get_buffer_obj()[0],
                      True,
                      0,
                      array.get_buffer_size(),
                      array.get_data_ptr()))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("read_dp_mem_buffer_from_device", -1)

    def get_context_ptr(self):
        return self._env_ptr.context

    def get_device_ptr(self):
        return self._env_ptr.device

    def get_queue_ptr(self):
        return self._env_ptr.queue

    def get_env_ptr(self):
        return self._env_ptr

    def get_max_work_item_dims(self):
        return self._env_ptr.max_work_item_dims

    def get_max_work_group_size(self):
        return self._env_ptr.max_work_group_size

    def dump(self):
        retval = self._env_ptr.dump_fn(self._env_ptr)
        if retval == -1:
            _raise_driver_error("env dump_fn", -1)

##########################################################################
# Runtime class
##########################################################################


class _Runtime():
    """Runtime is a singleton class that creates a dp_runtime
    object. The dp_runtime (runtime) object on creation
    instantiates a OpenCL context and a corresponding OpenCL command
    queue for the first available CPU on the system. Similarly, the
    runtime object also stores the context and command queue for the
    first available GPU on the system.
    """
    _runtime = None
    _cpu_device = None
    _gpu_device = None

    def __new__(cls):
        obj = cls._runtime
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            ffiobj = _numba_dppy_bindings.ffi.new("runtime_t *")
            retval = (_numba_dppy_bindings
                      .lib
                      .create_dp_runtime(ffiobj))
            if(retval):
                print("Error Code  : ", retval)
                _raise_driver_error("create_dp_runtime", -1)

            cls._runtime = ffiobj

            if cls._runtime[0][0].has_cpu:
                cls._cpu_device = DeviceEnv(cls._runtime[0][0].first_cpu_env)
            else:
                # What should we do here? Raise an exception? Provide warning?
                # Maybe do not do anything here, only when this context is to
                # be used then first check if the context is populated.
                print("No CPU device")

            if cls._runtime[0][0].has_gpu:
                cls._gpu_device = DeviceEnv(cls._runtime[0][0].first_gpu_env)
            else:
                # Same as the cpu case above.
                print("No GPU device")

        return obj

    def __init__(self):
        pass

    def __del__(self):
        print("Delete dp_runtime object.")
        retval = (_numba_dppy_bindings
                  .lib
                  .destroy_dp_runtime(_Runtime._runtime))
        if(retval):
            _raise_driver_error("destroy_dp_runtime", -1)

    def has_cpu_device(self):
        return self._cpu_device is not None

    def has_gpu_device(self):
        return self._gpu_device is not None

    def get_cpu_device(self):
        if(self._cpu_device is None):
            _raise_device_not_found_error("get_cpu_device")

        return self._cpu_device

    def get_gpu_device(self):
        if(self._gpu_device is None):
            _raise_device_not_found_error("get_gpu_device")

        return self._gpu_device

    def dump(self):
        retval = self._runtime.dump_fn(self._runtime)
        if retval == -1:
            _raise_driver_error("runtime dump_fn", -1)

runtime = _Runtime()

##########################################################################
# Public API
##########################################################################


def enqueue_kernel(device_env, kernel, kernelargs, global_work_size,
                   local_work_size):
    l_work_size_array = None
    kernel_arg_array = _numba_dppy_bindings.ffi.new(
                        "kernel_arg_t [" + str(len(kernelargs)) + "]")
    g_work_size_array = _numba_dppy_bindings.ffi.new(
                        "size_t [" + str(len(global_work_size)) + "]")
    if local_work_size:
        l_work_size_array = _numba_dppy_bindings.ffi.new(
                            "size_t [" + str(len(local_work_size)) + "]")
    else:
        l_work_size_array = ffi.NULL
    for i in range(len(kernelargs)):
        kernel_arg_array[i] = kernelargs[i].get_kernel_arg_obj()
    for i in range(len(global_work_size)):
        g_work_size_array[i] = global_work_size[i]
    for i in range(len(local_work_size)):
        l_work_size_array[i] = local_work_size[i]
    retval = (_numba_dppy_bindings
              .lib
              .set_args_and_enqueue_dp_kernel(
                  device_env.get_env_ptr(),
                  kernel.get_kernel_t_obj(),
                  len(kernelargs),
                  kernel_arg_array,
                  len(global_work_size),
                  ffi.NULL,
                  g_work_size_array,
                  l_work_size_array
                  ))
    if(retval):
        _raise_driver_error("set_args_and_enqueue_dp_kernel", -1)


def is_available():
    """Return a boolean to indicate the availability of a DPPY device.
    """
    return runtime.has_cpu_device()  or runtime.has_gpu_device()

def dppy_error():
    _raise_driver_error()
