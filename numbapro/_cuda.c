#include <Python.h>
#include <cuda.h>
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "_cuda.h"

/* prototypes */
static const char *curesult_to_str(CUresult e);

#define CHECK_CUDA_ERROR(error) if (cu_result != CUDA_SUCCESS) {       \
        PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result)); \
        return -1;                                                     \
    }

#define CUDA_ERROR_BUFFER_SIZE 128

static PyObject *cuda_exc_type;

void
init_cuda_exc_type(PyObject *exc_type)
{
    Py_INCREF(exc_type);
    cuda_exc_type = exc_type;
}

int
init_attributes(CudaDeviceAttrs *attrs)
{
    CUresult cu_result;
    CUdevice cu_device;
    CudaDeviceAttrs out;

    cu_result = cuCtxGetDevice(&cu_device);
    CHECK_CUDA_ERROR(cu_result);

#define _GETATTR(value, attr) \
        cu_result = cuDeviceGetAttribute(&out.value, attr, cu_device); \
        CHECK_CUDA_ERROR(cu_result);

    _GETATTR(MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    _GETATTR(MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    _GETATTR(MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    _GETATTR(MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    _GETATTR(MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    _GETATTR(MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    _GETATTR(MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    _GETATTR(MAX_SHARED_MEMORY, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

#undef _GETATTR

    *attrs = out;
    return 0;
}

int
cuda_load(PyObject *ptx_str, CUmodule *cu_module)
{
    char *ptx;
    CUresult cu_result;
    char cu_errors[CUDA_ERROR_BUFFER_SIZE];

    /* Capture error log */
    CUjit_option options[2];
    void *values[2];

    if (!PyBytes_Check(ptx_str)) {
        PyErr_SetString(PyExc_TypeError, "Expected byte string PTX assembly");
        return -1;
    }

    ptx = PyBytes_AS_STRING(ptx_str);

    options[0] = CU_JIT_ERROR_LOG_BUFFER;
    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    values[0] = (void *) cu_errors;
    values[1] = (void *) CUDA_ERROR_BUFFER_SIZE;

    cu_result = cuModuleLoadDataEx(cu_module, ptx, 3, options, values);

    if (cu_result != CUDA_SUCCESS) {
        PyErr_Format(PyExc_ValueError, "%s: %s",
                     curesult_to_str(cu_result), cu_errors);
        return -1;
    }

    return 0;
}

int
cuda_getfunc(CUmodule cu_module, CUfunction *cu_func, char *funcname)
{
    CUresult cu_result;
    cu_result = cuModuleGetFunction(cu_func, cu_module, funcname);
    CHECK_CUDA_ERROR(cu_result);
    return 0;
}

int
invoke_cuda_ufunc(PyUFuncObject *ufunc, CudaDeviceAttrs *device_attrs,
                  CUfunction cu_func, PyListObject *inputs,
                  PyObject *out, int copy_in, int copy_out, void **out_mem,
                  unsigned int griddimx, unsigned int griddimy,
                  unsigned int griddimz, unsigned int blockdimx,
                  unsigned int blockdimy, unsigned int blockdimz)
{
    CUresult cu_result;
    void **args;
    int i;
    int total_count;

    if (ufunc->nin != PyList_GET_SIZE(inputs)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected %d input arguments, got %ld.",
                     ufunc->nin, (long) PyList_GET_SIZE(inputs));
        return -1;
    }

    /*
    Set up kernel arguments:

        1) First pass all input arrays
        2) then the output array
        3) finally the total number of elements

    TODO: pass strides to allow strided arrays
    */
    args = malloc((ufunc->nin + 1) * sizeof(void *));
    if (!args) {
        PyErr_NoMemory();
        return -1;
    }

    for (i = 0; i < ufunc->nin; i++) {
        args[i] = PyArray_DATA(PyList_GET_ITEM(inputs, i));
    }
    args[ufunc->nin] = PyArray_DATA(out);
    args[ufunc->nin + 1] = &total_count;
    total_count = (int) PyArray_SIZE(out);

    /* Launch kernel & check result */
    cu_result = cuLaunchKernel(cu_func, griddimx, griddimy, griddimz,
                               blockdimx, blockdimy, blockdimz,
                               0 /* sharedMemBytes */, 0 /* hStream */,
                               args, 0);
    free(args);
    CHECK_CUDA_ERROR(cu_result);

    return 0;
}

int
invoke_cuda_gufunc(PyUFuncObject *ufunc, PyListObject *args)
{
    if (!ufunc->core_enabled) {
        PyErr_SetString(PyExc_TypeError, "Not a generalized ufunc");
        return -1;
    }

    /* implement, see PyUFunc_GeneralizedFunction */
    return 0;
}




/* Shamelessly copied from pycuda/src/cpp/cuda.hpp */
static const char *curesult_to_str(CUresult e)
{
    switch (e)
    {
      case CUDA_SUCCESS: return "success";
      case CUDA_ERROR_INVALID_VALUE: return "invalid value";
      case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
      case CUDA_ERROR_NOT_INITIALIZED: return "not initialized";

    #if CUDAPP_CUDA_VERSION >= 2000
      case CUDA_ERROR_DEINITIALIZED: return "deinitialized";
    #endif

    #if CUDAPP_CUDA_VERSION >= 4000
      case CUDA_ERROR_PROFILER_DISABLED: return "profiler disabled";
      case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "profiler not initialized";
      case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "profiler already started";
      case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "profiler already stopped";
    #endif

      case CUDA_ERROR_NO_DEVICE: return "no device";
      case CUDA_ERROR_INVALID_DEVICE: return "invalid device";

      case CUDA_ERROR_INVALID_IMAGE: return "invalid image";
      case CUDA_ERROR_INVALID_CONTEXT: return "invalid context";
      case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "context already current";
      case CUDA_ERROR_MAP_FAILED: return "map failed";
      case CUDA_ERROR_UNMAP_FAILED: return "unmap failed";
      case CUDA_ERROR_ARRAY_IS_MAPPED: return "array is mapped";
      case CUDA_ERROR_ALREADY_MAPPED: return "already mapped";
      case CUDA_ERROR_NO_BINARY_FOR_GPU: return "no binary for gpu";
      case CUDA_ERROR_ALREADY_ACQUIRED: return "already acquired";
      case CUDA_ERROR_NOT_MAPPED: return "not mapped";
    #if CUDAPP_CUDA_VERSION >= 3000
      case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "not mapped as array";
      case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "not mapped as pointer";
    #ifdef CUDAPP_POST_30_BETA
      case CUDA_ERROR_ECC_UNCORRECTABLE: return "ECC uncorrectable";
    #endif
    #endif
    #if CUDAPP_CUDA_VERSION >= 3010
      case CUDA_ERROR_UNSUPPORTED_LIMIT: return "unsupported limit";
    #endif
    #if CUDAPP_CUDA_VERSION >= 4000
      case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "context already in use";
    #endif

      case CUDA_ERROR_INVALID_SOURCE: return "invalid source";
      case CUDA_ERROR_FILE_NOT_FOUND: return "file not found";
    #if CUDAPP_CUDA_VERSION >= 3010
      case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        return "shared object symbol not found";
      case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        return "shared object init failed";
    #endif

      case CUDA_ERROR_INVALID_HANDLE: return "invalid handle";

      case CUDA_ERROR_NOT_FOUND: return "not found";

      case CUDA_ERROR_NOT_READY: return "not ready";

      case CUDA_ERROR_LAUNCH_FAILED: return "launch failed";
      case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "launch out of resources";
      case CUDA_ERROR_LAUNCH_TIMEOUT: return "launch timeout";
      case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "launch incompatible texturing";

    #if CUDAPP_CUDA_VERSION >= 4000
      case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "peer access already enabled";
      case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "peer access not enabled";
      case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "primary context active";
      case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "context is destroyed";
    #endif

    #if (CUDAPP_CUDA_VERSION >= 3000) && (CUDAPP_CUDA_VERSION < 3020)
      case CUDA_ERROR_POINTER_IS_64BIT:
         return "attempted to retrieve 64-bit pointer via 32-bit api function";
      case CUDA_ERROR_SIZE_IS_64BIT:
         return "attempted to retrieve 64-bit size via 32-bit api function";
    #endif

    #if CUDAPP_CUDA_VERSION >= 4010
      case CUDA_ERROR_ASSERT:
         return "device-side assert triggered";
      case CUDA_ERROR_TOO_MANY_PEERS:
         return "too many peers";
      case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
         return "host memory already registered";
      case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
         return "host memory not registered";
    #endif

      case CUDA_ERROR_UNKNOWN:
        return "unknown";

      default:
        return "invalid/unknown error code";
    }
}