#include <Python.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "_cuda.h"
#include "_internal.h"

/* prototypes */
static const char *curesult_to_str(CUresult e);

#define CHECK_CUDA_RESULT(cu_result)                                \
    if (cu_result != CUDA_SUCCESS) {                                \
        PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result)); \
        return -1;                                                  \
    }

#define CHECK_CUDA_ERROR(msg, error)                  \
    if (error != cudaSuccess) {                       \
        PyErr_Format(cuda_exc_type, "%s failed: %s",  \
                     msg, cudaGetErrorString(error)); \
        return -1;                                    \
    }

#define BUFSIZE 128

static PyObject *cuda_exc_type;

static void
import_numpy_array(void)
{
    import_array();
}

int
init_cuda_exc_type(void)
{
    import_numpy_array();
    if (PyErr_Occurred())
        return -1;

    if (!cuda_exc_type) {
        cuda_exc_type = PyErr_NewException("_cudadispatch.CudaError",
                                           NULL, NULL);
    }
    /* Return 0 on error */
    return !!cuda_exc_type;
}

int
dealloc(CUmodule cu_module, CUcontext cu_context)
{
    CUresult cu_result;

    if (cu_module) {
        cu_result = cuModuleUnload(cu_module);
        CHECK_CUDA_RESULT(cu_result)
    }
    if (cu_context) {
        cu_result = cuCtxDestroy(cu_context);
        CHECK_CUDA_RESULT(cu_result)
    }
    return 0;
}

int
get_device(CUdevice *cu_device, CUcontext *cu_context, int device_number)
{
    CUresult cu_result;
    cudaError_t cu_error;

    if (device_number < 0) {
        int i, device_count;

        cu_error = cudaGetDeviceCount(&device_count);
        CHECK_CUDA_ERROR("get CUDA device count", cu_error)

        for (i = 0; i < device_count; i++) {
            cu_error = cudaSetDevice(i);
            if (cu_error == cudaSuccess) {
                device_number = i;
                break;
            }
        }
        if (device_number < 0) {
            PyErr_SetString(cuda_exc_type, "No usable devices found");
            return -1;
        }
    }
    /* cu_result = cuCtxGetDevice(&cu_device); */
    cu_result = cuDeviceGet(cu_device, device_number);
    CHECK_CUDA_RESULT(cu_result)

    if (cu_context) {
        cuCtxCreate(cu_context, 0, *cu_device);
        CHECK_CUDA_RESULT(cu_result)
    }
    return 0;
}

int
init_attributes(CUdevice cu_device, CudaDeviceAttrs *attrs)
{
    CUresult cu_result;
    CudaDeviceAttrs out;

#define _GETATTR(value, attr) \
        cu_result = cuDeviceGetAttribute(&out.value, attr, cu_device); \
        CHECK_CUDA_RESULT(cu_result);

    _GETATTR(MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    _GETATTR(MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    _GETATTR(MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    _GETATTR(MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    _GETATTR(MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    _GETATTR(MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    _GETATTR(MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    _GETATTR(MAX_SHARED_MEMORY, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    cu_result = cuDeviceComputeCapability(&out.COMPUTE_CAPABILITY_MAJOR,
                                          &out.COMPUTE_CAPABILITY_MINOR,
                                          cu_device);
    CHECK_CUDA_RESULT(cu_result)

#undef _GETATTR

    *attrs = out;
    return 0;
}

int
cuda_load(PyObject *ptx_str, CUmodule *cu_module)
{
    char *ptx;
    CUresult cu_result;
    int bufsize = BUFSIZE;
    char cu_errors[BUFSIZE];

    cu_errors[0] = '\0';

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
    values[0] = &cu_errors[0];
    values[1] = &bufsize;

    cu_result = cuModuleLoadDataEx(cu_module, ptx, 2, options, values);

/*    cu_result = cuModuleLoadData(cu_module, ptx); */
    if (cu_result != CUDA_SUCCESS) {
        PyErr_Format(PyExc_ValueError, "%s (%d): %s",
                     curesult_to_str(cu_result), (int) cu_result, cu_errors);
        return -1;
    }

    return 0;
}

int
cuda_getfunc(CUmodule cu_module, CUfunction *cu_func, char *funcname)
{
    CUresult cu_result;
    cu_result = cuModuleGetFunction(cu_func, cu_module, funcname);
    CHECK_CUDA_RESULT(cu_result);
    return 0;
}

int
invoke_cuda_ufunc(PyUFuncObject *ufunc, CudaDeviceAttrs *device_attrs,
                  CUfunction cu_func, PyListObject *inputs,
                  PyObject *out, int copy_in, int copy_out,
                  unsigned int griddimx, unsigned int griddimy,
                  unsigned int griddimz, unsigned int blockdimx,
                  unsigned int blockdimy, unsigned int blockdimz)
{
    CUresult cu_result;
    cudaError_t error_code;
    void **args;
    CUdeviceptr *device_pointers;

    int i;
    int total_count;
    int retval = 0;

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
    args = calloc(ufunc->nin + 2,  sizeof(void *));
    device_pointers = calloc(ufunc->nin + 1, sizeof(CUdeviceptr));
    if (!args || !device_pointers) {
        PyErr_NoMemory();
        return -1;
    }

    if (PyList_Append((PyObject *) inputs, out) < 0)
        goto error;

#define CHECK_CUDA_MEM_ERR(action)                                          \
        if (error_code != cudaSuccess) {                                    \
            PyErr_Format(cuda_exc_type, "Got '%s' for memory %s",           \
                         cudaGetErrorString(error_code), action);           \
            goto error;                                                     \
        }

    for (i = 0; i < ufunc->nin + 1; i++) {
        PyObject *array = PyList_GET_ITEM(inputs, i);
        void *data = PyArray_DATA(array);
        npy_intp size = PyArray_NBYTES(array);

        /* Allocate memory on device for array */
        error_code = cudaMalloc((void **) &device_pointers[i], size);
        CHECK_CUDA_MEM_ERR("allocation")
        args[i] = &device_pointers[i];

        if (i != ufunc->nin || copy_in) {
            /* Copy array to device, skip 'out' unless 'copy_in' is true */
            error_code = cudaMemcpy((void *) device_pointers[i], data, size,
                                    cudaMemcpyHostToDevice);
            CHECK_CUDA_MEM_ERR("copy to device")
        }
    }
    total_count = (int) PyArray_SIZE(out);
    args[ufunc->nin + 1] = &total_count;

    /* Launch kernel & check result */
    cu_result = cuLaunchKernel(cu_func, griddimx, griddimy, griddimz,
                               blockdimx, blockdimy, blockdimz,
                               0 /* sharedMemBytes */, 0 /* hStream */,
                               args, 0);

    if (cu_result != CUDA_SUCCESS) {
        PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result));
        goto error;
    }

    /* Wait for kernel to finish */
    error_code = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("device synchronization", error_code)

    goto cleanup;

error:
    retval = -1;
cleanup:
    /* Copy memory back from device to host */
    for (i = 0; i < ufunc->nin + 1; i++) {
        PyObject *array = PyList_GET_ITEM(inputs, i);
        void *data = PyArray_DATA(array);
        npy_intp size = PyArray_NBYTES(array);

        if (args[i] == NULL)
            break;

        error_code = cudaMemcpy(data, (void *) device_pointers[i], size,
                                cudaMemcpyDeviceToHost);
        CHECK_CUDA_MEM_ERR("copy to host")

        error_code = cudaFree((void *) device_pointers[i]);
        CHECK_CUDA_MEM_ERR("free")
    }
    /* Deallocate packed arguments */
    free(device_pointers);
    free(args);

    return retval;
}


#define CHECK_CUDA_RESULT(cu_result)                                \
    if (cu_result != CUDA_SUCCESS) {                                \
        PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result)); \
        return -1;                                                  \
    }

#define CHECK_CUDA_ERROR(msg, error)                  \
    if (error != cudaSuccess) {                       \
        PyErr_Format(cuda_exc_type, "%s failed: %s",  \
                     msg, cudaGetErrorString(error)); \
        return -1;                                    \
    }

static void
print_array(ndarray *array, char *name)
{
    int i;
    printf("array %s, ndim=%d:\n", name, array->nd);
    printf("    shape:");
    for (i = 0; i < array->nd; i++) {
        printf(" %d ", array->dimensions[i]);
    }
    printf("\n    strides:");
    for (i = 0; i < array->nd; i++) {
        printf(" %d ", array->strides[i]);
    }
    puts("");
}

static int
alloc_and_copy(void *data, size_t size, void **result, cudaStream_t stream)
{
	cudaError_t error_code;
	void *p;

    /* printf("Allocating chunk of %d bytes\n", (int) size); */

	error_code = cudaMalloc(&p, size);
	CHECK_CUDA_MEM_ERR("allocation")

    if (stream)
        error_code = cudaMemcpyAsync((void *) p, data, size,
                                     cudaMemcpyHostToDevice, stream);
    else
	    error_code = cudaMemcpy((void *) p, data, size,
	                            cudaMemcpyHostToDevice);

	CHECK_CUDA_MEM_ERR("copy to device")

	*result = p;
	return 0;
error:
	return -1;
}

/*
    Call the generalized ufunc cuda kernel wrapper.

    Arguments are in the form of

        wrapper(PyArrayObject *A, char *data_A, npy_intp *shape_A,
                PyArrayObject *B, char *data_B, npy_intp *shape_B,
                npy_intp *steps)

    The wrapper fills out the data, shape and strides pointers on the GPU.
    The strides pointers are located at &shape[ndim].
*/
static inline int
_cuda_outer_loop(char **args, npy_intp *dimensions, npy_intp *steps, void *data,
                 PyObject **arrays)
{
    npy_intp i, j;
    CudaFunctionAndData *info = (CudaFunctionAndData *) data;
    int result = 0;

    cudaStream_t stream = NULL;
    CUresult cu_result;
    cudaError_t error_code;

    void *device_pointers[MAXARGS * 4 + 1] = { NULL };
    void *kernelargs[MAXARGS * 4 + 1];
    void *data_pointers[MAXARGS * 4 + 1];
    npy_intp sizes[MAXARGS * 4 + 1];

    int nargs = info->nops * 4 + 1;
    ndarray *array_copies;

    if (info->nops > MAXARGS) {
        PyErr_SetString(cuda_exc_type, "Too many array arguments to function");
        return -1;
    }

    error_code = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR("Creating a CUDA stream", error_code);

    array_copies = malloc(sizeof(ndarray) * info->nops * dimensions[0]);
    if (!array_copies) {
        PyErr_NoMemory();
        return -1;
    }

	for (i = 0; i < info->nops; i++) {
	    ndarray *array = (ndarray *) arrays[i];
	    for (j = 0; j < dimensions[0]; j++) {
	        array_copies[i * dimensions[0] + j] = *array;
	    }

		data_pointers[i*4] = &array_copies[i * dimensions[0]];
		data_pointers[i*4+1] = array->data;
		data_pointers[i*4+2] = array->dimensions;
		data_pointers[i*4+3] = array->strides;

		sizes[i*4] = sizeof(ndarray) * dimensions[0];
		sizes[i*4+1] = steps[i] * dimensions[0];
        sizes[i*4+2] = array->nd * sizeof(npy_intp);
        sizes[i*4+3] = array->nd * sizeof(npy_intp);
	}
	data_pointers[i*4] = steps; /* 'steps' kernel argument */
	sizes[i*4] = info->nops * sizeof(npy_intp);

	for (i = 0; i < nargs; i++) {
		/* This works best when data is contiguous !!! */
		if (alloc_and_copy(data_pointers[i], sizes[i],
						   &device_pointers[i], stream) < 0)
			goto error;
		kernelargs[i] = &device_pointers[i];
	}

	/* Launch kernel & check result */
	/* TODO: use multiple thread blocks */
	cu_result = cuLaunchKernel(info->cu_func,
							   dimensions[0], 1, 1,
							   1, 1, 1,
							   0 /* sharedMemBytes */, stream,
							   kernelargs, 0);

	if (cu_result != CUDA_SUCCESS) {
		PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result));
		goto error;
	}

    /* Wait for kernel to finish */
	error_code = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("device synchronization", error_code)

    goto cleanup;
error:
    (void) cudaGetLastError(); /* clear error */
	result = -1;
cleanup:
    /* TODO: error handling */
	for (i = 0; i < nargs; i++) {
		if (!device_pointers[i])
			break;

        if (i % 4 == 1) {
            /* Only copy data back */
            (void) cudaMemcpy(data_pointers[i], (void *) device_pointers[i],
                              sizes[i], cudaMemcpyDeviceToHost);
        }
		(void) cudaFree(device_pointers[i]);
	}
	free(array_copies);
	(void) cudaStreamDestroy(stream);
	return result;
}

void
cuda_outer_loop(char **args, npy_intp *dimensions, npy_intp *steps, void *data,
                PyObject **arrays)
{
    (void) _cuda_outer_loop(args, dimensions, steps, data, arrays);
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

      case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        return "context is destroyed";

      default:
        return "invalid/unknown error code";
    }
}
