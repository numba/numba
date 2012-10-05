#include <Python.h>
#include <cuda.h>
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "_cuda.h"
#include "_internal.h"

/* process wide globals */
static CUcontext global_context = NULL;
static CUdevice  *global_device  = NULL;

static struct {
    CUresult (*Init)(unsigned int Flags);
    CUresult (*DeviceGetCount)(int *count);
    CUresult (*DeviceGet)(CUdevice *device, int ordinal);
    CUresult (*DeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
    CUresult (*DeviceComputeCapability)(int *major, int *minor, CUdevice dev);
    CUresult (*CtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
    CUresult (*ModuleLoadDataEx)(CUmodule *module, const void *image,
                                 unsigned int numOptions, CUjit_option *options,
                                 void **optionValues);
    CUresult (*ModuleUnload)(CUmodule hmod);
    CUresult (*ModuleGetFunction)(CUfunction *hfunc, CUmodule hmod,
                                    const char *name);
    CUresult (*MemAlloc)(CUdeviceptr *dptr, size_t bytesize);
    CUresult (*MemcpyHtoD)(CUdeviceptr dstDevice, const void *srcHost,
                             size_t ByteCount);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr dstDevice, const void *srcHost,
                                  size_t ByteCount, CUstream hStream);
    CUresult (*MemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice,
                             unsigned int ByteCount);
    CUresult (*MemcpyDtoHAsync)(void *dstHost, CUdeviceptr srcDevice,
                                  unsigned int ByteCount, CUstream hStream);
    CUresult (*MemFree)(CUdeviceptr dptr);
    CUresult (*StreamCreate)(CUstream *phStream, unsigned int Flags);
    CUresult (*StreamDestroy)(CUstream hStream);
    CUresult (*StreamSynchronize)(CUstream hStream);
    CUresult (*LaunchKernel)(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ, unsigned int sharedMemBytes,
                             CUstream hStream, void **kernelParams, void **extra);

} cuda_api;


/* prototypes */
static const char *curesult_to_str(CUresult e);

#define CHECK_CUDA_RESULT(cu_result)                                \
    if (cu_result != CUDA_SUCCESS) {                                \
        PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result)); \
        return -1;                                                  \
    }

#define CHECK_CUDA_RESULT_MSG(msg, cu_result)                                \
    if (cu_result != CUDA_SUCCESS) {                                \
        PyErr_Format(cuda_exc_type, "%s failed: %s",  \
                     msg, curesult_to_str(cu_result)); \
        return -1;                                                  \
    }


#define CHECK_CUDA_MEM_ERR(action)                                          \
        if (cu_result != CUDA_SUCCESS) {                                    \
            PyErr_Format(cuda_exc_type, "Got '%s' for memory %s",           \
                         curesult_to_str(cu_result), action);           \
            goto error;                                                     \
        }

#define BUFSIZE 128

static PyObject *cuda_exc_type;


static void
import_numpy_array(void)
{
    import_array();
}

int
init_cuda_api(void)
{
    cuda_api.Init = &cuInit;
    cuda_api.DeviceGetCount = &cuDeviceGetCount;
    cuda_api.DeviceGet = &cuDeviceGet;
    cuda_api.DeviceGetAttribute = &cuDeviceGetAttribute;
    cuda_api.DeviceComputeCapability = &cuDeviceComputeCapability;
    cuda_api.CtxCreate = &cuCtxCreate;
    cuda_api.ModuleLoadDataEx = &cuModuleLoadDataEx;
    cuda_api.ModuleUnload = &cuModuleUnload;
    cuda_api.ModuleGetFunction = &cuModuleGetFunction;
    cuda_api.MemAlloc = &cuMemAlloc;
    cuda_api.MemcpyHtoD = &cuMemcpyHtoD;
    cuda_api.MemcpyHtoDAsync = &cuMemcpyHtoDAsync;
    cuda_api.MemcpyDtoH = &cuMemcpyDtoH;
    cuda_api.MemcpyDtoHAsync = &cuMemcpyDtoHAsync;
    cuda_api.MemFree = &cuMemFree;
    cuda_api.StreamCreate = &cuStreamCreate;
    cuda_api.StreamDestroy = &cuStreamDestroy;
    cuda_api.StreamSynchronize = &cuStreamSynchronize;
    cuda_api.LaunchKernel = &cuLaunchKernel;

    return 0;
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
        cu_result = cuda_api.ModuleUnload(cu_module);
        CHECK_CUDA_RESULT(cu_result)
    }
//    if (cu_context) {
//        cu_result = cuCtxDestroy(cu_context);
//        CHECK_CUDA_RESULT(cu_result)
//    }
    return 0;
}

int
get_device(CUdevice *cu_device, CUcontext *cu_context, int device_number)
{
    CUresult cu_result;

    if (device_number < 0 && !global_device) {
        int i, device_count;

        cuda_api.Init(0); //initialize the driver api

        cu_result = cuda_api.DeviceGetCount(&device_count);
        CHECK_CUDA_RESULT_MSG("get CUDA device count", cu_result)

        global_device = malloc(sizeof(CUdevice)); // never freed

        for (i = 0; i < device_count; i++) {
            cu_result = cuda_api.DeviceGet(global_device, i);
            if (cu_result == CUDA_SUCCESS) {
                device_number = i;
                break;
            }
        }
        if (device_number < 0) {
            PyErr_SetString(cuda_exc_type, "No usable devices found");
            return -1;
        }
    }

    if (cu_device) {
        *cu_device = *global_device;
    }

    if (!global_context) {
        cu_result = cuda_api.CtxCreate(&global_context, 0, *global_device);
        CHECK_CUDA_RESULT(cu_result)
    }
    if (cu_context) {
        *cu_context = global_context;
    }

    return 0;
}

int
init_attributes(CUdevice cu_device, CudaDeviceAttrs *attrs)
{
    CUresult cu_result;
    CudaDeviceAttrs out;

#define _GETATTR(value, attr) \
        cu_result = cuda_api.DeviceGetAttribute(&out.value, attr, cu_device); \
        CHECK_CUDA_RESULT(cu_result);

    _GETATTR(MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    _GETATTR(MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    _GETATTR(MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    _GETATTR(MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    _GETATTR(MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    _GETATTR(MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    _GETATTR(MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    _GETATTR(MAX_SHARED_MEMORY, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    cu_result = cuda_api.DeviceComputeCapability(&out.COMPUTE_CAPABILITY_MAJOR,
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

    /* Capture error log */
    CUjit_option options[2];
    void *values[2];

    cu_errors[0] = '\0';

    if (!PyBytes_Check(ptx_str)) {
        PyErr_SetString(PyExc_TypeError, "Expected byte string PTX assembly");
        return -1;
    }

    ptx = PyBytes_AS_STRING(ptx_str);

    options[0] = CU_JIT_ERROR_LOG_BUFFER;
    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    values[0] = &cu_errors[0];
    values[1] = &bufsize;

    cu_result = cuda_api.ModuleLoadDataEx(cu_module, ptx, 2, options, values);

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
    cu_result = cuda_api.ModuleGetFunction(cu_func, cu_module, funcname);
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

    for (i = 0; i < ufunc->nin + 1; i++) {
        PyObject *array = PyList_GET_ITEM(inputs, i);
        void *data = PyArray_DATA(array);
        npy_intp size = PyArray_NBYTES(array);

        /* Allocate memory on device for array */
        cu_result = cuda_api.MemAlloc(&device_pointers[i], size);
        CHECK_CUDA_MEM_ERR("allocation")
        args[i] = &device_pointers[i];

        if (i != ufunc->nin || copy_in) {
            /* Copy array to device, skip 'out' unless 'copy_in' is true */
            cu_result = cuda_api.MemcpyHtoD(device_pointers[i], data, size);
            CHECK_CUDA_MEM_ERR("copy to device")
        }
    }
    total_count = (int) PyArray_SIZE(out);
    args[ufunc->nin + 1] = &total_count;

    /* Launch kernel & check result */
    cu_result = cuda_api.LaunchKernel(cu_func, griddimx, griddimy, griddimz,
                               blockdimx, blockdimy, blockdimz,
                               0 /* sharedMemBytes */, 0 /* hStream */,
                               args, 0);

    if (cu_result != CUDA_SUCCESS) {
        PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result));
        goto error;
    }

    /* Wait for kernel to finish */
    //error_code = cuDeviceSynchronize();
    //CHECK_CUDA_ERROR("device synchronization", error_code)

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

        cu_result = cuda_api.MemcpyDtoH(data, device_pointers[i], size);
        CHECK_CUDA_MEM_ERR("copy to host")

        cu_result = cuda_api.MemFree(device_pointers[i]);
        CHECK_CUDA_MEM_ERR("free")
    }
    /* Deallocate packed arguments */
    free(device_pointers);
    free(args);

    return retval;
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
alloc_and_copy(void *data, size_t size, void **result, CUstream stream)
{
    CUresult cu_result;
	CUdeviceptr p;

    /* printf("Allocating chunk of %d bytes\n", (int) size); */

	cu_result = cuda_api.MemAlloc(&p, size);
	CHECK_CUDA_MEM_ERR("allocation")

    if (stream)
        cu_result = cuda_api.MemcpyHtoDAsync(p, data, size, stream);
    else
	    cu_result = cuda_api.MemcpyHtoD(p, data, size);

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
#ifdef _WIN32
static __inline int
#else
static inline int
#endif
_cuda_outer_loop(char **args, npy_intp *dimensions, npy_intp *steps, void *data,
                 PyObject **arrays)
{
    npy_intp i, j;
    CudaFunctionAndData *info = (CudaFunctionAndData *) data;
    int result = 0;

    CUstream stream = 0;
    CUresult cu_result;

    int dim_count;
    int total_size;

    CUdeviceptr device_args, device_dims, device_steps, device_arylen;
    const npy_intp device_count = dimensions[0];

    CUdeviceptr temp_args[MAXARGS] = {0};
    CUdeviceptr temp_dims[MAXARGS] = {0};
    CUdeviceptr temp_steps[MAXARGS] = {0};

    void * kernel_args[] = {&device_args, &device_dims, &device_steps,
                            &device_arylen, &device_count};

    int thread_per_block = dimensions[0];
	int block_per_grid = 1;

	/* XXX: assume a smaller thread limit to prevent CC support problem
	        and out of register problem */
	const int MAX_THREAD = 128;

    /*
    TODO: Use multiple streams to divide up the work since the order does not
    matter.
    */
    cu_result = cuda_api.StreamCreate(&stream, 0);
    CHECK_CUDA_RESULT_MSG("Creating a CUDA stream", cu_result)

    dim_count = 0;
    total_size = 0;
    for (i=0; i<info->nops; ++i){
        dim_count += ((ndarray*)arrays[i])->nd;
        total_size += steps[i] * dimensions[0];
    }

    // arguments
    for (i = 0; i < info->nops; ++i){
        if( alloc_and_copy(args[i], steps[i] * dimensions[0],
                           &temp_args[i], stream) < 0 )
            goto error;
    }

    if( alloc_and_copy(&temp_args, sizeof(void*) * info->nops,
                       &device_args, stream) < 0 )
        goto error;

    // dimensions
    for (i = 0; i < info->nops; ++i){
        ndarray *array = arrays[i];
        if( alloc_and_copy(array->dimensions, sizeof(npy_intp) * array->nd,
                           &temp_dims[i], stream) < 0 )
            goto error;
    }

    if( alloc_and_copy(&temp_dims, sizeof(void*) * info->nops,
                       &device_dims, stream) < 0 )
        goto error;

    // steps
    for (i = 0; i < info->nops; ++i){
        ndarray *array = arrays[i];
        if( alloc_and_copy(array->strides, sizeof(npy_intp) * array->nd,
                           &temp_steps[i], stream) < 0 )
            goto error;
    }

    if( alloc_and_copy(&temp_steps, sizeof(void*) * info->nops,
                       &device_steps, stream) < 0 )
        goto error;

    // outer loop step
    if( alloc_and_copy(steps, sizeof(npy_intp) * info->nops,
                       &device_arylen, stream) < 0 )
        goto error;


	/* Launch kernel & check result */
	/* TODO: use multiple thread blocks */

	if (thread_per_block >= MAX_THREAD) {
	    block_per_grid = thread_per_block / MAX_THREAD;
	    block_per_grid += thread_per_block % MAX_THREAD ? 1 : 0;
	    thread_per_block = MAX_THREAD;
	}

	cu_result = cuda_api.LaunchKernel(info->cu_func,
	                           block_per_grid, 1, 1,
							   thread_per_block, 1, 1,
							   0 /* sharedMemBytes */ , stream,
							   kernel_args, 0);

	if (cu_result != CUDA_SUCCESS) {
		PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result));
		goto error;
	}

    /* Wait for kernel to finish
    NOTE: cudaDeviceSynchronize should not be necessary.
    */
    //	error_code = cudaDeviceSynchronize();
    //	CHECK_CUDA_ERROR("device synchronization", error_code)

    /* Copy the output array */
    /* TODO: error handling */
    cu_result = cuda_api.MemcpyDtoHAsync(args[info->nops - 1],
                                  temp_args[info->nops - 1],
                                  steps[info->nops - 1] * dimensions[0],
                                  stream);
    CHECK_CUDA_MEM_ERR("retrieve result")

    cu_result = cuda_api.StreamSynchronize(stream);
    CHECK_CUDA_RESULT_MSG("stream synchronize", cu_result)

    goto cleanup;
error:
    //(void) cudaGetLastError(); /* clear error */
	result = -1;
cleanup:
    /* TODO: error handling */

    for(i = 0; i < info->nops; ++i ){
        cuda_api.MemFree(temp_args[i]);
        cuda_api.MemFree(temp_dims[i]);
    }

    cuda_api.MemFree(device_args);
    cuda_api.MemFree(device_dims);
    cuda_api.MemFree(device_steps);
    cuda_api.MemFree(device_arylen);

	(void) cuda_api.StreamDestroy(stream);
	return result;
}

void
cuda_outer_loop(char **args, npy_intp *dimensions, npy_intp *steps, void *data,
                PyObject **arrays)
{
    (void) _cuda_outer_loop(args, dimensions, steps, data, arrays);
}


int cuda_numba_function(PyListObject *args, void *func,
                        unsigned int griddimx, unsigned int griddimy,
                        unsigned int griddimz, unsigned int blockdimx,
                        unsigned int blockdimy, unsigned int blockdimz,
                        char * typemap)
{
    int result = 0;
    int i, j;
    CUresult cu_result;
    CUstream stream = 0;

    const long nargs = PyList_GET_SIZE(args);

    ndarray *arrays[MAXARGS] = {NULL};
    ndarray tmparys[MAXARGS];
    memset(tmparys, 0, sizeof(tmparys));
    CUdeviceptr device_pointers[MAXARGS] = {NULL};
    void* host_pointers[MAXARGS] = {NULL};
    void* kernel_args[MAXARGS] = {NULL};

    cu_result = cuda_api.StreamCreate(&stream, 0);
    CHECK_CUDA_RESULT_MSG("Creating a CUDA stream", cu_result);

    /* Prepare arguments */
    for (i=0; i<nargs; ++i){
        PyObject * pyobj = PyList_GET_ITEM(args, i);
        if (PyArray_Check(pyobj)) { // is pyarray
            ndarray *ary = (ndarray*) pyobj;
            ndarray *tmpary = &tmparys[i];
            arrays[i] = ary;


            if (alloc_and_copy(ary->data, ary->strides[0] * ary->dimensions[0],
                               &tmpary->data, stream) < 0)
                goto error;

            if (alloc_and_copy(ary->dimensions, sizeof(npy_intp) * ary->nd,
                               &tmpary->dimensions, stream) < 0)
                goto error;

            if (alloc_and_copy(ary->strides, sizeof(npy_intp) * ary->nd,
                               &tmpary->strides, stream) < 0)
                goto error;

            tmpary->nd = ary->nd;

            if (alloc_and_copy(tmpary, sizeof(ndarray),
                               &device_pointers[i], stream) < 0)
                goto error;

            kernel_args[i] = &device_pointers[i];
        } else if (PyInt_Check(pyobj)) {
            long value = PyInt_AsLong(pyobj);
            if (PyErr_Occurred()) {
                goto error;
            }
            host_pointers[i] = malloc(sizeof(value));
            memcpy(host_pointers[i], &value, sizeof(value));
            kernel_args[i] = host_pointers[i];
        } else if (PyFloat_Check(pyobj)) {
            double value = PyFloat_AsDouble(pyobj);
            if (PyErr_Occurred()) {
                goto error;
            }
            if (typemap[i] == 'd'){
                host_pointers[i] = malloc(sizeof(value));
                memcpy(host_pointers[i], &value, sizeof(value));
            } else if (typemap[i] == 'f'){
                float truncated = value;
                host_pointers[i] = malloc(sizeof(truncated));
                memcpy(host_pointers[i], &truncated, sizeof(truncated));
            } else {
                PyErr_SetString(PyExc_TypeError, "Invalid float argument");
                goto error;
            }

            kernel_args[i] = host_pointers[i];
        } else {
            PyErr_SetString(PyExc_TypeError, "Type not handled");
            goto error;
        }
    }

    cu_result = cuda_api.LaunchKernel(func,
	                           griddimx, griddimy, griddimz,
							   blockdimx, blockdimy, blockdimz,
							   0 /* sharedMemBytes */ , stream,
							   kernel_args, 0);

	if (cu_result != CUDA_SUCCESS) {
		PyErr_SetString(cuda_exc_type, curesult_to_str(cu_result));
		goto error;
	}

    /* Retrieve results */
    for (i=0; i<nargs; ++i){
        ndarray *ary = arrays[i];
        if (ary) {
            ndarray *tmpary = &tmparys[i];

            cu_result = cuda_api.MemcpyDtoHAsync(ary->data, tmpary->data,
                                          ary->strides[0] * ary->dimensions[0],
                                          stream);
            CHECK_CUDA_MEM_ERR("retrieve data")

            cu_result = cuda_api.MemcpyDtoHAsync(ary->dimensions, tmpary->dimensions,
                                          sizeof(npy_intp) * ary->nd,
                                          stream);
            CHECK_CUDA_MEM_ERR("retrieve dimension")

            cu_result = cuda_api.MemcpyDtoHAsync(ary->strides, tmpary->strides,
                                          sizeof(npy_intp) * ary->nd,
                                          stream);
            CHECK_CUDA_MEM_ERR("retrieve strides")
        }
    }

    cu_result = cuda_api.StreamSynchronize(stream);
    CHECK_CUDA_RESULT_MSG("stream synchronize", cu_result)

    goto cleanup;
error:
    //(void) cudaGetLastError(); /* clear error */
    result = -1;
cleanup:
    for (i=0; i<nargs; ++i){
        if (arrays[i]) {
            cuda_api.MemFree(tmparys[i].data);
            cuda_api.MemFree(tmparys[i].dimensions);
            cuda_api.MemFree(tmparys[i].strides);
            cuda_api.MemFree(device_pointers[i]);
        } else {
            free(host_pointers[i]);
        }
    }
    cuda_api.StreamDestroy(stream);
    return result;
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
