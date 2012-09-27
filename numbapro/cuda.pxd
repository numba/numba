cimport numpy as cnp
from cpython cimport PyObject

cdef extern from "cuda.h":
    ctypedef void *CUdevice
    ctypedef void *CUcontext
    ctypedef void *CUmodule
    ctypedef void *CUfunction

cdef extern from "_cuda.h": # external utilities from _cuda.c
    ctypedef struct CudaDeviceAttrs:
        # /* max total threads per block */
        int MAX_THREADS_PER_BLOCK
        # /* blocks per grid */
        int MAX_GRID_DIM_X
        int MAX_GRID_DIM_Y
        int MAX_GRID_DIM_Z
        # /* threads per block */
        int MAX_BLOCK_DIM_X
        int MAX_BLOCK_DIM_Y
        int MAX_BLOCK_DIM_Z
        # /* max device memory */
        int MAX_SHARED_MEMORY
        int COMPUTE_CAPABILITY_MAJOR
        int COMPUTE_CAPABILITY_MINOR

    ctypedef struct CudaFunctionAndData:
        CUfunction cu_func
        int nops

    int init_cuda_exc_type() except -1

    int get_device(CUdevice *cu_device, CUcontext *cu_context,
                   int device_number) except -1
    int init_attributes(CUdevice cu_device, CudaDeviceAttrs *attrs) except -1
    int cuda_load(object ptx_str, CUmodule *cu_module) except -1
    int cuda_getfunc(CUmodule cu_module, CUfunction *cu_func,
                     char *funcname) except -1
    int dealloc(CUmodule, CUcontext) except -1

    int invoke_cuda_ufunc(cnp.ufunc ufunc, CudaDeviceAttrs *device_attrs,
                          CUfunction cu_func, list inputs,
                          object out, int copy_in, int copy_out,
                          unsigned int griddimx, unsigned int griddimy,
                          unsigned int griddimz, unsigned int blockdimx,
                          unsigned int blockdimy, unsigned int blockdimz) except -1

    void cuda_outer_loop(char **args, cnp.npy_intp *dimensions, cnp.npy_intp *steps,
                         void *func, PyObject **arrays)
# XXX: Unused?
#    ctypedef struct CudaFunctionAndData:
#        CUfunction cu_func
#        int nops
#        int nout

    int cuda_numba_function(object args, CUfunction func,
                          unsigned int griddimx, unsigned int griddimy,
                          unsigned int griddimz, unsigned int blockdimx,
                          unsigned int blockdimy, unsigned int blockdimz) except -1

