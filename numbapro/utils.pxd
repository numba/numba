
cdef extern from *: # external utilities from _cuda.c
    ctypedef struct CudaDeviceAttrs:
        # /* max total threads per block */
        int MAX_THREADS_PER_BLOCK;
        # /* blocks per grid */
        int MAX_GRID_DIM_X;
        int MAX_GRID_DIM_Y;
        int MAX_GRID_DIM_Z;
        # /* threads per block */
        int MAX_BLOCK_DIM_X;
        int MAX_BLOCK_DIM_Y;
        int MAX_BLOCK_DIM_Z;
        # /* max device memory */
        int MAX_SHARED_MEMORY;

    int init_attributes(CudaDeviceAttrs *attrs) except -1

    int invoke_cuda_ufunc(object ufunc, list inputs, object out) except -1

    void init_cuda_exc_type(PyObject *exc_type);
    int init_attributes(CudaDeviceAttrs *attrs);
    int cuda_load(PyObject *ptx_str, CUmodule *cu_module);
    int cuda_getfunc(CUmodule cu_module, CUfunction *cu_func, char *funcname);

    int
    invoke_cuda_ufunc(PyUFuncObject *ufunc, CudaDeviceAttrs *device_attrs,
                      CUfunction cu_func, PyListObject *inputs,
                      PyObject *out, int copy_in, int copy_out, void **out_mem,
                      unsigned int griddimx, unsigned int griddimy,
                      unsigned int griddimz, unsigned int blockdimx,
                      unsigned int blockdimy, unsigned int blockdimz);

