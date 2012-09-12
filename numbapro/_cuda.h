#include <Python.h>
#include <cuda.h>

typedef struct {
    /* max total threads per block */
    int MAX_THREADS_PER_BLOCK;
    /* blocks per grid */
    int MAX_GRID_DIM_X;
    int MAX_GRID_DIM_Y;
    int MAX_GRID_DIM_Z;
    /* threads per block */
    int MAX_BLOCK_DIM_X;
    int MAX_BLOCK_DIM_Y;
    int MAX_BLOCK_DIM_Z;
    /* max device memory */
    int MAX_SHARED_MEMORY;
} CudaDeviceAttrs;

extern void init_cuda_exc_type(PyObject *exc_type);
extern int init_attributes(CudaDeviceAttrs *attrs);
extern int cuda_load(PyObject *ptx_str, CUmodule *cu_module);
extern int cuda_getfunc(CUmodule cu_module, CUfunction *cu_func, char *funcname);

extern int
invoke_cuda_ufunc(PyUFuncObject *ufunc, CudaDeviceAttrs *device_attrs,
                  CUfunction cu_func, PyListObject *inputs,
                  PyObject *out, int copy_in, int copy_out, void **out_mem,
                  unsigned int griddimx, unsigned int griddimy,
                  unsigned int griddimz, unsigned int blockdimx,
                  unsigned int blockdimy, unsigned int blockdimz);
