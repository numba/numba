#include <Python.h>
#include <cuda.h>

#define MAXARGS 32

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
    int COMPUTE_CAPABILITY_MAJOR;
    int COMPUTE_CAPABILITY_MINOR;
} CudaDeviceAttrs;

typedef struct {
    CUfunction cu_func;
    int nops;
} CudaFunctionAndData;

typedef struct {
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

} CudaAPI;

extern int is_cuda_api_initialized(void);
void set_cuda_api_initialized(void);
extern CudaAPI * get_cuda_api_ref(void);
extern int init_cuda_exc_type(void);
extern int get_device(CUdevice *cu_device, CUcontext *cu_context,
                      int device_number);
extern int init_attributes(CUdevice cu_device, CudaDeviceAttrs *attrs);
extern int cuda_load(PyObject *ptx_str, CUmodule *cu_module);
extern int cuda_getfunc(CUmodule cu_module, CUfunction *cu_func, char *funcname);
extern int dealloc(CUmodule, CUcontext);

extern int
invoke_cuda_ufunc(PyUFuncObject *ufunc, CudaDeviceAttrs *device_attrs,
                  CUfunction cu_func, PyListObject *inputs,
                  PyObject *out, int copy_in, int copy_out,
                  unsigned int griddimx, unsigned int griddimy,
                  unsigned int griddimz, unsigned int blockdimx,
                  unsigned int blockdimy, unsigned int blockdimz);

void cuda_outer_loop(char **args, npy_intp *dimensions, npy_intp *steps,
                     void *func, PyObject **arrays);

int cuda_numba_function(PyListObject *args, void *func,
                          unsigned int griddimx, unsigned int griddimy,
                          unsigned int griddimz, unsigned int blockdimx,
                          unsigned int blockdimy, unsigned int blockdimz,
                          char * typemap);



