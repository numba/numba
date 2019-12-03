from cffi import FFI
import os
import sys


ffibuilder = FFI()

BAD_ENV_PATH_ERRMSG = """
NUMBA_ONEAPI_GLUE_HOME is set to '{0}' which is not a valid path to a
dynamic link library for your system.
"""

def _raise_bad_env_path(path, extra=None):
    error_message=BAD_ENV_PATH_ERRMSG.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)

oneapiGlueHome = os.environ.get('NUMBA_ONEAPI_GLUE_HOME', None)

if oneapiGlueHome is None:
    raise ValueError("FATAL: Set the NUMBA_ONEAPI_GLUE_HOME for "
                     "numba_oneapi_glue.h and libnumbaoneapiglue.so")

if oneapiGlueHome is not None:
    try:
        oneapi_glue_home = os.path.abspath(oneapiGlueHome)
    except ValueError:
        _raise_bad_env_path(oneapiGlueHome)

    if not os.path.isfile(oneapiGlueHome+"/lib/libnumbaoneapiglue.so"):
        _raise_bad_env_path(oneapiGlueHome+"/lib/libnumbaoneapiglue.so")

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef("""
                enum NUMBA_ONEAPI_GLUE_ERROR_CODES
                {
                    NUMBA_ONEAPI_SUCCESS = 0,
                    NUMBA_ONEAPI_FAILURE = -1
                };
                

                typedef enum NUMBA_ONEAPI_GLUE_MEM_FLAGS
                {
                    NUMBA_ONEAPI_READ_WRITE = 0x0,
                    NUMBA_ONEAPI_WRITE_ONLY,
                    NUMBA_ONEAPI_READ_ONLY,
                } mem_flags_t;
                
                
                struct numba_oneapi_device_info_t
                {
                    void *device;
                    void *context;
                    void *queue;
                    unsigned int max_work_item_dims;
                };
                
                
                typedef struct numba_oneapi_device_info_t device_t;
                
                struct numba_oneapi_platform_t
                {
                    char *platform_name;
                    unsigned num_devices;
                };
                
                
                typedef struct numba_oneapi_platform_t platform_t;
                
                
                struct numba_oneapi_buffer_t
                {
                    void *buffer;
                };
                
                typedef struct numba_oneapi_buffer_t* buffer_t;
                
                
                struct numba_oneapi_kernel_t
                {
                    void *kernel;
                };
                
                typedef struct numba_oneapi_kernel_t kernel_t;
                
                
                struct numba_oneapi_program_t
                {
                    void *program;
                };
                
                typedef struct numba_oneapi_program_t program_t;
                
                
                struct numba_oneapi_runtime_t
                {
                    unsigned num_platforms;
                    void *platform_ids;
                    platform_t *platform_infos;
                    bool has_cpu;
                    bool has_gpu;
                    device_t first_cpu_device;
                    device_t first_gpu_device;
                };
                
                typedef struct numba_oneapi_runtime_t* runtime_t;
                
                
                int create_numba_oneapi_runtime (runtime_t *rt);
                int destroy_numba_oneapi_runtime (runtime_t *rt);
                int create_numba_oneapi_rw_mem_buffer (const void *context_ptr,
                                                       buffer_t *buff,
                                                       const size_t buffsize);
                int destroy_numba_oneapi_rw_mem_buffer (buffer_t *buff);
                int retain_numba_oneapi_context (const void *context_ptr);
                int release_numba_oneapi_context (const void *context_ptr);
                int dump_numba_oneapi_runtime_info (const runtime_t rt);
                int dump_device_info (const device_t *device_ptr);
                int write_numba_oneapi_mem_buffer_to_device (const void *q_ptr,
                                                             buffer_t buff,
                                                             bool blocking,
                                                             size_t offset,
                                                             size_t buffersize,
                                                             const void* d_ptr);
                int read_numba_oneapi_mem_buffer_from_device (const void *q_ptr,
                                                              buffer_t buff,
                                                              bool blocking,
                                                              size_t offset,
                                                              size_t buffersize,
                                                              void* d_ptr);
            """)

ffibuilder.set_source(
    "_numba_oneapi_pybindings",
    """
         #include "numba_oneapi_glue.h"   // the C header of the library
    """,
    libraries=["numbaoneapiglue", "OpenCL"],
    include_dirs=[oneapiGlueHome + "/include"],
    library_dirs=[oneapiGlueHome + "/lib"]
)   # library name, for the linker


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)