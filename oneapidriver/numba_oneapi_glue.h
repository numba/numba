#ifndef NUMBA_ONEAPI_GLUE_H_
#define NUMBA_ONEAPI_GLUE_H_

#include <stdbool.h>
#include <stdlib.h>

enum NUMBA_ONEAPI_GLUE_ERROR_CODES
{
    NUMBA_ONEAPI_SUCCESS = 0,
    NUMBA_ONEAPI_FAILURE = -1
};


typedef enum NUMBA_ONEAPI_GLUE_MEM_FLAGS
{
    NUMBA_ONEAPI_READ_WRITE = 0x0,
    NUMBA_ONEAPI_WRITE_ONLY,
    NUMBA_ONEAPI_READ_ONLY
} mem_flags_t;


/*!
 *
 */
struct numba_oneapi_env
{
    // TODO : Add members to store more device related information such as name
    void *context;
    void *device;
    void *queue;
    unsigned int max_work_item_dims;
    int (*dump_fn) (void *);
};

typedef struct numba_oneapi_env* env_t;


struct numba_oneapi_buffer
{
    void *buffer;
    // This may, for example, be a cl_mem pointer
    void *buffer_ptr;
    // Stores the size of the buffer_ptr (e.g sizeof(cl_mem))
    size_t sizeof_buffer_ptr;
};

typedef struct numba_oneapi_buffer* buffer_t;


struct numba_oneapi_kernel
{
    void *kernel;
};

typedef struct numba_oneapi_kernel* kernel_t;


struct numba_oneapi_program
{
    void *program;
};

typedef struct numba_oneapi_program* program_t;


struct numba_oneapi_kernel_arg
{
    const void *arg_value;
    size_t arg_size;
};

typedef struct numba_oneapi_kernel_arg* kernel_arg_t;


/*! @struct numba_oneapi_runtime_t
 *  @brief Stores an array of the available OpenCL or Level-0 platform/drivers.
 *
 *  @var numba_oneapi_runtime_t::num_platforms
 *  Depicts the number of platforms/drivers available on this system
 *
 *  @var numba_oneapi_runtime_t::platforms_ids
 *  An array of OpenCL platforms.
 *
 *  @var numba_one_api_runtime_t::platform_infos
 *  An array of platform_t objects each corresponding to an OpenCL platform
 */
struct numba_oneapi_runtime
{
    unsigned num_platforms;
    void *platform_ids;
    //platform_t *platform_infos;
    bool has_cpu;
    bool has_gpu;
    env_t first_cpu_env;
    env_t first_gpu_env;
    int (*dump_fn) (void *);
};

typedef struct numba_oneapi_runtime* runtime_t;


/*!
 * @brief Initializes a new oneapi_runtime_t object
 *
 * @param[in/out] rt - An uninitialized runtime_t pointer that is initialized
 *                     by the function.
 *
 * @return An error code indicating if the runtime_t object was successfully
 *         initialized.
 */
int create_numba_oneapi_runtime (runtime_t *rt);


/*!
 * @brief Free the runtime and all its resources.
 *
 * @param[in] rt - Pointer to the numba_one_api_runtime_t object to be freed
 *
 * @return An error code indicating if resource freeing was successful.
 */
int destroy_numba_oneapi_runtime (runtime_t *rt);


/*!
 *
 */
int create_numba_oneapi_rw_mem_buffer (env_t env_t_ptr,
                                       size_t buffsize,
                                       buffer_t *buff);


int destroy_numba_oneapi_rw_mem_buffer (buffer_t *buff);


/*!
 *
 */
int write_numba_oneapi_mem_buffer_to_device (env_t env_t_ptr,
                                             buffer_t buff,
                                             bool blocking_copy,
                                             size_t offset,
                                             size_t buffersize,
                                             const void *data_ptr);


/*!
 *
 */
int read_numba_oneapi_mem_buffer_from_device (env_t env_t_ptr,
                                              buffer_t buff,
                                              bool blocking_copy,
                                              size_t offset,
                                              size_t buffersize,
                                              void *data_ptr);


/*!
 *
 */
int create_numba_oneapi_program_from_spirv (env_t env_t_ptr,
                                            const void *il,
                                            size_t length,
                                            program_t *program_t_ptr);


/*!
 *
 */
int create_numba_oneapi_program_from_source (env_t env_t_ptr,
                                             unsigned int count,
                                             const char **strings,
                                             const size_t *lengths,
                                             program_t *program_t_ptr);

/*!
 *
 */
int destroy_numba_oneapi_program (program_t *program_t_ptr);


int build_numba_oneapi_program (env_t env_t_ptr, program_t program_t_ptr);

/*!
 *
 */
int create_numba_oneapi_kernel (env_t env_t_ptr,
                                program_t program_ptr,
                                const char *kernel_name,
                                kernel_t *kernel_ptr);


int destroy_numba_oneapi_kernel (kernel_t *kernel_ptr);


/*!
 *
 */
int create_numba_oneapi_kernel_arg (const void *arg_value,
                                    size_t arg_size,
                                    kernel_arg_t *kernel_arg_t_ptr);


/*!
 *
 */
int destroy_numba_oneapi_kernel_arg (kernel_arg_t *kernel_arg_t_ptr);


/*!
 *
 */
int set_args_and_enqueue_numba_oneapi_kernel (env_t env_t_ptr,
                                              kernel_t kernel_t_ptr,
                                              size_t nargs,
                                              const kernel_arg_t *array_of_args,
                                              unsigned int work_dim,
                                              const size_t *global_work_offset,
                                              const size_t *global_work_size,
                                              const size_t *local_work_size);

#if 0
/*!
 * @brief Creates all the boilerplate around creating an OpenCL kernel from a
 * source string. CreateProgram, BuildProgram, CreateKernel, CreateKernelArgs,
 * EnqueueKernels.
 *
 */
int enqueue_numba_oneapi_kernel_from_source (env_t env_t_ptr,
                                             const char **program_src,
                                             const char *kernel_name,
                                             const buffer_t buffers[],
                                             size_t nbuffers,
                                             unsigned int work_dim,
                                             const size_t *global_work_offset,
                                             const size_t *global_work_size,
                                             const size_t *local_work_size);
#endif

/*!
 *
 */
int retain_numba_oneapi_context (env_t env_t_ptr);


/*!
 *
 */
int release_numba_oneapi_context (env_t env_t_ptr);


//---- TODO:

// 1. Add release/retain methods for buffers

//---------

#endif /*--- NUMBA_ONEAPI_GLUE_H_ ---*/
