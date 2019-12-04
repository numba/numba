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
struct numba_oneapi_device_t
{
    // TODO : Add members to store more device related information such as name
    void *device;
    void *context;
    void *queue;
    unsigned int max_work_item_dims;
};

typedef struct numba_oneapi_device_t device_t;

/*! @struct numba_oneapi_platform_t
 *  @brief Stores the various information for a given platform/driver including
 *  an array of devices for this platform.
 *
 */
struct numba_oneapi_platform_t
{
    char *platform_name;
    unsigned num_devices;
};

typedef struct numba_oneapi_platform_t platform_t;


struct numba_oneapi_buffer_t
{
    void *buffer;
    // mem flag
    // context
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
 * @brief Create memory buffers.
 *
 * @param[in] context_ptr- The context in which the buffer needs to be created.
 * @param[in/out] buffs  - An array of buffer_t objects that would be
 *                         initialized by create_numba_oneapi_mem_buffers.
 * @param[in] nbuffers   - The number of buffer objects in the buffs array.
 * @param[in] mem_flags  - An array of flags indicating the type of buffer.
 *                         Currently, we only support the flags in mem_flags_t.
 *                         The size of this array needs to be equal to nbuffers.
 * @param[in] buffsizes  - An array of size_t indicating the size of each
 *                         buffer. The size of this array needs to be equal to
 *                         nbuffers.
 *
 * @return An error code indicating if all the buffers were successfully created
 *         or not.
 */
int create_numba_oneapi_mem_buffers (const void *context_ptr,
                                     buffer_t buffs[],
                                     size_t nbuffers,
                                     const mem_flags_t mem_flags[],
                                     const size_t buffsizes[]);


int create_numba_oneapi_rw_mem_buffer (const void *context_ptr,
                                       buffer_t *buff,
                                       const size_t buffsize);


/*!
 * @brief Frees the memory allocated for each buffer_t object.
 *
 * @param[in] buffs    - The array of buffer_t objects.
 * @param[in] nbuffers -  The size of the buffs array.
 *
 * @return An error code indicating if all the buffers were successfully
 *         destroyed or not.
 */
int destroy_numba_oneapi_mem_buffers (buffer_t buffs[], size_t nbuffers);


int destroy_numba_oneapi_rw_mem_buffer (buffer_t *buff);


/*!
 *
 */
int write_numba_oneapi_mem_buffer_to_device (const void *queue_ptr,
                                             buffer_t buff,
                                             bool blocking_copy,
                                             size_t offset,
                                             size_t buffersize,
                                             const void* data_ptr);


/*!
 *
 */
int read_numba_oneapi_mem_buffer_from_device (const void *queue_ptr,
                                              buffer_t buff,
                                              bool blocking_copy,
                                              size_t offset,
                                              size_t buffersize,
                                              void* data_ptr);


/*!
 * @brief Creates all the boilerplate around creating an OpenCL kernel from a
 * source string. CreateProgram, BuildProgram, CreateKernel, CreateKernelArgs,
 * EnqueueKernels.
 *
 */
int enqueue_numba_oneapi_kernel_from_source (const device_t *device_ptr,
                                             const char **program_src,
                                             const char *kernel_name,
                                             const buffer_t buffers[],
                                             size_t nbuffers,
                                             unsigned int work_dim,
                                             const size_t *global_work_offset,
                                             const size_t *global_work_size,
                                             const size_t *local_work_size);

/*!
 *
 */
int retain_numba_oneapi_context (const void *context_ptr);


/*!
 *
 */
int release_numba_oneapi_context (const void *context_ptr);


//---- TODO:

// 1. Add release/retain methods for buffers

//---------


/*!
 * @brief Helper function to print out information about the platform and
 * devices available to this runtime.
 */
int dump_numba_oneapi_runtime_info (const runtime_t rt);


int dump_device_info (const device_t *device_ptr);

#endif /*--- NUMBA_ONEAPI_GLUE_H_ ---*/
