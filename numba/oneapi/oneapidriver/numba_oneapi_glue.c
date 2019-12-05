#include "numba_oneapi_glue.h"
#include <assert.h>
#include <stdio.h>
#include <CL/cl.h>  /* OpenCL headers */

// TODO : Add branches to check for OpenCL error codes and print relevant error
//        messages. Then there would be no need to pass in the message string

// FIXME : The error check macro needs to be improved. Currently, we encounter
// an error and goto the error label. Directly going to the error label can lead
// to us not releasing resources prior to returning from the function. To work
// around this situation, add a stack to store all the objects that should be
// released prior to returning. The stack gets populated as a function executes
// and on encountering an error, all objects on the stack get properly released
// prior to returning. (Look at enqueue_numba_oneapi_kernel_from_source for a
// ghastly example where we really need proper resource management.)

// FIXME : memory allocated in a function should be released in the error
// section

#define CHECK_OPEN_CL_ERROR(x, M) do {                                         \
    int retval = (x);                                                          \
    switch(retval) {                                                           \
    case 0:                                                                    \
        break;                                                                 \
    case -36:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, "[CL_INVALID_COMMAND_QUEUE]command_queue is not a "    \
                        "valid command-queue.",                                \
                __LINE__, __FILE__);                                           \
        goto error;                                                            \
    case -45:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, "[CL_INVALID_PROGRAM_EXECUTABLE] no successfully "     \
                        "built program executable available for device "       \
                        "associated with command_queue.",                      \
                __LINE__, __FILE__);                                           \
        goto error;                                                            \
    default:                                                                   \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, M, __LINE__, __FILE__);                                \
        goto error;                                                            \
    }                                                                          \
} while(0)


#define CHECK_MALLOC_ERROR(type, x) do {                                       \
    type * ptr = (type*)(x);                                                   \
    if(ptr == NULL) {                                                          \
        fprintf(stderr, "Malloc Error for type %s on Line %d in %s",           \
                #type, __LINE__, __FILE__);                                    \
        perror(" ");                                                           \
        free(ptr);                                                             \
        ptr = NULL;                                                            \
        goto malloc_error;                                                     \
    }                                                                          \
} while(0)


#define CHECK_NUMBA_ONEAPI_GLUE_ERROR(x, M) do {                               \
    int retval = (x);                                                          \
    switch(retval) {                                                           \
    case 0:                                                                    \
        break;                                                                 \
    case -1:                                                                   \
        fprintf(stderr, "Numba-Oneapi-Glue Error: %d (%s) on Line %d in %s\n", \
                retval, M, __LINE__, __FILE__);                                \
        goto error;                                                            \
    default:                                                                   \
        fprintf(stderr, "Numba-Oneapi-Glue Error: %d (%s) on Line %d in %s\n", \
                retval, M, __LINE__, __FILE__);                                \
        goto error;                                                            \
    }                                                                          \
} while(0)


/*------------------------------- Private helpers ----------------------------*/


static int get_platform_name (cl_platform_id platform, char **platform_name)
{
    cl_int err;
    size_t n;

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, *platform_name, &n);
    CHECK_OPEN_CL_ERROR(err, "Could not get platform name length.");

    // Allocate memory for the platform name string
    *platform_name = (char*)malloc(sizeof(char)*n);
    CHECK_MALLOC_ERROR(char*, *platform_name);

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, n, *platform_name,
            NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get platform name.");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    free(*platform_name);
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
static int dump_device_info (void *obj)
{
    cl_int err;
    char *value;
    size_t size;
    cl_uint maxComputeUnits;
    env_t env_t_ptr;

    env_t_ptr = (env_t)obj;

    cl_device_id device = (cl_device_id)(env_t_ptr->device);

    err = clRetainDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not retain device.");

    // print device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get device name.");
    value = (char*)malloc(size);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get device name.");
    printf("Device: %s\n", value);
    free(value);

    // print hardware device version
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get device version.");
    value = (char*) malloc(size);
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get device version.");
    printf("Hardware version: %s\n", value);
    free(value);

    // print software driver version
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get driver version.");
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get driver version.");
    printf("Software version: %s\n", value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get open cl version.");
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get open cl version.");
    printf("OpenCL C version: %s\n", value);
    free(value);

    // print parallel compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get number of compute units.");
    printf("Parallel compute units: %d\n", maxComputeUnits);

    err = clReleaseDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not release device.");

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 * @brief Helper function to print out information about the platform and
 * devices available to this runtime.
 *
 */
static int dump_numba_oneapi_runtime_info (void *obj)
{
    size_t i;
    runtime_t rt;

    rt = (runtime_t)obj;

    if(rt) {
        printf("Number of platforms : %d\n", rt->num_platforms);
        cl_platform_id *platforms = rt->platform_ids;
        for(i = 0; i < rt->num_platforms; ++i) {
            char *platform_name = NULL;
            get_platform_name(platforms[i], &platform_name);
            printf("Platform #%ld: %s\n", i, platform_name);
            free(platform_name);
        }
    }

    return NUMBA_ONEAPI_SUCCESS;
}


/*!
 *
 */
static int get_first_device (cl_platform_id* platforms,
                             cl_uint platformCount,
                             cl_device_id *device,
                             cl_device_type device_ty)
{
    cl_int status;
    cl_uint ndevices = 0;
    unsigned int i;

    for (i = 0; i < platformCount; ++i) {
        // get all devices of env_ty
        status = clGetDeviceIDs(platforms[i], device_ty, 0, NULL, &ndevices);
        // If this platform has no devices of this type then continue
        if(!ndevices) continue;

        // get the first device
        status = clGetDeviceIDs(platforms[i], device_ty, 1, device, NULL);
        CHECK_OPEN_CL_ERROR(status, "Could not get first cl_device_id.");

        // If the first device of this type was discovered, no need to look more
        if(ndevices) break;
    }

    if(ndevices)
        return NUMBA_ONEAPI_SUCCESS;
    else
        return NUMBA_ONEAPI_FAILURE;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
static int create_numba_oneapi_env_t (cl_platform_id* platforms,
                                      size_t nplatforms,
                                      cl_device_type device_ty,
                                      env_t *env_t_ptr)
{
    cl_int err;
    int err1;
    env_t env;
    cl_device_id *device;

    env = NULL;
    device = NULL;

    // Allocate the env_t object
    env = (env_t)malloc(sizeof(struct numba_oneapi_env_t));
    CHECK_MALLOC_ERROR(env_t, env);

    device = (cl_device_id*)malloc(sizeof(cl_device_id));

    err1 = get_first_device(platforms, nplatforms, device, device_ty);
    CHECK_NUMBA_ONEAPI_GLUE_ERROR(err1, "Failed inside get_first_device");

    // get the CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS for this device
    err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(env->max_work_item_dims), &env->max_work_item_dims, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get max work item dims");

    // Create a context and associate it with device
    env->context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create device context.");
    // Create a queue and associate it with the context
    env->queue = clCreateCommandQueueWithProperties((cl_context)env->context,
            *device, 0, &err);

    CHECK_OPEN_CL_ERROR(err, "Could not create command queue.");

    env->device = *device;
    env ->dump_fn = dump_device_info;
    free(device);
    *env_t_ptr = env;

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    free(env);
    return NUMBA_ONEAPI_FAILURE;
}


static int destroy_numba_oneapi_env_t (env_t *env_t_ptr)
{
    cl_int err;

    err = clReleaseCommandQueue((cl_command_queue)(*env_t_ptr)->queue);
    CHECK_OPEN_CL_ERROR(err, "Could not release command queue.");
    err = clReleaseDevice((cl_device_id)(*env_t_ptr)->device);
    CHECK_OPEN_CL_ERROR(err, "Could not release device.");
    err = clReleaseContext((cl_context)(*env_t_ptr)->context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context.");

    free(*env_t_ptr);

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 * @brief Initialize the runtime object.
 */
static int init_runtime_t_obj (runtime_t rt)
{
    cl_int status;
    int ret;
    cl_platform_id *platforms;

    // get count of available platforms
    status = clGetPlatformIDs(0, NULL, &(rt->num_platforms));
    CHECK_OPEN_CL_ERROR(status, "Could not get platform count.");

    if(!rt->num_platforms) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        goto error;
    }

    // Allocate memory for the platforms array
    rt->platform_ids = (cl_platform_id*)malloc(
                           sizeof(cl_platform_id)*rt->num_platforms
                       );
    CHECK_MALLOC_ERROR(cl_platform_id, rt->platform_ids);

    // Get the platforms
    status = clGetPlatformIDs(rt->num_platforms, rt->platform_ids, NULL);
    CHECK_OPEN_CL_ERROR(status, "Could not get platform ids");
    // Cast rt->platforms to a pointer of type cl_platform_id, as we cannot do
    // pointer arithmetic on void*.
    platforms = (cl_platform_id*)rt->platform_ids;
    // Get the first cpu device on this platform
    ret = create_numba_oneapi_env_t(platforms, rt->num_platforms,
                                    CL_DEVICE_TYPE_CPU, &rt->first_cpu_env);
    rt->has_cpu = !ret;

#if DEBUG
    if(rt->has_cpu)
        printf("DEBUG: CPU device acquired...\n");
    else
        printf("DEBUG: No CPU available on the system\n");
#endif

    // Get the first gpu device on this platform
    ret = create_numba_oneapi_env_t(platforms, rt->num_platforms,
                                    CL_DEVICE_TYPE_GPU, &rt->first_gpu_env);
    rt->has_gpu = !ret;

#if DEBUG
    if(rt->has_gpu)
        printf("DEBUG: GPU device acquired...\n");
    else
        printf("DEBUG: No GPU available on the system.\n");
#endif

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:

    return NUMBA_ONEAPI_FAILURE;
error:
    free(rt->platform_ids);

    return NUMBA_ONEAPI_FAILURE;
}

/*-------------------------- End of private helpers --------------------------*/


/*!
 * @brief Initializes a new oneapi_runtime_t object
 *
 */
int create_numba_oneapi_runtime (runtime_t *rt)
{
    int err;
    runtime_t rtobj;

    rtobj = NULL;
    // Allocate a new struct numba_oneapi_runtime_t object
    rtobj = (runtime_t)malloc(sizeof(struct numba_oneapi_runtime_t));
    CHECK_MALLOC_ERROR(runtime_t, rt);

    rtobj->num_platforms = 0;
    rtobj->platform_ids  = NULL;
    err = init_runtime_t_obj(rtobj);
    CHECK_NUMBA_ONEAPI_GLUE_ERROR(err, "Could not initialize runtime object.");
    rtobj->dump_fn = dump_numba_oneapi_runtime_info;

    *rt = rtobj;
#if DEBUG
    printf("DEBUG: Created an new numba_oneapi_runtime object\n");
#endif
    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    free(rtobj);
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 * @brief Free the runtime and all its resources.
 *
 */
int destroy_numba_oneapi_runtime (runtime_t *rt)
{
    int err;
#if DEBUG
    printf("DEBUG: Going to destroy the numba_oneapi_runtime object\n");
#endif
    // free the first_cpu_device
    err = destroy_numba_oneapi_env_t(&(*rt)->first_cpu_env);
    CHECK_NUMBA_ONEAPI_GLUE_ERROR(err, "Could not destroy first_cpu_device.");

    // free the first_gpu_device
    err = destroy_numba_oneapi_env_t(&(*rt)->first_gpu_env);
    CHECK_NUMBA_ONEAPI_GLUE_ERROR(err, "Could not destroy first_gpu_device.");

    // free the platforms
    free((cl_platform_id*)(*rt)->platform_ids);
    // free the runtime_t object
    free(*rt);

#if DEBUG
    printf("DEBUG: Destroyed the new numba_oneapi_runtime object\n");
#endif
    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int retain_numba_oneapi_context (env_t env_t_ptr)
{
    cl_int err;
    cl_context context;

    context = (cl_context)(env_t_ptr->context);
    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed when calling clRetainContext.");

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int release_numba_oneapi_context (env_t env_t_ptr)
{
    cl_int err;
    cl_context context;

    context = (cl_context)(env_t_ptr->context);
    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed when calling clRetainContext.");

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


int create_numba_oneapi_rw_mem_buffer (env_t env_t_ptr,
                                       size_t buffsize,
                                       buffer_t *buffer_t_ptr)
{
    cl_int err;
    buffer_t buff;
    cl_context context;

    buff = NULL;

    // Get the context from the device
    context = (cl_context)(env_t_ptr->context);
    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain context.");

    // Allocate a numba_oneapi_buffer_t object
    buff = (buffer_t)malloc(sizeof(struct numba_oneapi_buffer_t));
    CHECK_MALLOC_ERROR(buffer_t, buffer_t_ptr);

    // Create the OpenCL buffer.
    // NOTE : Copying of data from host to device needs to happen explicitly
    // using clEnqueue[Write|Read]Buffer. This would change in the future.
    buff->buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffsize, NULL,
                                  &err);
    CHECK_OPEN_CL_ERROR(err, "Failed to create CL buffer.");
#if DEBUG
    printf("DEBUG: CL RW buffer created...\n");
#endif
    *buffer_t_ptr = buff;
    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed to release context.");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    free(buff);
    return NUMBA_ONEAPI_FAILURE;
}


int destroy_numba_oneapi_rw_mem_buffer (buffer_t *buff)
{
    cl_int err;

    err = clReleaseMemObject((cl_mem)(*buff)->buffer);
    CHECK_OPEN_CL_ERROR(err, "Failed to release CL buffer.");
    free(*buff);

#if DEBUG
    printf("DEBUG: CL buffer destroyed...\n");
#endif

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


int write_numba_oneapi_mem_buffer_to_device (env_t env_t_ptr,
                                             buffer_t buffer_t_ptr,
                                             bool blocking,
                                             size_t offset,
                                             size_t buffersize,
                                             const void* data_ptr)
{
    cl_int err;
    cl_command_queue queue;
    cl_mem mem;

    queue = (cl_command_queue)env_t_ptr->queue;
    mem = (cl_mem)buffer_t_ptr->buffer;

#if DEBUG
    assert(mem && "buffer memory is NULL");
#endif

    err = clRetainMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");
    err = clRetainCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the buffer memory object.");

    // Not using any events for the time being. Eventually we want to figure
    // out the event dependencies using parfor analysis.
    err = clEnqueueWriteBuffer(queue, mem, blocking?CL_TRUE:CL_FALSE,
            offset, buffersize, data_ptr, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Failed to write to CL buffer.");

    err = clReleaseCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the command queue.");
    err = clReleaseMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the buffer memory object.");

    //--- TODO: Implement a version that uses clEnqueueMapBuffer

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


int read_numba_oneapi_mem_buffer_from_device (env_t env_t_ptr,
                                              buffer_t buffer_t_ptr,
                                              bool blocking,
                                              size_t offset,
                                              size_t buffersize,
                                              void* data_ptr)
{
    cl_int err;
    cl_command_queue queue;
    cl_mem mem;

    queue = (cl_command_queue)env_t_ptr->queue;
    mem = (cl_mem)buffer_t_ptr->buffer;

    err = clRetainMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");
    err = clRetainCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");

    // Not using any events for the time being. Eventually we want to figure
    // out the event dependencies using parfor analysis.
    err = clEnqueueReadBuffer(queue, mem, blocking?CL_TRUE:CL_FALSE,
            offset, buffersize, data_ptr, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Failed to read from CL buffer.");

    err = clReleaseCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the command queue.");
    err = clReleaseMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the buffer memory object.");

    //--- TODO: Implement a version that uses clEnqueueMapBuffer

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


int create_numba_oneapi_program_from_spirv (env_t env_t_ptr,
                                            const void *il,
                                            size_t length,
                                            program_t *program_t_ptr)
{
    cl_int err;
    cl_context context;
    program_t prog;

    prog = NULL;

    prog = (program_t)malloc(sizeof(struct numba_oneapi_program_t));
    CHECK_MALLOC_ERROR(program_t, program_t_ptr);

    context = (cl_context)env_t_ptr->context;

    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    // Create a program with a SPIR-V file
    prog->program = clCreateProgramWithIL(context, il, length, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create program with IL");
#if DEBUG
    printf("DEBUG: CL program created from spirv...\n");
#endif

    *program_t_ptr = prog;

    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    free(prog);
    return NUMBA_ONEAPI_FAILURE;
}


int create_numba_oneapi_program_from_source (env_t env_t_ptr,
                                             unsigned int count,
                                             const char **strings,
                                             const size_t *lengths,
                                             program_t *program_t_ptr)
{
    cl_int err;
    cl_context context;
    program_t prog;

    prog = NULL;
    prog = (program_t)malloc(sizeof(struct numba_oneapi_program_t));
    CHECK_MALLOC_ERROR(program_t, program_t_ptr);

    context = (cl_context)env_t_ptr->context;

    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    // Create a program with string source files
    prog->program = clCreateProgramWithSource(context, count, strings,
            lengths, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create program with source");
#if DEBUG
    printf("DEBUG: CL program created from source...\n");
#endif

    *program_t_ptr = prog;

    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    free(prog);
    return NUMBA_ONEAPI_FAILURE;
}


int destroy_numba_oneapi_program (program_t *program_ptr)
{
    cl_int err;

    err = clReleaseProgram((cl_program)(*program_ptr)->program);
    CHECK_OPEN_CL_ERROR(err, "Failed to release CL program.");
    free(*program_ptr);

#if DEBUG
    printf("DEBUG: CL program destroyed...\n");
#endif

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


int build_numba_oneapi_program (env_t env_t_ptr, program_t program_t_ptr)
{
    cl_int err;
    cl_device_id device;

    device = (cl_device_id)env_t_ptr->device;
    err = clRetainDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not retain device");
    // Build (compile) the program for the device
    err = clBuildProgram((cl_program)program_t_ptr->program, 1, &device, NULL,
            NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not build program");
#if DEBUG
    printf("DEBUG: CL program successfully built.\n");
#endif
    err = clReleaseDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not release device");

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int create_numba_oneapi_kernel (env_t env_t_ptr,
                                program_t program_t_ptr,
                                const char *kernel_name,
                                kernel_t *kernel_ptr)
{
    cl_int err;
    cl_context context;
    kernel_t ker;

    ker = (kernel_t)malloc(sizeof(struct numba_oneapi_kernel_t));
    CHECK_MALLOC_ERROR(kernel_t, kernel_ptr);

    context = (cl_context)(env_t_ptr->context);
    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    ker->kernel = clCreateKernel((cl_program)(program_t_ptr->program),
            kernel_name, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create kernel");
    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");
#if DEBUG
    printf("DEBUG: CL kernel created\n");
#endif

    *kernel_ptr = ker;
    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    return NUMBA_ONEAPI_FAILURE;
}


int destroy_numba_oneapi_kernel (kernel_t *kernel_ptr)
{
    cl_int err;

    err = clReleaseKernel((cl_kernel)(*kernel_ptr)->kernel);
    CHECK_OPEN_CL_ERROR(err, "Failed to release CL kernel.");
    free(*kernel_ptr);

#if DEBUG
    printf("DEBUG: CL kernel destroyed...\n");
#endif

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}

#if 0
/*!
 *
 */
int enqueue_numba_oneapi_kernel_from_source (const env_t env_t_ptr,
                                             const char **program_src,
                                             const char *kernel_name,
                                             const buffer_t buffers[],
                                             size_t nbuffers,
                                             unsigned int work_dim,
                                             const size_t *global_work_offset,
                                             const size_t *global_work_size,
                                             const size_t *local_work_size)
{
    size_t i;
    cl_int err;
    cl_device_id *device;
    cl_context *context;
    cl_command_queue *queue;
    cl_program program;
    cl_kernel kernel;

    device = (cl_device_id*)device_ptr->device;
    context = (cl_context*)device_ptr->context;
    queue = (cl_command_queue*)device_ptr->queue;

    err = clRetainContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    err = clRetainCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(err, "Could not retain command queue");

    if(work_dim > device_ptr->max_work_item_dims) {
        fprintf(stderr, "ERROR: %s at in %s on Line %d\n.",
                "Invalid value for work_dim. Cannot be greater than "
                "CL_MAX_WORK_ITEM_DIMENSIONS.", __FILE__, __LINE__);
        goto error;
    }

    // Create a program with source code
    program = clCreateProgramWithSource(*context, 1, program_src, NULL, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create program with source");

    // Build (compile) the program for the device
    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not build program");

    // Create the vector addition kernel
    kernel = clCreateKernel(program, kernel_name, &err);
    if(err) {
        fprintf(stderr, "Could not create the OpenCL kernel.\n");
        err = clReleaseProgram(program);
        CHECK_OPEN_CL_ERROR(err, "Could not release the program");
        goto error;
    }

    // Retain all the memory buffers
    for(i = 0; i < nbuffers; ++i) {
        err = clRetainMemObject((cl_mem)(buffers[i]->buffer));
        CHECK_OPEN_CL_ERROR(err, "Could not retain the buffer mem object.");
    }

    // Set the kernel arguments
    err = 0;
    for(i = 0; i < nbuffers; ++i) {
        err |= clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[i]->buffer);
    }

    if(err) {
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",
                        err, "Could not set kernel argument.",
                        __LINE__, __FILE__);
        err = clReleaseProgram(program);
        if(err) {
            fprintf(stderr, "Could not set kernel argument.\n");
            err = clReleaseKernel(kernel);
            CHECK_OPEN_CL_ERROR(err, "Could not release the kernel");
        }
        goto error;
    }

    // Execute the kernel (Again not using events for the time being)
    err = clEnqueueNDRangeKernel(*queue, kernel, work_dim, global_work_offset,
            global_work_size, local_work_size, 0, NULL, NULL);

    // Release resources
    err = clReleaseProgram(program);
    if(err) {
        fprintf(stderr, "Could not set kernel argument.\n");
        err = clReleaseKernel(kernel);
        CHECK_OPEN_CL_ERROR(err, "Could not release the kernel");
        goto error;
    }
    err = clReleaseKernel(kernel);
    CHECK_OPEN_CL_ERROR(err, "Could not release the kernel");

    // Release all the memory buffers
    for(i = 0; i < nbuffers; ++i) {
        err = clReleaseMemObject((cl_mem)(buffers[i]->buffer));
        CHECK_OPEN_CL_ERROR(err, "Could not release the buffer mem object.");
    }

    err = clReleaseCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(err, "Could not release queue");
    err = clReleaseContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}
#endif
