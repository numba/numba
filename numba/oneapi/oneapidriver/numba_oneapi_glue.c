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

#define CHECK_OPEN_CL_ERROR(x, M) do {                                         \
    int retval = (x);                                                          \
    switch(retval) {                                                           \
    case 0:                                                                    \
        break;                                                                 \
    case -36:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, "command_queue is not a valid command-queue.",         \
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

/*------------------------------- Private helpers ----------------------------*/


static int
set_platform_name (const cl_platform_id* platform, platform_t *pd)
{
    cl_int status;
    size_t len;

    status = clGetPlatformInfo(*platform, CL_PLATFORM_NAME, 0,
            pd->platform_name, &len);
    CHECK_OPEN_CL_ERROR(status, "Could not get platform name length.");

    // Allocate memory for the platform name string
    pd->platform_name = (char*)malloc(sizeof(char)*len);
    CHECK_MALLOC_ERROR(char, pd->platform_name);

    status = clGetPlatformInfo(*platform, CL_PLATFORM_NAME, len,
            pd->platform_name, NULL);
    CHECK_OPEN_CL_ERROR(status, "Could not get platform name.");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
static int
initialize_cl_platform_infos (const cl_platform_id* platform,
                               platform_t *pd)
{
    cl_int status;

    if((set_platform_name(platform, pd)) == NUMBA_ONEAPI_FAILURE)
        goto error;

    // get the number of devices on this platform
    status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_ALL, 0, NULL,
            &pd->num_devices);
    CHECK_OPEN_CL_ERROR(status, "Could not get device count.");

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
static int
get_first_device (const cl_platform_id* platforms, cl_uint platformCount,
                  cl_device_id *device, cl_device_type device_ty)
{
    cl_int status;
    cl_uint ndevices = 0;
    unsigned int i;

    for (i = 0; i < platformCount; ++i) {
        // get all devices of device_ty
        status = clGetDeviceIDs(platforms[i], device_ty, 0, NULL, &ndevices);
        // If this platform has no devices of this type then continue
        if(!ndevices) continue;

        // get the first device
        status = clGetDeviceIDs(platforms[i], device_ty, 1, device, NULL);
        CHECK_OPEN_CL_ERROR(status, "Could not get first cl_device_id.");

        // If the first device of this type was discovered, no need to look more
        if(ndevices)
            break;
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
static int
initialize_cl_device_info (cl_platform_id* platforms, size_t nplatforms,
                           device_t *d, cl_device_type device_ty)
{
    cl_int err;
    int ret;
    cl_device_id *device;
    cl_context *context;
    cl_command_queue *queue;

    device = (cl_device_id*)malloc(sizeof(cl_device_id));
    CHECK_MALLOC_ERROR(cl_device_id, device);


    ret = get_first_device(platforms, nplatforms, device, device_ty);

    // If there are no devices of device_ty then do not allocate memory for the
    // device, context and queue. Instead, set the values to NULL.
    if(ret == NUMBA_ONEAPI_FAILURE) {
        free(device);
        d->device = NULL;
        d->context = NULL;
        d->queue = NULL;
        goto error;
    }

    // get the CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS for this device
    err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(d->max_work_item_dims), &d->max_work_item_dims, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get max work item dims");

    context = (cl_context*)malloc(sizeof(cl_context));
    CHECK_MALLOC_ERROR(cl_context, context);
    queue = (cl_command_queue*)malloc(sizeof(cl_command_queue));
    CHECK_MALLOC_ERROR(cl_command_queue, queue);

    // Create a context and associate it with device
    *context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create device context.");
    // Create a queue and associate it with the context
    *queue = clCreateCommandQueueWithProperties(*context, *device, 0, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create command queue.");

    d->device = device;
    d->context = context;
    d->queue = queue;

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    return NUMBA_ONEAPI_FAILURE;
}


static int
destroy_cl_device_info (device_t *d)
{
    cl_int status;
    cl_command_queue *queue;

    queue = (cl_command_queue*)d->queue;
    status = clReleaseCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(status, "Could not release command queue.");
    free(d->queue);

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 * @brief Initialize the runtime object.
 */
static int
initialize_runtime (runtime_t rt)
{
    cl_int status;
    int ret;
    size_t i;
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

    // Allocate memory for the platform_info array
    rt->platform_infos = (platform_t*)malloc(
                                sizeof(platform_t)*rt->num_platforms
                           );
    CHECK_MALLOC_ERROR(platform_t, rt->platform_infos);

    // Cast rt->platforms to a pointer of type cl_platform_id, as we cannot do
    // pointer arithmetic on void*.
    platforms = (cl_platform_id*)rt->platform_ids;

    // Initialize the platform_infos
    for(i = 0; i < rt->num_platforms; ++i) {
        // Initialize the platform_t object
        (rt->platform_infos+i)->platform_name = NULL;
        //(rt->platform_infos+i)->devices       = NULL;
        (rt->platform_infos+i)->num_devices   = 0;

        if((status = initialize_cl_platform_infos(
               platforms+i, rt->platform_infos+i)) == NUMBA_ONEAPI_FAILURE)
            goto error;

        printf("DEBUG: Platform name : %s\n",
               (rt->platform_infos+i)->platform_name);
    }

    // Get the first cpu device on this platform
    ret = initialize_cl_device_info(platforms, rt->num_platforms,
                                    &rt->first_cpu_device, CL_DEVICE_TYPE_CPU);
    rt->has_cpu = !ret;

#if DEBUG
    if(rt->has_cpu)
        printf("DEBUG: CPU device acquired...\n");
    else
        printf("DEBUG: No CPU available on the system\n");
#endif

    // Get the first gpu device on this platform
    ret = initialize_cl_device_info(platforms, rt->num_platforms,
                                    &rt->first_gpu_device, CL_DEVICE_TYPE_GPU);
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
    return NUMBA_ONEAPI_FAILURE;
}

/*-------------------------- End of private helpers --------------------------*/


/*!
 * @brief Initializes a new oneapi_runtime_t object
 *
 */
int create_numba_oneapi_runtime (runtime_t *rt)
{
    int status;

    // Allocate a new struct numba_oneapi_runtime_t object
    runtime_t rtobj = (runtime_t)malloc(sizeof(struct numba_oneapi_runtime_t));
    CHECK_MALLOC_ERROR(runtime_t, rt);

    rtobj->num_platforms = 0;
    rtobj->platform_ids  = NULL;
    status = initialize_runtime(rtobj);
    if(status == NUMBA_ONEAPI_FAILURE)
        goto error;
    *rt = rtobj;

    printf("INFO: Created an new numba_oneapi_runtime object\n");
    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 * @brief Free the runtime and all its resources.
 *
 */
int destroy_numba_oneapi_runtime (runtime_t *rt)
{
    size_t i;
    int err;

    printf("INFO: Going to destroy the numba_oneapi_runtime object\n");
    // release all the device arrays and platform names
    for(i = 0; i < (*rt)->num_platforms; ++i) {
        free((*rt)->platform_infos[i].platform_name);
    }

    // free the array of platform_t objects
    free((*rt)->platform_infos);
    // free the first_cpu_device
    err = destroy_cl_device_info(&(*rt)->first_cpu_device);
    if(err) {
        fprintf(stderr, "ERROR %d: %s\n",
                err, "Could not destroy first_cpu_device.");
        goto error;
    }
    // free the first_gpu_device
    err = destroy_cl_device_info(&(*rt)->first_gpu_device);
    if(err) {
        fprintf(stderr, "ERROR %d: %s\n",
                err, "Could not destroy first_gpu_device.");
        goto error;
    }
    // free the platforms
    free((cl_platform_id*)(*rt)->platform_ids);
    // free the runtime_t object
    free(*rt);

    printf("INFO: Destroyed the new numba_oneapi_runtime object\n");

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int retain_numba_oneapi_context (const void *context_ptr)
{
    cl_int err;
    const cl_context *context;

    context = (const cl_context*)(context_ptr);
    err = clRetainContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Failed when calling clRetainContext.");

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int release_numba_oneapi_context (const void *context_ptr)
{
    cl_int err;
    const cl_context *context;

    context = (const cl_context*)(context_ptr);
    err = clReleaseContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Failed when calling clRetainContext.");

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 * @brief Helper function to print out information about the platform and
 * devices available to this runtime.
 *
 */
int dump_numba_oneapi_runtime_info (const runtime_t rt)
{
    size_t i;

    if(rt) {
        printf("Number of platforms : %d\n", rt->num_platforms);
        for(i = 0; i < rt->num_platforms; ++i) {
            printf("Platform %ld. %s\n",
                    i, rt->platform_infos[i].platform_name);
            printf("    Number of devices on Platform %ld : %d\n",
                    i, rt->platform_infos[i].num_devices);
        }
    }

    return NUMBA_ONEAPI_SUCCESS;
}


/*!
 *
 */
int dump_device_info (const device_t *device_ptr)
{
    cl_int err;
    char *value;
    size_t size;
    cl_uint maxComputeUnits;

    cl_device_id * device = (cl_device_id*)(device_ptr->device);

    // print device name
    err = clGetDeviceInfo(*device, CL_DEVICE_NAME, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get device name.");
    value = (char*)malloc(size);
    err = clGetDeviceInfo(*device, CL_DEVICE_NAME, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get device name.");
    printf("Device: %s\n", value);
    free(value);

    // print hardware device version
    err = clGetDeviceInfo(*device, CL_DEVICE_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get device version.");
    value = (char*) malloc(size);
    err = clGetDeviceInfo(*device, CL_DEVICE_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get device version.");
    printf("Hardware version: %s\n", value);
    free(value);

    // print software driver version
    clGetDeviceInfo(*device, CL_DRIVER_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get driver version.");
    value = (char*) malloc(size);
    clGetDeviceInfo(*device, CL_DRIVER_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get driver version.");
    printf("Software version: %s\n", value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(*device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get open cl version.");
    value = (char*) malloc(size);
    clGetDeviceInfo(*device, CL_DEVICE_OPENCL_C_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get open cl version.");
    printf("OpenCL C version: %s\n", value);
    free(value);

    // print parallel compute units
    clGetDeviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get number of compute units.");
    printf("Parallel compute units: %d\n", maxComputeUnits);

    return NUMBA_ONEAPI_SUCCESS;

error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int create_numba_oneapi_mem_buffers (const void *context_ptr,
                                     buffer_t buffs[],
                                     size_t nbuffers,
                                     const mem_flags_t mem_flags[],
                                     const size_t buffsizes[])
{
    size_t i;
    cl_int err, err1;

    // Get the context from the device
    cl_context *context = (cl_context*)(context_ptr);
    err = clRetainContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain context.");

    // create the buffers
    err = 0;
    for(i = 0; i < nbuffers; ++i) {

        // Allocate a numba_oneapi_buffer_t object
        buffs[i] = (buffer_t)malloc(sizeof(struct numba_oneapi_buffer_t));
        CHECK_MALLOC_ERROR(buffer_t, buffs[i]);

        // Create the OpenCL buffer.
        // NOTE : Copying of data from host to device needs to happen
        // explicitly using clEnqueue[Write|Read]Buffer. This would change in
        // the future.
        buffs[i]->buffer = clCreateBuffer(*context, mem_flags[i], buffsizes[i],
                NULL, &err1);
        err |= err1;
    }
    CHECK_OPEN_CL_ERROR(err, "Failed to create CL buffer.");
#if DEBUG
    printf("DEBUG: CL buffers created...\n");
#endif

    err = clReleaseContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Failed to release context.");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    return NUMBA_ONEAPI_FAILURE;
}


int create_numba_oneapi_rw_mem_buffer (const void *context_ptr,
                                       buffer_t *buff,
                                       const size_t buffsize)
{
    cl_int err;

    // Get the context from the device
    cl_context *context = (cl_context*)(context_ptr);
    err = clRetainContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain context.");

    // Allocate a numba_oneapi_buffer_t object
    buffer_t b = (buffer_t)malloc(sizeof(struct numba_oneapi_buffer_t));
    CHECK_MALLOC_ERROR(buffer_t, buff);

    // Create the OpenCL buffer.
    // NOTE : Copying of data from host to device needs to happen explicitly
    // using clEnqueue[Write|Read]Buffer. This would change in the future.
    b->buffer = clCreateBuffer(*context, CL_MEM_READ_WRITE, buffsize,
                NULL, &err);
    CHECK_OPEN_CL_ERROR(err, "Failed to create CL buffer.");
#if DEBUG
    printf("DEBUG: CL RW buffer created...\n");
#endif
    *buff = b;
    err = clReleaseContext(*context);
    CHECK_OPEN_CL_ERROR(err, "Failed to release context.");

    return NUMBA_ONEAPI_SUCCESS;

malloc_error:
    return NUMBA_ONEAPI_FAILURE;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
 *
 */
int destroy_numba_oneapi_mem_buffers (buffer_t buffs[], size_t nbuffers)
{
    size_t i;
    cl_int err;

    for(i = 0; i < nbuffers; ++i) {
        err = clReleaseMemObject((cl_mem)buffs[i]->buffer);
        free(buffs[i]);
        CHECK_OPEN_CL_ERROR(err, "Failed to release CL buffer.");
    }
#if DEBUG
    printf("DEBUG: CL buffers destroyed...\n");
#endif

    return NUMBA_ONEAPI_SUCCESS;

error:
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


int write_numba_oneapi_mem_buffer_to_device (const void *queue_ptr,
                                            buffer_t buff,
                                            bool blocking,
                                            size_t offset,
                                            size_t buffersize,
                                            const void* data_ptr)
{
    cl_int err;

    const cl_command_queue *queue = (const cl_command_queue*)queue_ptr;
    cl_mem mem = (cl_mem)buff->buffer;

#if DEBUG
    assert(mem && "buffer memory is NULL");
#endif

    err = clRetainMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");
    err = clRetainCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the buffer memory object.");

    // Not using any events for the time being. Eventually we want to figure
    // out the event dependencies using parfor analysis.
    err = clEnqueueWriteBuffer(*queue, mem, blocking?CL_TRUE:CL_FALSE,
            offset, buffersize, data_ptr, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Failed to write to CL buffer.");

    err = clReleaseCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the command queue.");
    err = clReleaseMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the buffer memory object.");

    //--- TODO: Implement a version that uses clEnqueueMapBuffer

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


int read_numba_oneapi_mem_buffer_from_device (const void *queue_ptr,
                                              buffer_t buff,
                                              bool blocking,
                                              size_t offset,
                                              size_t buffersize,
                                              void* data_ptr)
{
    cl_int err;

    const cl_command_queue *queue = (const cl_command_queue*)queue_ptr;
    cl_mem mem = (cl_mem)buff->buffer;

    err = clRetainMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");
    err = clRetainCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");

    // Not using any events for the time being. Eventually we want to figure
    // out the event dependencies using parfor analysis.
    err = clEnqueueReadBuffer(*queue, mem, blocking?CL_TRUE:CL_FALSE,
            offset, buffersize, data_ptr, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Failed to read from CL buffer.");

    err = clReleaseCommandQueue(*queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the command queue.");
    err = clReleaseMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the buffer memory object.");

    //--- TODO: Implement a version that uses clEnqueueMapBuffer

    return NUMBA_ONEAPI_SUCCESS;
error:
    return NUMBA_ONEAPI_FAILURE;
}


/*!
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
