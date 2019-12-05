#include <numba_oneapi_glue.h>
#include <stdio.h>
#include <CL/cl.h>  /* OpenCL headers */

typedef enum
{
    ON_CPU,
    ON_GPU
} execution_ty;

// Array sizes
static const size_t N = 2048;

/* OpenCl kernel for element-wise addition of two arrays */
const char* programSource =
    "__kernel                                                             \n"
    "void vecadd(__global float *A, __global float *B, __global float *C) \n"
    "{                                                                    \n"
    "   int idx = get_global_id(0);                                       \n"
    "   C[idx] = A[idx] + B[idx];                                         \n"
    "}";
#if 0
void buildAndExecuteKernel (const runtime_t rt, execution_ty ex)
{
    int err;
    device_t device;
    void *context;
    void *queue;
    program_t program_ptr;
    kernel_t kernel_ptr;
    float *A, *B, *C;
    size_t i;
    size_t datasize;
    size_t num_buffers = 3;
    size_t indexSpaceSize[1], workGroupSize[1];
    buffer_t buffers[3];

    mem_flags_t memflags[3] = {
            NUMBA_ONEAPI_READ_ONLY,
            NUMBA_ONEAPI_READ_ONLY,
            NUMBA_ONEAPI_WRITE_ONLY
    };
    size_t datasizes[3] = {
            sizeof(float)*N,
            sizeof(float)*N,
            sizeof(float)*N
    };

    if(ex == ON_CPU) {
        device = rt->first_cpu_device;
        context = rt->first_cpu_device.context;
        queue = rt->first_cpu_device.queue;
    }
    else if(ex == ON_GPU) {
        device = rt->first_gpu_device;
        context = rt->first_gpu_device.context;
        queue = rt->first_gpu_device.queue;
    }

    // Memory requirement
    datasize = sizeof(float)*N;
    // Allocate space for the input/output arrays on host
    if((A = (float*)malloc(datasize)) == NULL) { perror("Error: "); exit(1); }
    if((B = (float*)malloc(datasize)) == NULL) { perror("Error: "); exit(1); }
    if((C = (float*)malloc(datasize)) == NULL) { perror("Error: "); exit(1); }

    //---- Initialize the input data
    for(i = 0; i < N; ++i) {
        A[i] = i+1;
        B[i] = 2*(i+1);
    }

    // Allocate two input buffers and one output buffers for the vectors
    //err = create_numba_oneapi_mem_buffers(context, buffers, num_buffers,
    //        memflags, datasizes);

    err =  create_numba_oneapi_rw_mem_buffer(context, &buffers[0], datasize);
    err |= create_numba_oneapi_rw_mem_buffer(context, &buffers[1], datasize);
    err |= create_numba_oneapi_rw_mem_buffer(context, &buffers[2], datasize);

    if(err) {
        fprintf(stderr, "Buffer creation failed. Abort!\n");
        exit(1);
    }

    // Write data from the input arrays to the buffers
    err = write_numba_oneapi_mem_buffer_to_device(queue, buffers[0], true,
            0, datasize, A);
    err |= write_numba_oneapi_mem_buffer_to_device(queue, buffers[1], true,
            0, datasize, B);
    if(err) {
        fprintf(stderr, "Could not write to buffer. Abort!\n");
        exit(1);
    }

    err = create_and_build_numba_oneapi_program_from_source(&device, 1,
            (const char **)&programSource, NULL, &program_ptr);
    if(err) {
        fprintf(stderr, "Could not create the program. Abort!\n");
        exit(1);
    }
    err = create_numba_oneapi_kernel(context, program_ptr, "vecadd",
            &kernel_ptr);
    if(err) {
        fprintf(stderr, "Could not create the kernel. Abort!\n");
        exit(1);
    }

#if 0
    // There are 'N' work-items
    indexSpaceSize[0] = N;
    workGroupSize[0] = 256;

    // Create a program with source code
    err = enqueue_numba_oneapi_kernel_from_source(
            &device,
            (const char **)&programSource,
            "vecadd",
            buffers,
            num_buffers,
            1,
            NULL,
            indexSpaceSize,
            workGroupSize);

    if(err) {
        fprintf(stderr, "ERROR (%d): Could not build OpenCL program. Abort!\n",
                err);
        exit(1);
    }
#endif
    // Copy the device output buffer to the host output array
    err = read_numba_oneapi_mem_buffer_from_device(queue, buffers[0], true, 0,
            datasize, C);

#if 1
    // Validate the output
    for(i = 0; i < N; ++i) {
        //if(C[i] != (i+1 + 2*(i+1))) {
        if(C[i] != A[i]) {
            printf("Position %ld Wrong Result\n", i);
            printf("%s", "Stop validating and exit...\n");
            exit(1);
        }
    }
    printf("Results Match\n");
#endif

    // Cleanup
    // free the kernel
    destroy_numba_oneapi_kernel(&kernel_ptr);
    // free the program
    destroy_numba_oneapi_program(&program_ptr);
    // free the buffers
    //destroy_numba_oneapi_mem_buffers(buffers, num_buffers);
    destroy_numba_oneapi_rw_mem_buffer(&buffers[0]);
    destroy_numba_oneapi_rw_mem_buffer(&buffers[1]);
    destroy_numba_oneapi_rw_mem_buffer(&buffers[2]);
    // free allocated memory for the arrays
    free(A);
    free(B);
    free(C);
}
#endif

int main (int argc, char** argv)
{
    runtime_t rt;
    int err;

    err = create_numba_oneapi_runtime(&rt);
    if(err == NUMBA_ONEAPI_FAILURE) goto error;
    rt->dump_fn(rt);

    printf("\n===================================\n\n");
    //--- Execute on CPU
    printf("Executing on the first CPU device info: \n");
    rt->first_cpu_env->dump_fn(rt->first_cpu_env);
    //buildAndExecuteKernel(rt, ON_CPU);

    printf("\n===================================\n\n");

    printf("Executing on the first GPU device info: \n");
    rt->first_gpu_env->dump_fn(rt->first_gpu_env);
    //buildAndExecuteKernel(rt, ON_GPU);

    printf("\n===================================\n\n");

    //--- Cleanup
    destroy_numba_oneapi_runtime(&rt);

    return 0;

error:
    return NUMBA_ONEAPI_FAILURE;
}
