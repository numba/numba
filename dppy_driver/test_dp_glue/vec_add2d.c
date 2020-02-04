#include <dp_glue.h>
#include <stdio.h>
#include <CL/cl.h>  /* OpenCL headers */

typedef enum
{
    ON_CPU,
    ON_GPU
} execution_ty;

// Forward declaration
void buildAndExecuteKernel (runtime_t rt, execution_ty ex);

// Array sizes
static int X = 8;
static int Y = 8;
static int N = 8*8;

/* OpenCl kernel for element-wise addition of two arrays */
const char* programSource =
    "__kernel                                                             \n"
    "void vecadd(__global float *A, __global float *B, __global float *C, int shape1) \n"
    "{                                                                    \n"
    "   int i   = get_global_id(0);                                       \n"
    "   int j   = get_global_id(1);                                       \n"
    "   C[i*shape1 + j] = A[i*shape1 + j] + B[i*shape1 +j];               \n"
    "}";

void buildAndExecuteKernel (runtime_t rt, execution_ty ex)
{
    int err;
    env_t env_t_ptr;
    program_t program_ptr;
    kernel_t kernel_ptr;
    size_t num_args = 4;
    kernel_arg_t kernel_args[num_args];
    float *A, *B, *C;
    size_t i;
    size_t datasize;
    size_t indexSpaceSize[2];
    buffer_t buffers[3];

    if(ex == ON_CPU)
        env_t_ptr = rt->first_cpu_env;
    else if(ex == ON_GPU)
        env_t_ptr = rt->first_gpu_env;

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

    err =  create_dp_rw_mem_buffer(env_t_ptr, datasize, &buffers[0]);
    err |= create_dp_rw_mem_buffer(env_t_ptr, datasize, &buffers[1]);
    err |= create_dp_rw_mem_buffer(env_t_ptr, datasize, &buffers[2]);

    if(err) {
        fprintf(stderr, "Buffer creation failed. Abort!\n");
        exit(1);
    }

    // Write data from the input arrays to the buffers
    err = write_dp_mem_buffer_to_device(env_t_ptr, buffers[0], true,
            0, datasize, A);
    err |= write_dp_mem_buffer_to_device(env_t_ptr, buffers[1], true,
            0, datasize, B);
    if(err) {
        fprintf(stderr, "Could not write to buffer. Abort!\n");
        exit(1);
    }

    err = create_dp_program_from_source(env_t_ptr, 1,
            (const char **)&programSource, NULL, &program_ptr);
    err |= build_dp_program (env_t_ptr, program_ptr);
    if(err) {
        fprintf(stderr, "Could not create the program. Abort!\n");
        exit(1);
    }
    err = create_dp_kernel(env_t_ptr, program_ptr, "vecadd",
            &kernel_ptr);
    if(err) {
        fprintf(stderr, "Could not create the kernel. Abort!\n");
        exit(1);
    }
    kernel_ptr->dump_fn(kernel_ptr);
    // Set kernel arguments
    err = create_dp_kernel_arg(&buffers[0]->buffer_ptr,
                               buffers[0]->sizeof_buffer_ptr,
                               &kernel_args[0]);
    err |= create_dp_kernel_arg(&buffers[1]->buffer_ptr,
                                buffers[1]->sizeof_buffer_ptr,
                                &kernel_args[1]);
    err |= create_dp_kernel_arg(&buffers[2]->buffer_ptr,
                                buffers[2]->sizeof_buffer_ptr,
                                &kernel_args[2]);
    err |= create_dp_kernel_arg(&Y,
                                sizeof(int),
                                &kernel_args[3]);
    if(err) {
        fprintf(stderr, "Could not create the kernel_args. Abort!\n");
        exit(1);
    }

    // There are 'N' work-items
    indexSpaceSize[0] = 8;
    indexSpaceSize[1] = 8;

    // Create a program with source code
    err = set_args_and_enqueue_dp_kernel(env_t_ptr, kernel_ptr,
            num_args, kernel_args, 2, NULL, indexSpaceSize, NULL);

    if(err) {
        fprintf(stderr, "ERROR (%d): Could not build enqueue kernel. Abort!\n",
                err);
        exit(1);
    }

    // Copy the device output buffer to the host output array
    err = read_dp_mem_buffer_from_device(env_t_ptr, buffers[2], true,
            0, datasize, C);

#if 1
    // Validate the output
    for(i = 0; i < N; ++i) {
        if(C[i] != (i+1 + 2*(i+1))) {
        //if(C[i] != A[i]) {
            printf("Wrong value at C[%ld]. Expected %ld Actual %f\n",
                    i, (i+1 + 2*(i+1)), C[i]);
            printf("%s", "Stop validating and exit...\n");
            exit(1);
        }
#if 0
        else {
            printf("Right value at C[%ld]. Expected %ld Actual %f\n",
            i, (i+1 + 2*(i+1)), C[i]);
        }
#endif
    }
    printf("Results Match\n");
#endif

    // Cleanup
    // free the kernel args
    destroy_dp_kernel_arg(&kernel_args[0]);
    destroy_dp_kernel_arg(&kernel_args[1]);
    destroy_dp_kernel_arg(&kernel_args[2]);
    // free the kernel
    destroy_dp_kernel(&kernel_ptr);
    // free the program
    destroy_dp_program(&program_ptr);
    // free the buffers
    destroy_dp_rw_mem_buffer(&buffers[0]);
    destroy_dp_rw_mem_buffer(&buffers[1]);
    destroy_dp_rw_mem_buffer(&buffers[2]);
    // free allocated memory for the arrays
    free(A);
    free(B);
    free(C);
}


int main (int argc, char** argv)
{
    runtime_t rt;
    int err;

    err = create_dp_runtime(&rt);
    if(err == DP_GLUE_FAILURE) goto error;
    rt->dump_fn(rt);

    printf("\n===================================\n\n");
    //--- Execute on CPU
    printf("Executing on the first CPU device info: \n");
    rt->first_cpu_env->dump_fn(rt->first_cpu_env);
    buildAndExecuteKernel(rt, ON_CPU);

    printf("\n===================================\n\n");

    printf("Executing on the first GPU device info: \n");
    rt->first_gpu_env->dump_fn(rt->first_gpu_env);
    buildAndExecuteKernel(rt, ON_GPU);

    printf("\n===================================\n\n");

    //--- Cleanup
    destroy_dp_runtime(&rt);

    return 0;

error:
    return DP_GLUE_FAILURE;
}
