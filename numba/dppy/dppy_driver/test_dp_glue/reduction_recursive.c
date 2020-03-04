#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int size = 1024;
int size_work_group = 256;
int rounds = 1;

int init_data(int size, double filler, double *data)
{
    int i;
    for(i = 0; i < size; ++i) {
        data[i] = filler;
    }
    return 0;
}

void
print_time(char *str, double elapsed)
{
    printf("Time elapsed for %s: %lf us\n", str, elapsed);
}

void
print_stat(char *str, int *times)
{
    double avg_time;
    int sum_time = 0;
    int round;

    for (round = 0; round < rounds; ++round) {
        sum_time += times[round];
    }
    avg_time = sum_time / rounds;

    print_time(str, avg_time);
}

void
help(void)
{
    printf("./reduction [-g / --global] [-l / --local] [-r / --rounds] [-h / --help]\n");
    printf("\t[-g / --global]:\tGlobal work size\n");
    printf("\t[-l / --local]: \tLocal work size\n");
    printf("\t[-r / --rounds]:\tNumber of times each test will be repeted\n");
    printf("\t[-h / --help]:  \tPrint this menu\n");
}

void
print_args(void)
{
    printf("\t$ Global Work Size:  %d\n"
           "\t$ Local Work Size:   %d\n"
           "\t$ Rounds:            %d\n",
             size, size_work_group, rounds);
}

int
get_args(int argc, char **argv)
{
    int i;

    /* iterate over all arguments */
    for (i = 1; i < argc; i++) {
        if ((strcmp("-g", argv[i]) == 0) || (strcmp("--global", argv[i]) == 0)) {
            size = atoi(argv[++i]);
            continue;
        }
        if ((strcmp("-l", argv[i]) == 0) || (strcmp("--local", argv[i]) == 0)) {
            size_work_group = atoi(argv[++i]);
            continue;
        }
        if ((strcmp("-r", argv[i]) == 0) || (strcmp("--rounds", argv[i]) == 0)) {
            rounds = atoi(argv[++i]);
            continue;
        }
        if ((strcmp("-h", argv[i]) == 0) || (strcmp("--help", argv[i]) == 0)) {
            help();
            exit(0);
        }
        help();
        exit(0);
    }
    return 0;
}

double
recursive_reduction(int size, size_t group_size,
        cl_mem input_buffer, cl_mem partial_sum_buffer,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, cl_uint num_events, cl_event *event)
{
    double result;
    size_t nb_work_groups = 0;
    cl_int ret;
    size_t passed_size = size;

    if (size <= group_size) {
        nb_work_groups = 1;
    } else {
        nb_work_groups = size / group_size;
        // If it is not divisible we have one extra group and pad the rest
        if(size % group_size != 0) {
            nb_work_groups++;
            passed_size = nb_work_groups * group_size;
        }
    }

    cl_event cur_event[1];
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
    ret = clSetKernelArg(kernel, 1, sizeof(int), (void *)&size);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&partial_sum_buffer);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&nb_work_groups);
    ret = clSetKernelArg(kernel, 4, group_size * sizeof(double), NULL);

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &passed_size, &group_size, num_events, event, cur_event);

    /* Make sure everything is done before this */
    //ret = clFlush(command_queue);
    //ret = clFinish(command_queue);

    if (nb_work_groups <= group_size) {
        int tmp_buffer_length = 1;
        cl_event last_event[1];
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&partial_sum_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(int), (void *)&nb_work_groups);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&tmp_buffer_length);
        ret = clSetKernelArg(kernel, 4, group_size * sizeof(double), NULL);

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &group_size, &group_size, 1, cur_event, last_event);

        ret = clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0,
                sizeof(double), &result, 1, last_event, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
    } else {
        result = recursive_reduction(nb_work_groups, group_size,
            partial_sum_buffer, input_buffer,
            context, command_queue,
            kernel, 1, cur_event);
    }

    return result;
}

double
reduction_wrapper(double *input, int size, int group_size, int *times)
{
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    double result;
    int i;

    fp = fopen("reduction.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices = 0;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(0, NULL, &(ret_num_platforms));
    cl_platform_id *platform_ids = (cl_platform_id *) malloc(sizeof(cl_platform_id)*ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform_ids, NULL);

    for (i = 0; i < ret_num_platforms; ++i) {
        ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_devices);
        if(!ret_num_devices) continue;
        ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if(ret_num_devices) break;

    }
    if (ret !=  CL_SUCCESS) {
        printf("Get Device ID failed with code %d\n", ret);
        abort();
    } else {
        //print_device_info(device_id);
    }

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Context creation failed with code %d\n", ret);
        abort();
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Program creation failed with code %d\n", ret);
        abort();
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret !=  CL_SUCCESS) {
        printf("Program building failed with code %d\n", ret);
        abort();
    }

    // Create the OpenCL kernel
    cl_kernel kernelGPU = clCreateKernel(program, "reduceGPU", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    int round;
    struct timeval start_time, end_time;

    for (round = 0; round < rounds; ++round) {
        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                size * sizeof(double), NULL, &ret);
        int nb_work_groups = size / group_size;
        // If it is not divisible we have one extra group and pad the rest
        if(size % group_size != 0) {
            nb_work_groups++;
        }

        double *partial_sums = (double* )malloc(nb_work_groups * sizeof(double));
        init_data(nb_work_groups, 0, partial_sums);

        cl_mem partial_sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                nb_work_groups * sizeof(double), NULL, &ret);

        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                size * sizeof(double), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, partial_sum_buffer, CL_TRUE, 0,
                nb_work_groups * sizeof(double), partial_sums, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        gettimeofday(&start_time, NULL);

        result = recursive_reduction(size, group_size,
                input_buffer, partial_sum_buffer,
                context, command_queue,
                kernelGPU, 0, NULL);

        gettimeofday(&end_time, NULL);
        times[round] = end_time.tv_usec - start_time.tv_usec;

        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(partial_sum_buffer);
        free(partial_sums);
    }

    // Clean up
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseKernel(kernelGPU);
    ret = clReleaseProgram(program);
    ret = clReleaseContext(context);

    free(source_str);
    return result;
}

int main(int argc, char **argv) {

    get_args(argc, argv);
    print_args();

    int *times = (int *) malloc(sizeof(int) * rounds);
    double *input = (double *) malloc(sizeof(double) * size);
    init_data(size, 1, input);

    int tmp_rounds = rounds;
    rounds = 1;
    //Warmup
    double result = reduction_wrapper(input, size, size_work_group, times);

    rounds = tmp_rounds;
    result = reduction_wrapper(input, size, size_work_group, times);
    //printf("result: %lf\n", result);
    assert(result == size);
    print_stat("Reduction in Device", times);

    free(input);
    free(times);

    return 0;
}
