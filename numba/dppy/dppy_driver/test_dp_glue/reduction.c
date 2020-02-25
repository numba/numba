#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

#define MAX_SOURCE_SIZE (0x100000)


int print_device_info (cl_device_id device)
{
    cl_int err;
    size_t size;
    char *value;
    cl_uint maxComputeUnits;

    err = clRetainDevice(device);

    // print device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &size);
    value = (char*)malloc(size);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, size, value, NULL);
    printf("Device: %s\n", value);
    free(value);

    // print hardware device version
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &size);
    value = (char*) malloc(size);
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, size, value, NULL);
    printf("Hardware version: %s\n", value);
    free(value);

    // print software driver version
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &size);
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, size, value, NULL);
    printf("Software version: %s\n", value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &size);
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, size, value, NULL);
    printf("OpenCL C version: %s\n", value);
    free(value);

    // print parallel compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Parallel compute units: %d\n", maxComputeUnits);

    err = clReleaseDevice(device);

    return 0;
}

int size = 10;
int size_work_group = 2;
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

double
reduction_in_device(size_t global_item_size, size_t nb_work_groups, size_t local_item_size,
        double *input, double *sum_reduction, double *final_sum,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, int *times)
{
    cl_int ret;
    struct timeval start_time, end_time;
    int round;

    for (round = 0; round < rounds; ++round) {
        init_data(global_item_size, 1, input);
        init_data(nb_work_groups, 0, sum_reduction);
        init_data(1, 0, final_sum);

        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                global_item_size * sizeof(double), NULL, &ret);
        cl_mem sum_reduction_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                nb_work_groups * sizeof(double), NULL, &ret);
        cl_mem final_sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(double), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                global_item_size * sizeof(double), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, sum_reduction_buffer, CL_TRUE, 0,
                nb_work_groups * sizeof(double), sum_reduction, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, final_sum_buffer, CL_TRUE, 0,
                sizeof(double), final_sum, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&sum_reduction_buffer);
        ret = clSetKernelArg(kernel, 2, local_item_size * sizeof(double), NULL);
        ret = clSetKernelArg(kernel, 3, sizeof(double), (void *)&final_sum_buffer);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        gettimeofday(&start_time, NULL);
        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &global_item_size, &local_item_size, 0, NULL, NULL);


        ret = clEnqueueReadBuffer(command_queue, final_sum_buffer, CL_TRUE, 0, sizeof(double), final_sum, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        gettimeofday(&end_time, NULL);
        //print_time("Reduction in Device only", &start_time, &end_time);
        times[round] = end_time.tv_usec - start_time.tv_usec;


        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(sum_reduction_buffer);
        ret = clReleaseMemObject(final_sum_buffer);
    }

    return *final_sum;
}

double
reduction_in_host_and_device(size_t global_item_size, size_t nb_work_groups, size_t local_item_size,
        double *input, double *sum_reduction,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, int *times)
{
    cl_int ret;
    int i;
    struct timeval start_time, end_time;
    double final_sum_host;

    int round;

    for (round = 0; round < rounds; ++round) {
        final_sum_host = 0;
        init_data(global_item_size, 1, input);
        init_data(nb_work_groups, 0, sum_reduction);

        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                global_item_size * sizeof(double), NULL, &ret);
        cl_mem sum_reduction_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                nb_work_groups * sizeof(double), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                global_item_size * sizeof(double), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, sum_reduction_buffer, CL_TRUE, 0,
                nb_work_groups * sizeof(double), sum_reduction, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&sum_reduction_buffer);
        ret = clSetKernelArg(kernel, 2, local_item_size * sizeof(double), NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        gettimeofday(&start_time, NULL);

        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &global_item_size, &local_item_size, 0, NULL, NULL);

        ret = clEnqueueReadBuffer(command_queue, sum_reduction_buffer, CL_TRUE, 0, nb_work_groups * sizeof(double), sum_reduction, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        // Display the result to the screen
        for(i = 0; i < nb_work_groups; i++)
            final_sum_host += sum_reduction[i];

        gettimeofday(&end_time, NULL);
        //print_time("Reduction in Host and Device", &start_time, &end_time);
        times[round] = end_time.tv_usec - start_time.tv_usec;

        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(sum_reduction_buffer);
    }

    return final_sum_host;
}

double
reduction_in_host(int size, double *input, int *times) {
    int i;
    double sum;
    struct timeval start_time, end_time;

    int round;

    for (round = 0; round < rounds; ++round) {
        sum = 0;
        init_data(size, 1, input);

        gettimeofday(&start_time, NULL);
        for (i = 0; i < size; ++i) {
            sum += input[i];
        }
        gettimeofday(&end_time, NULL);
        //print_time("Reduction in Host only", &start_time, &end_time);
        times[round] = end_time.tv_usec - start_time.tv_usec;
    }

    return sum;
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

int main(int argc, char **argv) {
    // Create the two input vectors
    int i;

    get_args(argc, argv);
    print_args();

    int *times = (int *) malloc(sizeof(int) * rounds);
    size_t global_item_size = size;
    size_t local_item_size  = size_work_group;

    int nb_work_groups = global_item_size / local_item_size;

    double *input = (double *) malloc(sizeof(double) * global_item_size);
    double *sum_reduction = (double *)malloc(sizeof(double) * nb_work_groups);
    double *final_sum = (double *)malloc(sizeof(double));

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("sum_reduction_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
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
    cl_kernel kernelGPU = clCreateKernel(program, "sumGPU", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    cl_kernel kernelGPUCPU = clCreateKernel(program, "sumGPUCPU", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    double result;

    /* Warmup */
    result = reduction_in_device(global_item_size, nb_work_groups, local_item_size,
        input, sum_reduction, final_sum,
        context, command_queue,
        kernelGPU, times);

    result = reduction_in_device(global_item_size, nb_work_groups, local_item_size,
        input, sum_reduction, final_sum,
        context, command_queue,
        kernelGPU, times);
    //printf("result Host        \t%lf\n", result);
    assert(result == global_item_size);
    print_stat("Reduction in Device only    ", times);

    result = reduction_in_host_and_device(global_item_size, nb_work_groups, local_item_size,
        input, sum_reduction,
        context, command_queue,
        kernelGPUCPU, times);
    //printf("result Host+Device \t%lf\n", result);
    assert(result == global_item_size);
    print_stat("Reduction in Host and Device", times);

    result = reduction_in_host(global_item_size, input, times);
    //printf("result Host+Device \t%lf\n", result);
    assert(result == global_item_size);
    print_stat("Reduction in Host only      ", times);


    // Clean up
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseKernel(kernelGPU);
    ret = clReleaseKernel(kernelGPUCPU);
    ret = clReleaseProgram(program);
    ret = clReleaseContext(context);
    free(input);
    free(sum_reduction);
    free(final_sum);
    free(times);

    return 0;
}
