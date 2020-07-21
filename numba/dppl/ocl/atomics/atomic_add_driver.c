#define CL_TARGET_OPENCL_VERSION 220

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>

#define MAX_SOURCE_SIZE (0x100000)

int long_atomics_present = 0;
int double_atomics_present = 0;

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

int
print_supported_extension(cl_device_id device) {
	cl_int err = 0;
	size_t size = 0;
    char *value;

    err = clRetainDevice(device);

	// print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &size);
	if (err != CL_SUCCESS ) {
		printf("Unable to obtain device info for param\n");
		return 1;
	}
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, value, NULL);
    //printf("Supported Extension: %s\n", value);

    if(strstr(value, "cl_khr_int64_base_atomics") != NULL) {
        //printf("Long atmoics found!\n");
        long_atomics_present = 1;
    }

    if(strstr(value, "cl_khr_fp64") != NULL) {
        //printf("double atmoics found!\n");
        double_atomics_present = 1;
    }


    free(value);

    err = clReleaseDevice(device);

    return 0;
}

int
atomic_op_float(size_t global_item_size, float val,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, int option)
{
    float *input = (float *) malloc(sizeof(float));
    cl_int ret;
    int rounds = 1000, round;

    for (round = 0; round < rounds; ++round) {
        if (option == 1) {
            input[0] = 0;
        } else if (option == 2) {
            input[0] = global_item_size * val;
        } else {
            printf("invalid option\n");
            abort();
        }
        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(float), NULL, &ret);
        cl_mem val_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                sizeof(float), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                sizeof(float), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, val_buffer, CL_TRUE, 0,
                sizeof(float), &val, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&val_buffer);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &global_item_size, NULL, 0, NULL, NULL);


        ret = clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0, sizeof(float), input, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        if (option == 1) {
            assert(input[0] == global_item_size * val);
        } else if (option == 2) {
            assert(input[0] == 0);
        }

        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(val_buffer);
    }

    return 0;
}

int
atomic_op_int(size_t global_item_size, int val,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, int option)
{
    int *input = (int *) malloc(sizeof(int));
    cl_int ret;
    int rounds = 1000, round;

    for (round = 0; round < rounds; ++round) {
        if (option == 1) {
            input[0] = 0;
        } else if (option == 2) {
            input[0] = global_item_size * val;
        } else {
            printf("invalid option\n");
            abort();
        }

        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(int), NULL, &ret);
        cl_mem val_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                sizeof(int), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                sizeof(int), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, val_buffer, CL_TRUE, 0,
                sizeof(int), &val, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&val_buffer);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &global_item_size, NULL, 0, NULL, NULL);


        ret = clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0, sizeof(int), input, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        if (option == 1) {
            assert(input[0] == global_item_size * val);
        } else if (option == 2) {
            assert(input[0] == 0);
        }

        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(val_buffer);
    }

    free(input);

    return 0;
}

int
atomic_op_double(size_t global_item_size, double val,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, int option)
{
    double *input = (double *) malloc(sizeof(double));
    cl_int ret;
    int rounds = 1000, round;

    for (round = 0; round < rounds; ++round) {
        if (option == 1) {
            input[0] = 0;
        } else if (option == 2) {
            input[0] = global_item_size * val;
        } else {
            printf("invalid option\n");
            abort();
        }
        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(double), NULL, &ret);
        cl_mem val_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                sizeof(double), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                sizeof(double), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, val_buffer, CL_TRUE, 0,
                sizeof(double), &val, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&val_buffer);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &global_item_size, NULL, 0, NULL, NULL);


        ret = clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0, sizeof(double), input, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        if (option == 1) {
            assert(input[0] == global_item_size * val);
        } else if (option == 2) {
            assert(input[0] == 0);
        }

        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(val_buffer);
    }

    return 0;
}

int
atomic_op_long(size_t global_item_size, long val,
        cl_context context, cl_command_queue command_queue,
        cl_kernel kernel, int option)
{
    long *input = (long *) malloc(sizeof(long));
    cl_int ret;
    long rounds = 1000, round;

    for (round = 0; round < rounds; ++round) {
        if (option == 1) {
            input[0] = 0;
        } else if (option == 2) {
            input[0] = global_item_size * val;
        } else {
            printf("invalid option\n");
            abort();
        }

        // Create memory buffers on the device for each vector
        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(long), NULL, &ret);
        cl_mem val_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                sizeof(long), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                sizeof(long), input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, val_buffer, CL_TRUE, 0,
                sizeof(long), &val, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&val_buffer);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                &global_item_size, NULL, 0, NULL, NULL);


        ret = clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0, sizeof(long), input, 0, NULL, NULL);

        /* Make sure everything is done before this */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);

        if (option == 1) {
            assert(input[0] == global_item_size * val);
        } else if (option == 2) {
            assert(input[0] == 0);
        }

        ret = clReleaseMemObject(input_buffer);
        ret = clReleaseMemObject(val_buffer);
    }

    free(input);

    return 0;
}


int main(int argc, char **argv) {
    int i;

    size_t global_item_size = 1000;

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
    print_supported_extension(device_id);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Context creation failed with code %d\n", ret);
        abort();
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    FILE *fp;
    size_t source_size;
    struct stat info;
    void *il;

    char *filename = "atomic_op_final.spir";
    stat(filename, &info);
    il = malloc(sizeof(char) * info.st_size);
    fp = fopen(filename, "rb");
    fread(il, 1, info.st_size, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithIL(context, il, info.st_size, &ret);
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
    cl_kernel kernel_atomic_add_float = clCreateKernel(program, "atomic_add_float", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    cl_kernel kernel_atomic_sub_float = clCreateKernel(program, "atomic_sub_float", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    // Create the OpenCL kernel
    cl_kernel kernel_atomic_add_int = clCreateKernel(program, "atomic_add_int", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    // Create the OpenCL kernel
    cl_kernel kernel_atomic_sub_int = clCreateKernel(program, "atomic_sub_int", &ret);
    if (ret !=  CL_SUCCESS) {
        printf("Kernel compilation failed with code %d\n", ret);
        abort();
    }

    ret = atomic_op_float(global_item_size, 2,
        context, command_queue,
        kernel_atomic_add_float, 1);

    ret = atomic_op_float(global_item_size, 2,
        context, command_queue,
        kernel_atomic_sub_float, 2);

    ret = atomic_op_int(global_item_size, 2,
        context, command_queue,
        kernel_atomic_add_int, 1);

    ret = atomic_op_int(global_item_size, 2,
        context, command_queue,
        kernel_atomic_sub_int, 2);

    if(double_atomics_present) {
        cl_kernel kernel_atomic_add_double = clCreateKernel(program, "atomic_add_double", &ret);
        if (ret !=  CL_SUCCESS) {
            printf("Kernel compilation failed with code %d\n", ret);
            abort();
        }

        cl_kernel kernel_atomic_sub_double = clCreateKernel(program, "atomic_sub_double", &ret);
        if (ret !=  CL_SUCCESS) {
            printf("Kernel compilation failed with code %d\n", ret);
            abort();
        }

        ret = atomic_op_double(global_item_size, 2,
            context, command_queue,
            kernel_atomic_add_double, 1);

        ret = atomic_op_double(global_item_size, 2,
            context, command_queue,
            kernel_atomic_sub_double, 2);

        ret = clReleaseKernel(kernel_atomic_add_double);
        ret = clReleaseKernel(kernel_atomic_sub_double);
    }


    if(long_atomics_present) {
        // Create the OpenCL kernel
        cl_kernel kernel_atomic_add_long = clCreateKernel(program, "atomic_add_long", &ret);
        if (ret !=  CL_SUCCESS) {
            printf("Kernel compilation failed with code %d\n", ret);
            abort();
        }

        cl_kernel kernel_atomic_sub_long = clCreateKernel(program, "atomic_sub_long", &ret);
        if (ret !=  CL_SUCCESS) {
            printf("Kernel compilation failed with code %d\n", ret);
            abort();
        }

        ret = atomic_op_long(global_item_size, 2,
            context, command_queue,
            kernel_atomic_add_long, 1);

        ret = atomic_op_long(global_item_size, 2,
            context, command_queue,
            kernel_atomic_sub_long, 2);

        ret = clReleaseKernel(kernel_atomic_add_long);
        ret = clReleaseKernel(kernel_atomic_sub_long);
    }

    // Clean up
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseKernel(kernel_atomic_add_float);
    ret = clReleaseKernel(kernel_atomic_sub_float);
    ret = clReleaseKernel(kernel_atomic_add_int);
    ret = clReleaseKernel(kernel_atomic_sub_int);
    ret = clReleaseProgram(program);
    ret = clReleaseContext(context);

    return 0;
}
