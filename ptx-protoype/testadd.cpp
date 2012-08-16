#include "cuda.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(void) {
    srand(time(NULL));
    const int threadsPerBlock = 32;
    const int blocksPerGrid = 10;
    const unsigned N = blocksPerGrid * threadsPerBlock;
    const unsigned size = sizeof(int)*N;

    int A[N];
    int B[N];
    int S[N];

    for (int i=0; i<N; ++i){
        A[i] = rand();
        B[i] = rand();
    }

    // Initialize
    cuInit(0);
    // Get number of devices supporting CUDA
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(0);
    }

    // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Create module from binary file
    CUmodule cuModule;
    cuModuleLoad(&cuModule, "add.ptx");

    // Allocate vectors in device memory
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, size);

    CUdeviceptr d_B;
    cuMemAlloc(&d_B, size);

    CUdeviceptr d_S;
    cuMemAlloc(&d_S, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, A, size);
    cuMemcpyHtoD(d_B, B, size);

    // Get function handle from module
    CUfunction function;
    cuModuleGetFunction(&function, cuModule, "ptx_add");

    // Invoke kernel
    void* args[] = { (void*)&d_S, (void*)&d_A, (void*)&d_B, (void*)&N };
    cuLaunchKernel(function, 
                   blocksPerGrid, 1, 1, 
                   threadsPerBlock, 1, 1,     
                   0, 0, args, 0);

    // Retrieve result
    cuMemcpyDtoH(S, d_S, size);

    cuMemFree(d_S);
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuCtxDestroy(cuContext);

    // Check
    unsigned count_error = 0;
    for (int i=0; i<N; ++i){
        // printf("%d ", S[i]);
        if (A[i] + B[i] != S[i]){
            printf("error at i=%d\n", S[i]);
            ++count_error;
        }
    }

    if(!count_error){
        printf("All is well\n");
    }
    return 0;
}
