// Compile with:
//
//   nvcc -gencode arch=compute_53,code=compute_53 -rdc true -ptx jitlink.cu
//
// using the oldest supported toolkit version (10.2 at the time of writing).

extern "C"{
    __device__
    int bar(int* out, int a) {
        *out = a * 2;
        return 0;
    }
}
