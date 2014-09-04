// compile with:
//
//  nvcc -gencode arch=compute_20,code=compute_20 -ptx jitlink.cu -o jitlink.ptx
//
//

extern "C"{
    __device__
    int bar(int* out, int a) {
        *out = a * 2;
        return 0;
    }
}
