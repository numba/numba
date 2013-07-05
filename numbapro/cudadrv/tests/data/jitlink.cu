// compile with:
//
//  nvcc -arch=sm_20 -dc jitlink.cu  -o jitlink.o
//
//

extern "C"{
    __device__
    void bar(int a, int* out) {
        *out = a * 2;
    }
}