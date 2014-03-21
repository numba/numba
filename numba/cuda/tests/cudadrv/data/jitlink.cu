// compile with:
//
//  nvcc -arch=sm_20 -dc jitlink.cu  -o jitlink.o
//
//

extern "C"{
    __device__
    int bar(int* out, int a) {
        *out = a * 2;
        return 0;
    }
}
