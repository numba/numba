CUB_INCLUDE = -I"../../../../thirdparty/cub"
MGPU_INCLUDE = -I../../../../thirdparty/moderngpu/include
FLAG32 = --no-align-double
CFLAGS = -O3

GENCODE_SM20	:= -gencode arch=compute_20,code=sm_20
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35

GENCODE_FLAGS	:= $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

mgpusort.so: mgpusort.cu
	nvcc --compiler-options "-fPIC" $(MGPU_INCLUDE) -O3 -arch=sm_30 --shared -o $@ $+

all: libradix64 libradix32

radixsort.run: radixsort.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ $(CUB_INCLUDE)

radixselect.run: radixselect.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ $(CUB_INCLUDE)

testsort: radixsort.run
	./radixsort.run

testselect: radixselect.run
	./radixselect.run

libradix32: libradix32_sm30.ptx libradix32_sm35.ptx libradix32_sm20.ptx
libradix64: libradix64_sm30.ptx libradix64_sm35.ptx libradix64_sm20.ptx

libradix32_sm30.ptx: libradix.cu
	nvcc  $(FLAG32) -m32 $(CFLAGS) -arch=sm_30 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix32_sm35.ptx: libradix.cu
	nvcc $(FLAG32) -m32 $(CFLAGS) -arch=sm_35 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix32_sm20.ptx: libradix.cu
	nvcc $(FLAG32) -m32 $(CFLAGS) -arch=sm_20 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix64_sm30.ptx: libradix.cu
	nvcc -m64 $(CFLAGS) -arch=sm_30 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix64_sm35.ptx: libradix.cu
	nvcc -m64 $(CFLAGS) -arch=sm_35 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix64_sm20.ptx: libradix.cu
	nvcc -m64 $(CFLAGS) -arch=sm_20 -ptx $+ -o $@ $(CUB_INCLUDE)

clean:
	rm -f radixsort.run radixselect.run
	rm -f *.ptx

radixsort64.so : cubradixsort.cu
	nvcc -m64 --compiler-options "-fPIC" $(CUB_INCLUDE) -O3 $(GENCODE_FLAGS) --shared -o $@ $+
