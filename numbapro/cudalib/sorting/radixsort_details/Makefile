CUB_INCLUDE = -I"../../../../thirdparty/cub"

all: libradix64 libradix

radixsort.run: radixsort.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ $(CUB_INCLUDE)

radixselect.run: radixselect.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ $(CUB_INCLUDE)

testsort: radixsort.run
	./radixsort.run

testselect: radixselect.run
	./radixselect.run

libradix: libradix_sm30.ptx libradix_sm35.ptx libradix_sm20.ptx
libradix64: libradix64_sm30.ptx libradix64_sm35.ptx libradix64_sm20.ptx

libradix_sm30.ptx: libradix.cu
	nvcc -m32 -O3 -arch=sm_30 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix_sm35.ptx: libradix.cu
	nvcc -m32 -O3 -arch=sm_35 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix_sm20.ptx: libradix.cu
	nvcc -m32 -O3 -arch=sm_20 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix64_sm30.ptx: libradix.cu
	nvcc -m64 -O3 -arch=sm_30 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix64_sm35.ptx: libradix.cu
	nvcc -m64 -O3 -arch=sm_35 -ptx $+ -o $@ $(CUB_INCLUDE)

libradix64_sm20.ptx: libradix.cu
	nvcc -m64 -O3 -arch=sm_20 -ptx $+ -o $@ $(CUB_INCLUDE)

clean:
	rm -f radixsort.run radixselect.run
	rm -f *.ptx
