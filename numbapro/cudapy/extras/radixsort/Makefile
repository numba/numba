all: libradix64

radixsort.run: radixsort.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ -I./cub

radixselect.run: radixselect.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ -I./cub

testsort: radixsort.run
	./radixsort.run

testselect: radixselect.run
	./radixselect.run

libradix: libradix_sm30.ptx libradix_sm35.ptx libradix_sm20.ptx
libradix64: libradix64_sm30.ptx libradix64_sm35.ptx libradix64_sm20.ptx

libradix_sm30.ptx: libradix.cu
	nvcc -m32 -O3 -arch=sm_30 -ptx $+ -o $@ -I./cub

libradix_sm35.ptx: libradix.cu
	nvcc -m32 -O3 -arch=sm_35 -ptx $+ -o $@ -I./cub

libradix_sm20.ptx: libradix.cu
	nvcc -m32 -O3 -arch=sm_20 -ptx $+ -o $@ -I./cub

libradix64_sm30.ptx: libradix.cu
	nvcc -m64 -O3 -arch=sm_30 -ptx $+ -o $@ -I./cub

libradix64_sm35.ptx: libradix.cu
	nvcc -m64 -O3 -arch=sm_35 -ptx $+ -o $@ -I./cub

libradix64_sm20.ptx: libradix.cu
	nvcc -m64 -O3 -arch=sm_20 -ptx $+ -o $@ -I./cub


libradix_sm30.o: libradix.cu
	nvcc -O3 -arch=sm_30 -dc $+ -o $@ -I./cub

libradix_sm35.o: libradix.cu
	nvcc -O3 -arch=sm_35 -dc $+ -o $@ -I./cub

libradix_sm20.o: libradix.cu
	nvcc -O3 -arch=sm_20 -dc $+ -o $@ -I./cub


clean:
	rm -f radixsort.run radixselect.run
	rm -f *.ptx
