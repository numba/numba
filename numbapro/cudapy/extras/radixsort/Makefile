all: libradix_sm30.o libradix_sm35.o libradix_sm20.o

radixsort.run: radixsort.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ -I./cub

radixselect.run: radixselect.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ -I./cub

testsort: radixsort.run
	./radixsort.run

testselect: radixselect.run
	./radixselect.run

libradix: libradix_sm30.ptx libradix_sm35.ptx libradix_sm20.ptx

libradix_sm30.ptx: libradix.cu
	nvcc -O3 -arch=sm_30 -ptx $+ -o $@ -I./cub

libradix_sm35.ptx: libradix.cu
	nvcc -O3 -arch=sm_35 -ptx $+ -o $@ -I./cub

libradix_sm20.ptx: libradix.cu
	nvcc -O3 -arch=sm_20 -ptx $+ -o $@ -I./cub


libradix_sm30.o: libradix.cu
	nvcc -O3 -arch=sm_30 -dc $+ -o $@ -I./cub

libradix_sm35.o: libradix.cu
	nvcc -O3 -arch=sm_35 -dc $+ -o $@ -I./cub

libradix_sm20.o: libradix.cu
	nvcc -O3 -arch=sm_20 -dc $+ -o $@ -I./cub


clean:
	rm -f radixsort.run radixselect.run
