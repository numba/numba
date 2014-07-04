all: radixsort.run radixselect.run

radixsort.run: radixsort.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ -I./cub

radixselect.run: radixselect.cu
	nvcc -O3 -arch=sm_30 -o $@ $+ -I./cub

testsort: radixsort.run
	./radixsort.run
	
testselect: radixselect.run
	./radixselect.run
	
clean:
	rm -f radixsort.run radixselect.run