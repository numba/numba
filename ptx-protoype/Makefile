NVCC=/usr/local/cuda/bin/nvcc

all: testadd

testadd: testadd.cpp
	$(NVCC) -o $@ $+ -lcuda

clean:
	rm -f testadd
