all: atomic.bc

atomic.bc: atomic.ll
	llvm-as atomic.ll

clean:
	rm atomic.bc
