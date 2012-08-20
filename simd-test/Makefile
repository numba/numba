
.PHONY: install test clean

all:
	python setup.py build_ext -i

install:
	python setup.py install

test:
	python test_simd.py

clean:
	rm -f simdtest.so

