#!/bin/sh

find . -name "*.so" -exec rm {} \;

cd numba/oneapi/oneapidriver
gcc -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -shared -fPIC -o libnumbaoneapiglue.so numba_oneapi_glue.c
cd ../../..

python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
