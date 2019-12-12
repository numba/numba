#!/bin/sh

find . -name "*.so" -exec rm {} \;

cd numba/oneapi/oneapidriver

gcc -g -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC -c numba_oneapi_glue.c -o numba_oneapi_glue.o
gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "numba_oneapi_glue failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi
ar rcs libnumbaoneapiglue.a numba_oneapi_glue.o

#gcc -L. -shared -o libnumbaoneapiglue.so -Wl,--whole-archive -lnumbaoneapiglue

gcc -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -shared -fPIC -o libnumbaoneapiglue_so.so numba_oneapi_glue.c
gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "numba_oneapi_glue_so failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi

cd ../../..

python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
