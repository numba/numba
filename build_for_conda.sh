#!/bin/sh
set -x

find . -name "*.so" -exec rm {} \;

cd numba/dppy/dppy_driver

gcc -g -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC -c opencllite.c -o opencllite.o
gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "opencllite failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi
ar rcs libdpglue.a opencllite.o

#gcc -L. -shared -o libdpglue.so -Wl,--whole-archive -lopencllite

gcc -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -shared -fPIC -o libdpglue_so.so opencllite.c
gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "opencllite failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi

cd ../../..

python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
