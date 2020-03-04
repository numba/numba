#!/bin/bash

function usage() {
	echo "usage: ./build_for_conda.sh [ [-h | --help] | [-d | --debug] ]"
}

DEBUG_FLAGS=""

while [ "$1" != "" ]; do
    case $1 in
        "-d" | "--debug" )         shift
                                    DEBUG_FLAGS="-g -DDEBUG"
                                ;;
        "-h" | "--help" )           usage
                                exit
                                ;;
        * )                         usage
                                exit
    esac
    shift
done

OPENCL_INCLUDE=""
if [ -z "$OPENCL_HOME" ]; then
    OPENCL_INCLUDE=""
else
    OPENCL_INCLUDE=-I${OPENCL_HOME}/include
fi

set -x

cd numba/dppy/dppy_driver

$CC ${DEBUG_FLAGS} -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -fPIC ${OPENCL_INCLUDE} -c opencllite.c -o opencllite.o
#gcc -g -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC -c opencllite.c -o opencllite.o

gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "opencllite failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi

ar rcs libdpglue.a opencllite.o

#gcc -L. -shared -o libdpglue.so -Wl,--whole-archive -lopencllite
$CC ${DEBUG_FLAGS} -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -shared -fPIC ${OPENCL_INCLUDE} -o libdpglue_so.so opencllite.c
#gcc -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -shared -fPIC -o libdpglue_so.so opencllite.c

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
