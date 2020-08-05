#!/bin/bash

if [ -z "$CC" ]; then
    CC=gcc
    CXX=g++
fi

function usage() {
    echo "usage: ./build_for_develop.sh [ [-h | --help] | [-d | --debug] ]"
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

set -x

python setup.py clean --all
python setup.py build --inplace
python setup.py develop

CC=clang
CXX=dpcpp

$CC -flto -target spir64-unknown-unknown -c -x cl -emit-llvm -cl-std=CL2.0 -Xclang -finclude-default-header numba/dppl/ocl/atomics/atomic_ops.cl -o numba/dppl/ocl/atomics/atomic_ops.bc
llvm-spirv -o numba/dppl/ocl/atomics/atomic_ops.spir numba/dppl/ocl/atomics/atomic_ops.bc
