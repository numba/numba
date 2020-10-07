#!/bin/bash

CC=clang
CXX=dpcpp

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build install --single-version-externally-managed --record=record.txt ----old-and-unmanageable


$CC -flto -target spir64-unknown-unknown -c -x cl -emit-llvm -cl-std=CL2.0 -Xclang -finclude-default-header numba/dppl/ocl/atomics/atomic_ops.cl -o numba/dppl/ocl/atomics/atomic_ops.bc
llvm-spirv -o numba/dppl/ocl/atomics/atomic_ops.spir numba/dppl/ocl/atomics/atomic_ops.bc
cp numba/dppl/ocl/atomics/atomic_ops.spir ${SP_DIR}/numba/dppl/ocl/atomics/
