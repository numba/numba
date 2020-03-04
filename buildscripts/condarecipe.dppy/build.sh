#!/bin/bash

OPENCL_INCLUDE=""
if [ -z "$OPENCL_HOME" ]; then
    OPENCL_INCLUDE=-I/usr/include
else
    OPENCL_INCLUDE=-I${OPENCL_HOME}/include
fi

SRC_PATH="${SRC_DIR}/numba/dppy/dppy_driver"

$CC -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -fPIC ${OPENCL_INCLUDE} -c ${SRC_PATH}/opencllite.c -o ${SRC_PATH}/opencllite.o

ar rcs ${SRC_PATH}/libdpglue.a ${SRC_PATH}/opencllite.o

$CC -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -shared -fPIC ${OPENCL_INCLUDE} -o ${SRC_PATH}/libdpglue_so.so ${SRC_PATH}/opencllite.c

cp ${SRC_PATH}/libdpglue_so.so $PREFIX/lib/

$PYTHON buildscripts/remove_unwanted_files.py

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build_ext --inplace install --single-version-externally-managed --record=record.txt
