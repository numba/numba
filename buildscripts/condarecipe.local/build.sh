#!/bin/bash

if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"86"* ]]; then
    EXTRA_BUILD_EXT_FLAGS="--werror --wall"
else
    EXTRA_BUILD_EXT_FLAGS=""
fi

if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"ppc64le"* ]]; then
    # To workaround https://github.com/numba/numba/issues/7302 
    # because of a python build problem that the -pthread could be stripped.
    export CC="$CC -pthread"
    export CXX="$CXX -pthread"
fi

export NUMBA_PACKAGE_FORMAT="conda"

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build_ext $EXTRA_BUILD_EXT_FLAGS build install --single-version-externally-managed --record=record.txt
