#!/bin/bash

source  activate
conda activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Build numba extensions without silencing compile errors
if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"86"* ]]; then
    EXTRA_BUILD_EXT_FLAGS="--werror --wall"
else
    EXTRA_BUILD_EXT_FLAGS=""
fi

if [[ $(uname) == "Darwin" ]]; then
    # Set up compiler environment for macOS
    export CC="/Users/runner/miniconda3/envs/azure_ci/bin/clang"
    export CXX="/Users/runner/miniconda3/envs/azure_ci/bin/clang++"
    export CONDA_BUILD_SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    export MACOSX_DEPLOYMENT_TARGET=10.15
    export CPATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"
    export LIBRARY_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
    export LDFLAGS="-Wl,-rpath,/Users/runner/miniconda3/envs/azure_ci/lib -L$LIBRARY_PATH"
    export CFLAGS="-isysroot $CONDA_BUILD_SYSROOT -isystem $CPATH"
    export CXXFLAGS="-isysroot $CONDA_BUILD_SYSROOT -isystem $CPATH"

    # Debug output
    echo "=== Compiler Settings ==="
    echo "CC: $CC"
    echo "CXX: $CXX"
    echo "SDKROOT: $CONDA_BUILD_SYSROOT"
    echo "CPATH: $CPATH"
    echo "LIBRARY_PATH: $LIBRARY_PATH"
    echo "Clang version: $($CC --version)"
    echo "=== End Compiler Settings ==="

    # Determine architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        CONDA_SUBDIR=osx-arm64
        DARWIN_TARGET=arm64-apple-darwin20.0.0
    else
        CONDA_SUBDIR=osx-64
        DARWIN_TARGET=x86_64-apple-darwin13.4.0
    fi
fi

python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS --verbose
# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
