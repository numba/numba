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
    export MACOSX_DEPLOYMENT_TARGET='11.0'
    # Determine architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        CONDA_SUBDIR=osx-arm64
        # Bootstrap with ARM64 compiler
        conda create -y -p ${PWD}/bootstrap clangxx_osx-arm64
    else
        CONDA_SUBDIR=osx-64
        # Bootstrap with x86_64 compiler
        conda create -y -p ${PWD}/bootstrap clangxx_osx-64
    fi
    # Use explicit SDK path if set, otherwise detect
    if [ -z "$SDKROOT" ]; then
        SDKPATH=$(xcrun --show-sdk-path)
    else
        SDKPATH=$SDKROOT
    fi
    # Set minimum deployment target if not already set
    if [ -z "$MACOSX_DEPLOYMENT_TARGET" ]; then
        export MACOSX_DEPLOYMENT_TARGET=11.0
    fi
    # Set Darwin target based on architecture
    if [[ "$ARCH" == "arm64" ]]; then
        DARWIN_TARGET=arm64-apple-darwin20.0.0
    else
        DARWIN_TARGET=x86_64-apple-darwin13.4.0
    fi
fi
if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
    export MACOSX_DEPLOYMENT_TARGET  # Keep existing value
fi

python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS --verbose
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
