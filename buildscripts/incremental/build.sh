#!/bin/bash

source  activate
conda activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e


git clone https://github.com/numba/llvmlite 
pushd llvmlite
git fetch origin pull/869/head:pr/869
git checkout pr/869
python -m pip install . 
popd

# Build numba extensions without silencing compile errors
# if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"86"* ]]; then
#     EXTRA_BUILD_EXT_FLAGS="--werror --wall"
# else
#     EXTRA_BUILD_EXT_FLAGS=""
# fi
EXTRA_BUILD_EXT_FLAGS=""

if [[ $(uname) == "Darwin" ]]; then
    # The following is suggested in https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html?highlight=SDK#macos-sdk
    wget -q https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.10.sdk.tar.xz
    shasum -c ./buildscripts/incremental/MacOSX10.10.sdk.checksum
    tar -xf ./MacOSX10.10.sdk.tar.xz
    export SDKROOT=`pwd`/MacOSX10.10.sdk
fi
python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS --verbose
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
