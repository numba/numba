#!/bin/bash

source  activate
conda activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

if [[ $(uname) == "Darwin" ]]; then
    # The following is suggested in https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html?highlight=SDK#macos-sdk
    wget -q https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.10.sdk.tar.xz
    shasum -c ./buildscripts/incremental/MacOSX10.10.sdk.checksum
    tar -xf ./MacOSX10.10.sdk.tar.xz
    export SDKROOT=`pwd`/MacOSX10.10.sdk
fi
python setup.py build_ext -q --inplace
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
