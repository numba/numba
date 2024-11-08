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
    export SDKROOT=/Applications/Xcode_14.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
fi
python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS --verbose
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
