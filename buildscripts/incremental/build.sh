#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Build numba extensions without silencing compile errors
if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"86"* ]]; then
    EXTRA_BUILD_EXT_FLAGS="--werror --wall"
else
    EXTRA_BUILD_EXT_FLAGS=""
fi

if [[ $(uname) == "Darwin" ]]; then
    export MACOSX_SDK_VERSION=10.15
    SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
    OSX_SDK_DIR="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs"
    USING_SYSTEM_SDK_DIR=1
    plutil -replace MinimumSDKVersion -string ${MACOSX_SDK_VERSION} $(xcode-select -p)/Platforms/MacOSX.platform/Info.plist
    plutil -replace DTSDKName -string macosx${MACOSX_SDK_VERSION}internal $(xcode-select -p)/Platforms/MacOSX.platform/Info.plist
fi
python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
