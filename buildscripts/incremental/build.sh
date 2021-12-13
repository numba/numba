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
    export MACOSX_SDK_VERSION=10.15
    export MACOSX_DEPLOYMENT_TARGET=10.15
    export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
    export OSX_SDK_DIR="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs"
    export USING_SYSTEM_SDK_DIR=1
    export CPATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
    plutil -replace MinimumSDKVersion -string ${MACOSX_SDK_VERSION} $(xcode-select -p)/Platforms/MacOSX.platform/Info.plist
    plutil -replace DTSDKName -string macosx${MACOSX_SDK_VERSION}internal $(xcode-select -p)/Platforms/MacOSX.platform/Info.plist
fi
python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS --verbose
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
