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

echo "Compile gufunc scheduler"

clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -isystem /Users/runner/miniconda3/envs/azure_ci/include -arch x86_64 -I/Users/runner/miniconda3/envs/azure_ci/include -fPIC -O2 -isystem /Users/runner/miniconda3/envs/azure_ci/include -arch x86_64 -g -I/Users/runner/miniconda3/envs/azure_ci/include -I/Users/runner/miniconda3/envs/azure_ci/include/python3.9 -c numba/np/ufunc/gufunc_scheduler.cpp -o gufunc_scheduler.o -std=c++11

echo "Compile gufunc scheduler finished"

exit 1

python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install --no-deps -e .
