#!/bin/bash

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ] && [[ "$(uname -p)" == *"86"* ]]; then
    EXTRA_BUILD_EXT_FLAGS="--werror --wall"
else
    EXTRA_BUILD_EXT_FLAGS=""
fi

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build_ext --inplace $EXTRA_BUILD_EXT_FLAGS build install --single-version-externally-managed --record=record.txt
