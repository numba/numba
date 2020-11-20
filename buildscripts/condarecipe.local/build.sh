#!/bin/bash

if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"86"* ]]; then
    EXTRA_BUILD_EXT_FLAGS="--werror --wall"
else
    EXTRA_BUILD_EXT_FLAGS=""
fi

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build_ext $EXTRA_BUILD_EXT_FLAGS build install --single-version-externally-managed --record=record.txt
