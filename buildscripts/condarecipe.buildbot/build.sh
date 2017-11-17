#!/bin/bash

$PYTHON buildscripts/remove_unwanted_files.py
# conda-build sets MACOSX_DEPLOYMENT_TARGET to 10.6, which is
# too old for some features.  Also, llvmlite requires 10.9 or higher.
MACOSX_DEPLOYMENT_TARGET=10.9 $PYTHON setup.py build install --single-version-externally-managed --record=record.txt
