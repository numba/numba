#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Check if we are on MACOSX and Python 2.7 and reset -isysroot if so
UNAME=$(uname -a)
PYVERSION=$(python -V 2>&1 | grep -o "Python \d")
if [ "$UNAME"  = "Darwin" ] && [ "$PYVERSION" = "Python 2" ]; then
    export CFLAGS="${CFLAGS} -isysroot /"
fi

# Build numba extensions without silencing compile errors
python setup.py build_ext -q --inplace
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install -e .
