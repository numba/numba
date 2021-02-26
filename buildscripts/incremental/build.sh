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
python setup.py build_ext -q --inplace --debug $EXTRA_BUILD_EXT_FLAGS
# (note we don't install to avoid problems with extra long Windows paths
#  during distutils-dependent tests -- e.g. test_pycc)

# Don't build with tbb support unless tbb testing was requested. Loading the
# threading layers with an unsupported tbb raises a warning that conflicts with
# tests counting warnings.
if [ "$TEST_THREADING" != "tbb" ]; then DISABLE_TBB=1; fi

# Install numba locally for use in `numba -s` sys info tool at test time
NUMBA_DISABLE_TBB=$DISABLE_TBB python -m pip install --no-deps -e .
