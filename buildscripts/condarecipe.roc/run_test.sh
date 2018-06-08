#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1

# Check Numba executables are there
pycc -h
numba -h

# run system info tool
numba -s

# Check test discovery works
python -m numba.tests.test_runtests

# Run the ROC test suite
python -m numba.runtests -v -m -b numba.hsa.tests
