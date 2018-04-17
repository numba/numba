#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Ensure that the documentation builds without warnings
pushd docs
if [ "$BUILD_DOC" == "yes" ]; then make SPHINXOPTS=-W clean html; fi
popd
# Run system info tool
pushd bin
numba -s
popd

# switch off color messages
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
# switch on developer mode
export NUMBA_DEVELOPER_MODE=1

# First check that the test discovery works
python -m numba.tests.test_runtests
# Now run the Numba test suite
# Note that coverage is run from the checkout dir to match the "source"
# directive in .coveragerc
if [ "$RUN_COVERAGE" == "yes" ]; then
    export PYTHONPATH=.
    coverage erase
    coverage run runtests.py -b -m numba.tests
else
    NUMBA_ENABLE_CUDASIM=1 python -m numba.runtests -b -m numba.tests
fi
