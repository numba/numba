#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Ensure that the documentation builds without warnings
pushd docs
if [ "$BUILD_DOC" == "yes" ]; then make SPHINXOPTS=-W clean html; fi
popd

# We only build on PyPy at present, not test.
if [ "$PYTHON" == "pypy" ]; then
  # Since we can't run the system info tool, just state the Python version.
  python --version
  exit 0
fi

# Run system info tool
pushd bin
numba -s
popd


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
