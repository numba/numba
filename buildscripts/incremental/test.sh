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
# enable the fault handler
export PYTHONFAULTHANDLER=1

# deal with threading layers
if [ -z ${TEST_THREADING+x} ]; then
    echo "INFO: Threading layer not explicitly set."
else
    case "${TEST_THREADING}" in "workqueue"|"omp"|"tbb")
        export NUMBA_THREADING_LAYER="$TEST_THREADING"
        echo "INFO: Threading layer set as: $TEST_THREADING"
        ;;
        *)
        echo "INFO: Threading layer explicitly set to bad value: $TEST_THREADING."
        exit 1
        ;;
    esac
fi


unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    if [[ "${BITS32}" == "yes" ]]; then
        SEGVCATCH=""
    else
        SEGVCATCH=catchsegv
    fi
elif [[ "$unamestr" == 'Darwin' ]]; then
  SEGVCATCH=""
else
  echo Error
fi

# limit CPUs in use on PPC64LE, fork() issues
# occur on high core count systems
archstr=`uname -m`
if [[ "$archstr" == 'ppc64le' ]]; then
    TEST_NPROCS=16
fi

# First check that the test discovery works
python -m numba.tests.test_runtests
# Now run the Numba test suite
# Note that coverage is run from the checkout dir to match the "source"
# directive in .coveragerc
if [ "$RUN_COVERAGE" == "yes" ]; then
    export PYTHONPATH=.
    coverage erase
    $SEGVCATCH coverage run runtests.py -b -m $TEST_NPROCS -- numba.tests
else
    NUMBA_ENABLE_CUDASIM=1 $SEGVCATCH python -m numba.runtests -b -m $TEST_NPROCS -- numba.tests
fi
