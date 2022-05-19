#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Ensure the README is correctly formatted
if [ "$BUILD_DOC" == "yes" ]; then rstcheck README.rst; fi
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

# enable new style error handling
export NUMBA_CAPTURED_ERRORS="new_style"

# Disable NumPy dispatching to AVX512_SKX feature extensions if the chip is
# reported to support the feature and NumPy >= 1.22 as this results in the use
# of low accuracy SVML libm replacements in ufunc loops.
_NPY_CMD='from numba.misc import numba_sysinfo;\
          sysinfo=numba_sysinfo.get_sysinfo();\
          print(sysinfo["NumPy AVX512_SKX detected"] and
                sysinfo["NumPy Version"]>="1.22")'
NUMPY_DETECTS_AVX512_SKX_NP_GT_122=$(python -c "$_NPY_CMD")
echo "NumPy >= 1.22 with AVX512_SKX detected: $NUMPY_DETECTS_AVX512_SKX_NP_GT_122"

if [[ "$NUMPY_DETECTS_AVX512_SKX_NP_GT_122" == "True" ]]; then
    export NPY_DISABLE_CPU_FEATURES="AVX512_SKX"
fi

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

# If TEST_THREADING is set in the env, then check that Numba agrees that the
# environment can support the requested threading.
function check_sysinfo() {
    cmd="import os;\
         from numba.misc.numba_sysinfo import get_sysinfo;\
         assert get_sysinfo()['$1 Threading'] is True, 'Threading layer $1 '\
         'is not supported';\
         print('Threading layer $1 is supported')"
    python -c "$cmd"
}

if [[ "$TEST_THREADING" ]]; then
    if [[ "$TEST_THREADING" == "tbb" ]]; then
        check_sysinfo "TBB"
    elif [[ "$TEST_THREADING" == "omp" ]]; then
        check_sysinfo "OpenMP"
    elif [[ "$TEST_THREADING" == "workqueue" ]]; then
        check_sysinfo "Workqueue"
    else
        echo "Unknown threading layer requested: $TEST_THREADING"
        exit 1
    fi
fi

# Find catchsegv
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

# setup SDKROOT on Mac
if [[ $(uname) == "Darwin" ]]; then
    export SDKROOT=`pwd`/MacOSX10.10.sdk
fi

# First check that the test discovery works
python -m numba.tests.test_runtests

# Now run tests based on the changes identified via git
NUMBA_ENABLE_CUDASIM=1 $SEGVCATCH python -m numba.runtests -b -v -g -m $TEST_NPROCS -- numba.tests

# List the tests found
echo "INFO: All discovered tests:"
python -m numba.runtests -l

# Now run the Numba test suite with slicing
# Note that coverage is run from the checkout dir to match the "source"
# directive in .coveragerc
echo "INFO: Running slice of discovered tests: ($TEST_START_INDEX,None,$TEST_COUNT)"
if [ "$RUN_COVERAGE" == "yes" ]; then
    export PYTHONPATH=.
    coverage erase
    $SEGVCATCH coverage run runtests.py -b -j "$TEST_START_INDEX,None,$TEST_COUNT" --exclude-tags='long_running' -m $TEST_NPROCS -- numba.tests
elif [ "$RUN_TYPEGUARD" == "yes" ]; then
    echo "INFO: Running with typeguard"
    NUMBA_USE_TYPEGUARD=1 NUMBA_ENABLE_CUDASIM=1 PYTHONWARNINGS="ignore:::typeguard" $SEGVCATCH python runtests.py -b -j "$TEST_START_INDEX,None,$TEST_COUNT" --exclude-tags='long_running' -m $TEST_NPROCS -- numba.tests
else
    NUMBA_ENABLE_CUDASIM=1 $SEGVCATCH python -m numba.runtests -b -j "$TEST_START_INDEX,None,$TEST_COUNT" --exclude-tags='long_running' -m $TEST_NPROCS -- numba.tests
fi
