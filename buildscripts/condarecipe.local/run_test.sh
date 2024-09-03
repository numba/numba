#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1

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


unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  SEGVCATCH=catchsegv
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

# Check Numba executable is there
numba -h

# run system info tool
numba -s

# Check test discovery works
python -m numba.tests.test_runtests

# Run the whole test suite
echo "Running: $SEGVCATCH python -m numba.runtests -b -m $TEST_NPROCS -- $TESTS_TO_RUN"
$SEGVCATCH python -m numba.runtests -b -m $TEST_NPROCS -- $TESTS_TO_RUN
