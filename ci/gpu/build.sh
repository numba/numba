#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
##########################################
# Numba GPU build and test script for CI #
##########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export CUDA_REL=${CUDA_VERSION%.*}

# Set python version
export PY_VER=`python --version | awk '{print $2}'`

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
conda config --add channels numba
conda create -n numbatest python=$PY_VER "llvmlite" "numpy" "scipy" "jinja2" "cffi" "cudatoolkit=$CUDA_REL"
source activate numbatest

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build C extensions and link into environment
################################################################################

logger "Build Numba..."
python setup.py build_ext --inplace
python setup.py develop

# Dump system information from Numba
python -m numba -s

################################################################################
# TEST - Run CUDA tests
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Check GPU usage..."
    nvidia-smi

    logger "Run tests in numba.cuda.tests..."
    python -m numba.runtests numba.cuda.tests -m
fi
