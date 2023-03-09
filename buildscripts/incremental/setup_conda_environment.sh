#!/bin/bash

set -v -e

# first configure conda to have more tolerance of network problems, these
# numbers are not scientifically chosen, just merely larger than defaults
conda config --write-default
conda config --set remote_connect_timeout_secs 30.15
conda config --set remote_max_retries 10
conda config --set remote_read_timeout_secs 120.2
conda config --set show_channel_urls true
conda info
conda config --show

CONDA_INSTALL="conda install -q -y"
PIP_INSTALL="pip install -q"


EXTRA_CHANNELS=""
if [ "${USE_C3I_TEST_CHANNEL}" == "yes" ]; then
    EXTRA_CHANNELS="${EXTRA_CHANNELS} -c c3i_test"
fi


# Deactivate any environment
source deactivate
# Display root environment (for debugging)
conda list

# Use conda-forge for NumPy 1.24 - at the time of writing it is not available
# on the defaults channel.

if [ "${NUMPY}" == "1.24" ]; then
  NUMPY_CHANNEL_PKG=conda-forge::numpy
else
  NUMPY_CHANNEL_PKG=numpy
fi

# If VANILLA_INSTALL is yes, then only Python, NumPy and pip are installed, this
# is to catch tests/code paths that require an optional package and are not
# guarding against the possibility that it does not exist in the environment.
# Create a base env first and then add to it...
# NOTE: gitpython is needed for CI testing to do the test slicing
# NOTE: pyyaml is used to ensure that the Azure CI config is valid

conda create -n $CONDA_ENV -q -y ${EXTRA_CHANNELS} python=$PYTHON $NUMPY_CHANNEL_PKG=$NUMPY pip gitpython pyyaml

# Activate first
set +v
source activate $CONDA_ENV
set -v

# Install optional packages into activated env
echo "PYTHON=$PYTHON"
echo "VANILLA_INSTALL=$VANILLA_INSTALL"
if [ "${VANILLA_INSTALL}" != "yes" ]; then
    # Scipy, CFFI, jinja2, IPython and pygments are optional
    # dependencies, but exercised in the test suite
    # pexpect is used to run the gdb tests.
    # ipykernel is used for testing ipython behaviours.
    $CONDA_INSTALL ${EXTRA_CHANNELS} cffi jinja2 ipython ipykernel scipy pygments pexpect
fi

# Install the compiler toolchain and gdb (if available)
if [[ $(uname) == Linux ]]; then
    $CONDA_INSTALL gcc_linux-64 gxx_linux-64 gdb gdb-pretty-printer
elif  [[ $(uname) == Darwin ]]; then
    $CONDA_INSTALL clang_osx-64 clangxx_osx-64
    # Install llvm-openmp on OSX for headers during build and runtime during
    # testing
    $CONDA_INSTALL llvm-openmp
fi

# Install latest correct build
$CONDA_INSTALL -c numba/label/dev llvmlite=0.40

# Install importlib-metadata for Python < 3.9
if [ $PYTHON \< "3.9" ]; then $CONDA_INSTALL importlib_metadata; fi

# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then $CONDA_INSTALL sphinx docutils sphinx_rtd_theme pygments numpydoc; fi
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL rstcheck; fi
# Install dependencies for code coverage (codecov.io)
if [ "$RUN_COVERAGE" == "yes" ]; then $PIP_INSTALL codecov; fi
# Install SVML
if [ "$TEST_SVML" == "yes" ]; then $CONDA_INSTALL -c numba icc_rt; fi
# Install Intel TBB parallel backend
if [ "$TEST_THREADING" == "tbb" ]; then $CONDA_INSTALL -c numba tbb=2021 "tbb-devel>=2021,<2021.6"; fi
# Install typeguard
if [ "$RUN_TYPEGUARD" == "yes" ]; then $CONDA_INSTALL conda-forge::typeguard; fi

# environment dump for debug
# echo "DEBUG ENV:"
# echo "-------------------------------------------------------------------------"
# conda env export
# echo "-------------------------------------------------------------------------"
