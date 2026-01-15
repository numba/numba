#!/bin/bash

set -v -e

# first configure conda to have more tolerance of network problems, these
# numbers are not scientifically chosen, just merely larger than defaults
# Note: --write-default may fail if .condarc already exists (e.g., from setup-miniconda action)
conda config --write-default 2>/dev/null || true
conda config --set remote_connect_timeout_secs 30.15
conda config --set remote_max_retries 10
conda config --set remote_read_timeout_secs 120.2
conda config --set show_channel_urls true
conda info
conda config --show

CONDA_INSTALL="conda install -q -y"
PIP_INSTALL="pip install -q"


EXTRA_CHANNELS="$EXTRA_CHANNELS"
if [ "${USE_C3I_TEST_CHANNEL}" == "yes" ]; then
    EXTRA_CHANNELS="${EXTRA_CHANNELS} -c c3i_test"
fi

# Deactivate any environment
source deactivate
# Display root environment (for debugging)
conda list

# If VANILLA_INSTALL is yes, then only Python, NumPy and pip are installed, this
# is to catch tests/code paths that require an optional package and are not
# guarding against the possibility that it does not exist in the environment.
# Create a base env first and then add to it...
# NOTE: gitpython is needed for CI testing to do the test slicing
# NOTE: pyyaml is used to ensure that the Azure CI config is valid

conda create -n $CONDA_ENV -q -y ${EXTRA_CHANNELS} python=$PYTHON numpy=$NUMPY pip gitpython pyyaml psutil

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
    if [ $PYTHON \< "3.12" ]; then
        $CONDA_INSTALL ${EXTRA_CHANNELS} cffi jinja2 ipython ipykernel pygments pexpect
    elif [ $PYTHON \< "3.13" ]; then
        # At the time of writing `ipykernel` was not available for Python 3.12
        $CONDA_INSTALL ${EXTRA_CHANNELS} cffi jinja2 ipython pygments pexpect
    else
        echo "no extra packages for 3.13"

    fi

    if [ $NUMPY \< "2.0" ]; then
        $CONDA_INSTALL ${EXTRA_CHANNELS} scipy
    fi
fi

# Python 3.14+ requires setuptools
if [ ! $PYTHON \< "3.14" ]; then
    $CONDA_INSTALL ${EXTRA_CHANNELS} setuptools
fi

# Install the compiler toolchain and gdb (if available)
if [[ $(uname) == Linux ]]; then
    if [ $PYTHON \< "3.12" ]; then
        $CONDA_INSTALL gcc_linux-64=11 gxx_linux-64=11 gdb gdb-pretty-printer
    else
        # At the time of writing gdb and gdb-pretty-printer were not available
        # for 3.12.
        $CONDA_INSTALL gcc_linux-64=11 gxx_linux-64=11
    fi
elif  [[ $(uname) == Darwin ]]; then
    # Detect architecture and install appropriate compiler toolchain
    if [[ $(uname -m) == "arm64" ]]; then
        $CONDA_INSTALL clang_osx-arm64 clangxx_osx-arm64
    else
        $CONDA_INSTALL clang_osx-64 clangxx_osx-64
    fi
    # Install llvm-openmp on OSX for headers during build and runtime during
    # testing
    $CONDA_INSTALL llvm-openmp
fi

# Install latest correct build
$CONDA_INSTALL -c numba/label/dev llvmlite=0.47

# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then $CONDA_INSTALL sphinx docutils sphinx_rtd_theme pygments numpydoc; fi
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL rstcheck; fi
# Install dependencies for code coverage
if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coverage; fi
# Install SVML
if [ "$TEST_SVML" == "yes" ]; then $CONDA_INSTALL -c numba icc_rt; fi
# Install Intel TBB parallel backend
if [ "$TEST_THREADING" == "tbb" ]; then $CONDA_INSTALL "tbb>=2021.6" "tbb-devel>=2021.6"; fi
# Install typeguard
if [ "$RUN_TYPEGUARD" == "yes" ]; then $CONDA_INSTALL typeguard; fi

# environment dump for debug
echo "DEBUG ENV:"
echo "-------------------------------------------------------------------------"
conda env export
echo "-------------------------------------------------------------------------"
