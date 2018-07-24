#!/bin/bash

set -v -e

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
# Clean up any left-over from a previous build
# (note workaround for https://github.com/conda/conda/issues/2679:
#  `conda env remove` issue)
conda remove --all -q -y -n $CONDA_ENV

# If VANILLA_INSTALL is yes, then only Python, NumPy and pip are installed, this
# is to catch tests/code paths that require an optional package and are not
# guarding against the possibility that it does not exist in the environment.
# Create a base env first and then add to it...

conda create -n $CONDA_ENV -q -y ${EXTRA_CHANNELS} python=$PYTHON numpy=$NUMPY pip

if [ "${VANILLA_INSTALL}" != "yes" ]; then
    # Scipy, CFFI, jinja2, IPython and pygments are optional dependencies, but exercised in the test suite
    $CONDA_INSTALL ${EXTRA_CHANNELS} cffi scipy jinja2 ipython pygments
fi

set +v
source activate $CONDA_ENV
set -v

# Install the compiler toolchain
if [[ $(uname) == Linux ]]; then
    if [[ "$CONDA_SUBDIR" == "linux-32" ]]; then
        $CONDA_INSTALL gcc_linux-32 gxx_linux-32
    else
        $CONDA_INSTALL gcc_linux-64 gxx_linux-64
    fi
elif  [[ $(uname) == Darwin ]]; then
    $CONDA_INSTALL clang_osx-64 clangxx_osx-64
fi

# Install latest llvmlite build
$CONDA_INSTALL -c numba llvmlite
# Install enum34 and singledispatch for Python < 3.4
if [ $PYTHON \< "3.4" ]; then $CONDA_INSTALL enum34; fi
if [ $PYTHON \< "3.4" ]; then $PIP_INSTALL singledispatch; fi
# Install funcsigs for Python < 3.3
if [ $PYTHON \< "3.3" ]; then $CONDA_INSTALL -c numba funcsigs; fi
# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then $CONDA_INSTALL sphinx pygments; fi
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL sphinx_bootstrap_theme; fi
# Install dependencies for code coverage (codecov.io)
if [ "$RUN_COVERAGE" == "yes" ]; then $PIP_INSTALL codecov; fi
# Install SVML
if [ "$TEST_SVML" == "yes" ]; then $CONDA_INSTALL -c numba icc_rt; fi

if [ $PYTHON \< "3.0" ]; then $CONDA_INSTALL faulthandler; fi
