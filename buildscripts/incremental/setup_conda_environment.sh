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

CONDA_CREATE="conda create -n ${CONDA_ENV} -q -y"
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

# If VANILLA_INSTALL is yes, then only Python, NumPy and pip are installed, this
# is to catch tests/code paths that require an optional package and are not
# guarding against the possibility that it does not exist in the environment.
# Create a base env first and then add to it...
# NOTE: gitpython is needed for CI testing to do the test slicing
# NOTE: pyyaml is used to ensure that the Azure CI config is valid
CONDA_PACKAGES=(python=$PYTHON numpy=$NUMPY pip gitpython pyyaml "\"setuptools<60\"")

# Install optional packages into activated env
echo "PYTHON=$PYTHON"
echo "VANILLA_INSTALL=$VANILLA_INSTALL"
if [ "${VANILLA_INSTALL}" != "yes" ]; then
    # Scipy, CFFI, jinja2, IPython and pygments are optional
    # dependencies, but exercised in the test suite
    # pexpect is used to run the gdb tests.
    # ipykernel is used for testing ipython behaviours.
    CONDA_PACKAGES+=(cffi jinja2 ipython ipykernel pygments pexpect)

    # NumPy 1.23 needs a conda-forge SciPy package, this is installed later.
    if [[ "$NUMPY" != "1.23" ]] ; then
        CONDA_PACKAGES+=(scipy)
    fi
fi

# Install the compiler toolchain and gdb (if available)
if [[ $(uname) == Linux ]]; then
    CONDA_PACKAGES+=(gcc_linux-64 gxx_linux-64 gdb gdb-pretty-printer)
elif  [[ $(uname) == Darwin ]]; then
    CONDA_PACKAGES+=(clang_osx-64 clangxx_osx-64)
    # Install llvm-openmp on OSX for headers during build and runtime during
    # testing
    CONDA_PACKAGES+=(llvm-openmp)
fi

# Install latest correct build
EXTRA_CHANNELS="-c numba/label/dev ${EXTRA_CHANNELS}"
CONDA_PACKAGES+=(llvmlite=0.40)

# Install importlib-metadata for Python < 3.9
if [ $PYTHON \< "3.9" ]; then CONDA_PACKAGES+=(importlib_metadata); fi

# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then CONDA_PACKAGES+=(sphinx=2.4.4 docutils=0.17 sphinx_rtd_theme pygments numpydoc); fi
# Install SVML
if [ "$TEST_SVML" == "yes" ]; then CONDA_PACKAGES+=(icc_rt); fi
# Install Intel TBB parallel backend
if [ "$TEST_THREADING" == "tbb" ]; then CONDA_PACKAGES+=(tbb=2021 tbb-devel); fi

CONDA_PACKAGES_AS_LIST=$(printf "%s " "${CONDA_PACKAGES[@]}")
echo "Running: $CONDA_CREATE $EXTRA_CHANNELS $CONDA_PACKAGES_AS_LIST"
$CONDA_CREATE $EXTRA_CHANNELS $CONDA_PACKAGES_AS_LIST

# Anything pip related goes last, conda env solution must be installed first.
# Activate env such that installation of extra packages is into the env.
set +v
source activate $CONDA_ENV
set -v

# Install conda-forge packages after the main solve so as to have a more
# controlled environment.
EXTRA_CONDA_PACKAGES=()
if [ "${VANILLA_INSTALL}" != "yes" ]; then
    if [[ "$NUMPY" == "1.23" ]] ; then
        EXTRA_CONDA_PACKAGES+=(conda-forge::scipy)
    fi
    # Install typeguard
    if [ "$RUN_TYPEGUARD" == "yes" ]; then
        EXTRA_CONDA_PACKAGES+=(conda-forge::typeguard)
    fi
    EXTRA_CONDA_PACKAGES_AS_LIST=$(printf "%s " "${EXTRA_CONDA_PACKAGES[@]}")
    echo "Running: $CONDA_INSTALL $EXTRA_CHANNELS $EXTRA_CONDA_PACKAGES_AS_LIST"
    $CONDA_INSTALL $EXTRA_CHANNELS $EXTRA_CONDA_PACKAGES_AS_LIST
fi

# Install rstcheck to check RST file changes
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL rstcheck; fi
# Install dependencies for code coverage (codecov.io)
if [ "$RUN_COVERAGE" == "yes" ]; then $PIP_INSTALL codecov; fi
# Install pickle5
if [ "$TEST_PICKLE5" == "yes" ]; then $PIP_INSTALL pickle5; fi
