#!/bin/bash

set -v -e

# first configure conda to have more tolerance of network problems, these
# numbers are not scientifically chosen, just merely larger than defaults
conda config --write-default
conda config --set remote_connect_timeout_secs 30.15
conda config --set remote_max_retries 10
conda config --set remote_read_timeout_secs 120.2
conda config --set show_channel_urls true
if [[ $(uname) == Linux ]]; then
    if [[ "$CONDA_SUBDIR" != "linux-32" && "$BITS32" != "yes" ]] ; then
        conda config --set restore_free_channel true
    fi
fi
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

# If VANILLA_INSTALL is yes, then only Python, NumPy and pip are installed, this
# is to catch tests/code paths that require an optional package and are not
# guarding against the possibility that it does not exist in the environment.
# Create a base env first and then add to it...
# NOTE: gitpython is needed for CI testing to do the test slicing
# NOTE: pyyaml is used to ensure that the Azure CI config is valid
# NOTE: 32 bit linux... do not install NumPy, there's no conda package for >1.15
# so it has to come from pip later
if [[ "$CONDA_SUBDIR" == "linux-32" || "$BITS32" == "yes" ]]; then
    conda create -n $CONDA_ENV -q -y ${EXTRA_CHANNELS} python=$PYTHON pip gitpython pyyaml
else
    conda create -n $CONDA_ENV -q -y ${EXTRA_CHANNELS} python=$PYTHON numpy=$NUMPY pip gitpython pyyaml
fi

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
    $CONDA_INSTALL ${EXTRA_CHANNELS} cffi jinja2 ipython ipykernel pygments pexpect
    # Only install scipy on 64bit, else it'll pull in NumPy, 32bit linux needs
    # to get scipy from pip
    if [[ "$CONDA_SUBDIR" != "linux-32" && "$BITS32" != "yes" ]] ; then
        $CONDA_INSTALL ${EXTRA_CHANNELS} scipy
    fi
fi

# Install the compiler toolchain
if [[ $(uname) == Linux ]]; then
    if [[ "$CONDA_SUBDIR" == "linux-32" || "$BITS32" == "yes" ]] ; then
        $CONDA_INSTALL gcc_linux-32 gxx_linux-32
    else
        $CONDA_INSTALL gcc_linux-64 gxx_linux-64
    fi
elif  [[ $(uname) == Darwin ]]; then
    $CONDA_INSTALL clang_osx-64 clangxx_osx-64
    # Install llvm-openmp on OSX for headers during build and runtime during
    # testing
    $CONDA_INSTALL llvm-openmp
fi

# If on 32bit linux, now pip install NumPy (no conda package), SciPy is broken?!
if [[ "$CONDA_SUBDIR" == "linux-32" || "$BITS32" == "yes" ]] ; then
    $PIP_INSTALL numpy==$NUMPY
fi

# Install latest correct build
$CONDA_INSTALL -c numba/label/dev llvmlite=0.40

# Install importlib-metadata for Python < 3.9
if [ $PYTHON \< "3.9" ]; then $CONDA_INSTALL importlib_metadata; fi

# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then $CONDA_INSTALL sphinx=2.4.4 docutils=0.17 sphinx_rtd_theme pygments numpydoc; fi
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL rstcheck; fi
# Install dependencies for code coverage (codecov.io)
if [ "$RUN_COVERAGE" == "yes" ]; then $PIP_INSTALL codecov; fi
# Install SVML
if [ "$TEST_SVML" == "yes" ]; then $CONDA_INSTALL -c numba icc_rt; fi
# Install Intel TBB parallel backend
if [ "$TEST_THREADING" == "tbb" ]; then $CONDA_INSTALL -c numba tbb=2021 tbb-devel; fi
# Install pickle5
if [ "$TEST_PICKLE5" == "yes" ]; then $PIP_INSTALL pickle5; fi
# Install typeguard
if [ "$RUN_TYPEGUARD" == "yes" ]; then $CONDA_INSTALL conda-forge::typeguard; fi

# environment dump for debug
# echo "DEBUG ENV:"
# echo "-------------------------------------------------------------------------"
# conda env export
# echo "-------------------------------------------------------------------------"
