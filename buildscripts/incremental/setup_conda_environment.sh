#!/bin/bash

set -v

CONDA_INSTALL="conda install -q -y"
PIP_INSTALL="pip install -q"

# Deactivate any environment
source deactivate
# Display root environment (for debugging)
conda list
# Clean up any left-over from a previous build
# (note workaround for https://github.com/conda/conda/issues/2679:
#  `conda env remove` issue)
conda remove --all -q -y -n $CONDA_ENV
# Scipy, CFFI, jinja2 and IPython are optional dependencies, but exercised in the test suite
conda create -n $CONDA_ENV -q -y python=$PYTHON numpy=$NUMPY cffi pip scipy jinja2 ipython

set +v
source activate $CONDA_ENV
set -v

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
