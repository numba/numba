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

if [ "$PYTHON" == "pypy" ]; then
  # Create a basic environment for PyPy - no Scipy, jinja2, or ipython (CFFI is part of PyPy)
  conda create -c gmarkall -n $CONDA_ENV -q -y pypy
else
  # Scipy, CFFI, jinja2 and IPython are optional dependencies, but exercised in the test suite
  conda create -n $CONDA_ENV -q -y python=$PYTHON numpy=$NUMPY cffi pip scipy jinja2 ipython
fi

set +v
source activate $CONDA_ENV
set -v

# Install latest llvmlite build
if [ "$PYTHON" == "pypy" ]; then
  $CONDA_INSTALL -c numba llvmlite
else
  python -m ensurepip
  $CONDA_INSTALL -c gmarkall pypy-llvmlite
  # There are no conda packages for PyPy for singledispatch and funcsigs. enum34
  # is presently rolled into the pypy-llvmlite package
  $PIP_INSTALL singledispatch funcsigs
fi

# Install enum34 and singledispatch for Python < 3.4
if [ $PYTHON \< "3.4" ]; then $CONDA_INSTALL enum34; fi
if [ $PYTHON \< "3.4" ]; then $PIP_INSTALL singledispatch; fi
# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then $CONDA_INSTALL sphinx pygments; fi
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL sphinxjp.themecore sphinxjp.themes.basicstrap; fi
# Install dependencies for code coverage (codecov.io)
if [ "$RUN_COVERAGE" == "yes" ]; then $PIP_INSTALL codecov; fi
