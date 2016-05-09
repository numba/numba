#!/bin/bash

CONDA_INSTALL="conda install -q -y"
PIP_INSTALL="pip install -q"

conda update -q -y conda
# Clean up any left-over from a previous build
conda env remove -q -y -n $CONDA_ENV
# Scipy, CFFI and jinja2 are optional dependencies, but exercised in the test suite
conda create -n $CONDA_ENV -q -y python=$PYTHON numpy=$NUMPY cffi pip scipy jinja2

source activate $CONDA_ENV
set -v
# Install llvmdev (separate channel, for now)
$CONDA_INSTALL -c numba llvmdev="3.7*" llvmlite
# Install enum34 and singledispatch for Python < 3.4
if [ $PYTHON \< "3.4" ]; then $CONDA_INSTALL enum34; fi
if [ $PYTHON \< "3.4" ]; then $PIP_INSTALL singledispatch; fi
# Install funcsigs for Python < 3.3
if [ $PYTHON \< "3.3" ]; then $CONDA_INSTALL -c numba funcsigs; fi
# Install dependencies for building the documentation
if [ "$BUILD_DOC" == "yes" ]; then $CONDA_INSTALL sphinx pygments; fi
if [ "$BUILD_DOC" == "yes" ]; then $PIP_INSTALL sphinxjp.themecore sphinxjp.themes.basicstrap; fi
# Install dependencies for code coverage (codecov.io)
if [ "$RUN_COVERAGE" == "yes" ]; then $PIP_INSTALL codecov; fi
