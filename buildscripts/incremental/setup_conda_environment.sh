#!/bin/bash

source activate $CONDA_ENV
set -v
# Setup environment
CONDA_INSTALL="conda install --yes -q"
PIP_INSTALL="pip install -q"
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
