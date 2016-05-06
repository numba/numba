#!/bin/bash

# Setup environment
conda update --yes conda
# Clean up any left-over from a previous build
conda env remove -q -y -n $CONDA_ENV
# Scipy, CFFI and jinja2 are optional dependencies, but exercised in the test suite
conda create -n $CONDA_ENV --yes python=$PYTHON numpy=$NUMPY cffi pip scipy jinja2
