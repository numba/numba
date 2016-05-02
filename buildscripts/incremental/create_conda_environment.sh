#!/bin/bash

# Setup environment
conda update --yes conda
# Scipy, CFFI and jinja2 are optional dependencies, but exercised in the test suite
conda create -n travisci --yes python=$PYTHON numpy=$NUMPY cffi pip scipy jinja2
