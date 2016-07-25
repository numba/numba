#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Build numba extensions without silencing compile errors
python setup.py build_ext -q --inplace
python setup.py install -q >/dev/null
