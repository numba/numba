#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

# Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install -e .
