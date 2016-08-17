#!/bin/bash

source activate $CONDA_ENV

# Make sure any error below is reported as such
set -v -e

if [ "$RUN_COVERAGE" == "yes" ]; then
    coverage combine
    codecov
fi
