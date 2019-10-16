#!/bin/sh

# Have to be on the pvc branch of Numba.

find . -name "*.so" -exec rm {} \;
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
