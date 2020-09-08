#!/bin/bash

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build_ext --inplace --werror --wall build install --single-version-externally-managed --record=record.txt
