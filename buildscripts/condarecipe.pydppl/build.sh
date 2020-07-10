#!/bin/bash

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build install --single-version-externally-managed --record=record.txt
