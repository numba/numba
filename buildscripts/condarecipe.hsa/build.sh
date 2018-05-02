#!/bin/bash

$PYTHON buildscripts/remove_unwanted_files.py

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build install
