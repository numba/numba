#!/bin/bash
cp -r $RECIPE_DIR/../.. src
cd src
$PYTHON setup.py install
