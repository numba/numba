#!/bin/bash

function build() {
  cd $RECIPE_DIR/..
  python build.py
}

build
mkdir $PREFIX/lib
cp $RECIPE_DIR/../lib/* $PREFIX/lib
