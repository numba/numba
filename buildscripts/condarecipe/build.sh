#!/bin/bash

if [ `uname` == Darwin ]; then
    export CC=gcc-4.2
    if [ $CONDA_PY == 26 ]; then
        $REPLACE gcc gcc-4.2 $PREFIX/lib/python2.6/config/Makefile
    fi
fi

MVC=numbapro/_utils/mviewbuf.c


$PYTHON setup.py install
