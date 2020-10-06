#!/bin/bash

if [ -z "$CC" ]; then
    CC=gcc
    CXX=g++
fi

function usage() {
    echo "usage: ./build_for_develop.sh [ [-h | --help] | [-d | --debug] ]"
}

DEBUG_FLAGS=""

while [ "$1" != "" ]; do
    case $1 in
        "-d" | "--debug" )         shift
                                    DEBUG_FLAGS="-g -DDEBUG"
                                ;;
        "-h" | "--help" )           usage
                                exit
                                ;;
        * )                         usage
                                exit
    esac
    shift
done

set -x

python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
