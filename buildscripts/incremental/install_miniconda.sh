#!/bin/bash

set -v -e

# Install Miniconda
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    if [[ "$BITS32" == "yes" ]]; then
        wget https://repo.anaconda.com/pkgs/misc/conda-execs/conda-latest-linux-32.exe -O conda
    else
        wget https://repo.anaconda.com/pkgs/misc/conda-execs/conda-latest-linux-64.exe -O conda
    fi
elif [[ "$unamestr" == 'Darwin' ]]; then
    wget https://repo.anaconda.com/pkgs/misc/conda-execs/conda-latest-osx-64.exe -O conda
else
  echo Error
fi
chmod +x conda
