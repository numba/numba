#!/bin/bash

set -v -e

# Install Miniconda
unamestr=`uname`
archstr=`uname -m`
if [[ "$unamestr" == 'Linux' ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
elif [[ "$unamestr" == 'Darwin' ]]; then
    if [[ "$archstr" == 'arm64' ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    fi
else
  echo Error
fi
chmod +x miniconda.sh
bash ./miniconda.sh -b
