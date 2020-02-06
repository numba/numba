#!/bin/bash

function usage() {
	echo "usage: ./build_for_conda.sh [ [-h | --help] | [-d | --debug] ]"
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

cd numba/dppy/dppy_driver

gcc ${DEBUG_FLAGS} -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC -c opencllite.c -o opencllite.o
#gcc -g -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC -c opencllite.c -o opencllite.o

gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "opencllite failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi

ar rcs libdpglue.a opencllite.o

#gcc -L. -shared -o libdpglue.so -Wl,--whole-archive -lopencllite
gcc ${DEBUG_FLAGS} -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -shared -fPIC -o libdpglue_so.so opencllite.c
#gcc -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -shared -fPIC -o libdpglue_so.so opencllite.c

gcc_ret=$?
if [ $gcc_ret -ne 0 ]; then
    echo "opencllite failed to build...exiting"
    cd ../../..
    exit $gcc_ret
fi

cd ../../..

LLVM_SPIRV_ERROR_MSG="We need llvm-spirv (https://github.com/KhronosGroup/SPIRV-LLVM-Translator) installed for NUMBA-PVC"
SPIRV_TOOLS_ERROR_MSG="We need SPIRV-Tools (https://github.com/KhronosGroup/SPIRV-Tools/tree/stable) installed for NUMBA-PVC"

if ! { command -v opt && command -v llvm-spirv; }; then
    echo $LLVM_SPIRV_ERRO_MSG
    exit 1
fi

if ! { command -v spirv-dis && command -v spirv-val && command -v spirv-opt; }; then
    echo $SPIRV_TOOLS_ERROR_MSG
    exit 1
fi

python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
