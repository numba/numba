#!/bin/bash
export LD_LIBRARY_PATH=../install/lib

gcc -g -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC -c -I.. ../numba_oneapi_glue.c ${1} -lOpenCL 

