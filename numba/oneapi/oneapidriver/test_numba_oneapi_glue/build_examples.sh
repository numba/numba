#!/bin/bash

rm -rf *.o
gcc -g -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC  -I.. ../numba_oneapi_glue.c ${1} -lOpenCL 
