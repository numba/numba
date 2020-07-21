#! /bin/bash
clang -flto  -target spir64-unknown-unknown -c -x cl -emit-llvm  -cl-std=CL2.0 -Xclang -finclude-default-header atomic_op.cl -o atomic_op.bc
llvm-spirv -o ./atomic_op.spir atomic_op.bc


clang -flto  -target spir64-unknown-unknown -c -x cl -emit-llvm  -cl-std=CL2.0 -Xclang -finclude-default-header atomic_op_driver.cl -o atomic_op_driver.bc
llvm-spirv -o ./atomic_op_driver.spir atomic_op_driver.bc

spirv-link -o atomic_op_final.spir  atomic_op_driver.spir atomic_op.spir

gcc atomic_add_driver.c -lOpenCL -o atomic_add_driver
