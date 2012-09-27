#!/bin/bash

function run_test_in
{
  echo "== Testing $1"
  cd $1
  if ! python -m unittest discover -vfp "*.py"
    then
      echo "== Fail test in $1"
      exit
  fi
  cd ..
}

function run_test
{
    echo "== Testing $1"
    if ! python $1
      then
        echo "== Fail test in $1"
        exit
    fi
}

run_test_in llvm_cbuilder_tests
run_test_in basic_vectorize
run_test_in parallel_vectorize
run_test_in stream_vectorize

run_test test_basic_vectorize.py
run_test test_parallel_vectorize.py
run_test test_stream_vectorize.py

run_test test_gufunc.py
run_test test_cuda_vectorize.py
run_test test_mini_vectorize.py

run_test test_cuda_jit.py
