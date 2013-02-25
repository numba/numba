#!/bin/bash

# Use -cuda switch to enable CUDA tests, which are disabled by default.
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

function run_doctest_in
{
    echo "== Testing $1"
    cd $1
    for f in ./*.py
      do
        if ! python -m doctest -v $f
          then
            echo "== Fail test in $1"
            exit
        fi
    done
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

run_test_in parallel_vectorize
run_test_in stream_vectorize
run_doctest_in prange
echo "== SKIP array_exprs"
#run_doctest_in array_exprs # known failure due to numpy 1.5 missing API

run_test test_basic_vectorize.py
run_test test_parallel_vectorize.py
run_test test_stream_vectorize.py

run_test test_gufunc.py
run_test test_mini_vectorize.py
run_test_in vectorize_decorator

if [ "$1" == "-cuda" ]
  then
    run_test_in cuda
    run_test test_cuda_vectorize.py
    run_test test_cuda_gufunc.py
    run_test test_cuda_jit.py
    run_test_in cuda_vectorize_decorator
    run_test_in cuda_vectorize_device
fi
