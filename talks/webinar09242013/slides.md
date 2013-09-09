% Pythonic Parallel Patterns for the GPU with NumbaPro
% Siu Kwan Lam
% September 24, 2013

# Overview

## NumbaPro

Commerical extension to Numba.

Goal

Accelerate numerical Python code by fully utilizing available hardware:

- Multicore CPU
- Manycore GPU
- more in the future

## As a JIT Compiler

Compile Python to:

- CUDA PTX
- Parallel CPU code

## As a Library

Bindings to:

- cuRAND
- cuBLAS
- cuFFT


## Decorators: @vectorize

Convert a scalar function into a NumPy Universal function that operates on NumPy array operands.

```python
def foo(a, b):
    return (a + b) ** 2
```

## Decorators: @vectorize Usage

```python
@vectorize([prototype0,
            prototype1,
            ...],
           target="targetname")
def a_scalar_function(a, b, ...):
    ...
```

## Decorators: @vectorize Signature example

- takes 2 float32 and returns a float32

```python
'float32(float32, float32)'
```

- takes 2 int32 and returns a int32

```python
'int32(int32, int32)'
```


## Decorators: @vectorize for CUDA

```python
@vectorize(['float32(float32, float32)'], 
           target='gpu')
def foo(a, b):
    return (a + b) ** 2

N = 10000
A = numpy.arange(N, dtype=numpy.float32)
B = numpy.arange(N, dtype=numpy.float32)

C = foo(A, B)
```

## Decorators: @vectorize for Parallel CPU cores

```python
@vectorize(['float32(float32, float32)'], 
           target='parallel')
def foo(a, b):
    return (a + b) ** 2

N = 10000
A = numpy.arange(N, dtype=numpy.float32)
B = numpy.arange(N, dtype=numpy.float32)

C = foo(A, B)
```


## Decorators: @vectorize for Single CPU core

```python
@vectorize(['float32(float32, float32)'], 
           target='cpu')
def foo(a, b):
    return (a + b) ** 2

N = 10000
A = numpy.arange(N, dtype=numpy.float32)
B = numpy.arange(N, dtype=numpy.float32)

C = foo(A, B)
```

## Decorators: @vectorize Speedup

<center>
<img width="70%" src="img/vectorize_speedup_teslaC2075.png" />
</center>

## Decorators: @guvectorize

Creates a Generialized Universal Function from a Python function

```python
@guvectorize([prototype0,
              prototype1,
              ...], 
             signature)
def gufunc_core(a, b, ...):
    ...
```

## Decorators: @vectorize Summary

## Decorators: @guvectorize Usage

- Prototypes are the same as `@vectorize`
- The signature specificies the dimension requirement:

Signature example matrix-matrix multiplication

```python
"(m, n), (n, p) -> (m, p)"
```

## Decorators: @guvectorize CUDA

```python
prototype = '''void(float32[:,:], 
                    float32[:,:], 
                    float32[:,:])'''
@guvectorize([prototype], '(m,n),(n,p)->(m,p)',
             target='gpu')
def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
```


## Decorators: @guvectorize Launch

Performs batch matrix-matrix multiplication

```python
matrix_ct = 1000
# creates an array of 1000 x 2 x 4
A = np.arange(matrix_ct * 2 * 4, 
              dtype=np.float32
              ).reshape(matrix_ct, 2, 4)
# creates an array of 1000 x 4 x 5
B = np.arange(matrix_ct * 4 * 5, 
              dtype=np.float32
              ).reshape(matrix_ct, 4, 5)
# outputs an array of 1000 x 2 x 5
C = gufunc(A, B)
```

## Decorators: @guvectorize Summary

- Similar to `@vectorize` but can use array as operands to the core function.
- Can also target the CPU.

## CUDA Library Support

## Controlling Memory Transfer


## Where to Get?

<center>
Parts of Anaconda Accelerate
https://store.continuum.io/cshop/accelerate/
</center>