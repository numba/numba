% Pythonic Parallel Patterns for the GPU with NumbaPro
% Siu Kwan Lam
% September 24, 2013

# What is NumbaPro?

## NumbaPro

A commercial extension to Numba

## Numba

- Speedup numeric Python code
- Native code speed
- Free from the GIL

## Numba + Parallel = NumbaPro

### **Accelerate** numerical Python code by fully utilizing **parallel hardware**:

- Multicore CPUs
- Manycore GPUs

<img style="box-shadow: none; background: none; border: none;" width="33%" src="img/intelcpu.png" />
<img style="box-shadow: none; background: none; border: none;" width="33%" src="img/teslac2075.png" />

# As a JIT Compiler

## Compile Python to:

- CUDA PTX
- Parallel CPU code

## Through simple decorators

- @jit
- @autojit
- **@vectorize**
- **@guvectorize**

# As a Library

## Bindings to

- cuRAND
    - Random number generation
- cuBLAS
    - Linear algebra
- cuFFT
    - Fourier transform

# Softare Stack

## Components

<img style="box-shadow: none; border: none;" src="img/numbapro_stack.png" />

LLVM + NVVM -> PTX

# Why use NumbaPro?

## Use existing data analytics stack

<center>
| NumPy | Scipy | MatPlotlib |
| --- | --- | --- |
| NumExpr | Pandas | PyTables |
| Scikit-Learn | Scikit-Image |
</center>

## Speedup data processing
## Scale to BIG data sets with GPUs effortlessly
## CUDA architecture knowledge not required

# Pattern 1: @vectorize

## @vectorize

- Convert a scalar function into a NumPy **Universal function** that operates 
  on NumPy array operands
  
- Applies a **element-wise function** over all dimensions

## Example

```python
@vectorize(['float32(float32, float32)'], 
           target='gpu')
def foo(a, b):
    return (a + b) ** 2
```

```python
an_array = numpy.arange(10, dtype=numpy.float32)
a_scalar = numpy.float32(1.2)
foo(an_array, a_scalar)
```

- Call `foo` on any arrays or numeric scalars


## Usage

```python

@vectorize([prototype0,
            prototype1,
            ...],
           target="targetname")
def a_scalar_function(a, b, ...):
    ...

```

## Prototype

- takes 2 float32 and returns a float32

```python
'float32(float32, float32)'
```

- takes 2 int32 and returns a int32

```python
'int32(int32, int32)'
```


## Single CPU code

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

## Parallel CPU code

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


## CUDA code

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

## Benchmark

<img src="img/vectorize_speedup_teslaC2075.png" />

## Compile multiple versions for CPU and GPU

## Choose hardware according to the size of your data set

```python
def foo(a, b):
    return (a + b) ** 2

prototypes = ['float32(float32, float32)']
cpu_foo = vectorize(prototypes, target='cpu')(foo)
par_foo = vectorize(prototypes, target='parallel')(foo)
gpu_foo = vectorize(prototypes, target='gpu')(foo)
```

## Scale your program without rewriting your algorithm


# Pattern 2: @guvectorize


## Example

Creates a Generalized Universal Function from a Python function

```python
@guvectorize([prototype0,
              prototype1,
              ...], 
             signature)
def gufunc_core(a, b, ...):
    ...
```

Each element is a slice of an array

## Usage

- Prototypes are the same as `@vectorize`
- The signature specifies the dimension requirement:

Signature example matrix-matrix multiplication

```python
"(m, n), (n, p) -> (m, p)"
```

## CPU

```python

prototype = "void(float32[:,:], float32[:,:], float32[:,:])"
@guvectorize([prototype], '(m,n),(n,p)->(m,p)', target='cpu')
def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
```

## GPU

```python

prototype = "void(float32[:,:], float32[:,:], float32[:,:])"
@guvectorize([prototype], '(m,n),(n,p)->(m,p)', target='gpu')
def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
```


## In Use

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

## Similar to `@vectorize` but use array operands

# Pattern 3: Optimize only when necessary

## Manage data movement

between CPU and GPU

## Why??

- Avoid redundant transfer
- Allocate intermediate arrays on the GPU

## to_device

Copy memory to the device

```python
from numbapro import cuda
A = numpy.arange(10)    # cpu
dA = cuda.to_device(A)  # gpu
```

## copy_to_host

Copy memory to back to the host

```python
A = dA.copy_to_host()
```

Note: `dA` is a device array 

## device_array

Allocate device array like NumPy array

```python

B = cuda.device_array(shape=(2,3,4))
C = cuda.device_array(shape=10, 
                      dtype=numpy.float32)
```

# Pattern 4: Use CUDA Libraries

## Don't Rewrite

- CUDA Libraries contains highly optimized code

## FFT convolution

- Use cuFFT to build a FFT convolution.

## How?

1. forward FFT transform data and filter
2. multiply data and filter in Fourier domain
3. inverse FFT transform the product

## Forward FFT

### NumbaPro

```python
from numbapro.cudalib import cufft

fft = cufft.fft(host_or_device_array)
cufft.fft_inplace(host_or_device_array)
```

### NumPy

```python
import numpy
fft = numpy.fft.fft(array)
```

## Inverse FFT

### NumbaPro

```python
from numbapro.cudalib import cufft

ifft = cufft.ifft(host_or_device_array)
cufft.ifft_inplace(host_or_device_array)
```

### NumPy

```python
import numpy
ifft = numpy.fft.ifft(ary)
```

## Code: FFT Convolution

```python
@vectorize(['complex64(complex64, complex64)'],
           target='gpu')
def vmult(a, b):
    return a * b
```

```python
# host -> device
d_img = cuda.to_device(img)     # image
d_fltr = cuda.to_device(fltr)   # filter
# FFT forward
cufft.fft_inplace(d_img)
cufft.fft_inplace(d_fltr)
# multply
vmult(d_img, d_fltr, out=d_img) # inplace
# FFT inverse
cufft.ifft_inplace(d_img)
# device -> host
filted_img = d_img.copy_to_host()
```
Works for 1D, 2D, 3D images

## Benchmark

<img src="img/fftconvolve.png" />

# Coda

## Summary

- Use decorators to compile Python functions for the CUDA
- Manage CUDA memory transfers
- Use cuFFT to implement FFT convolution

## Questions?

<hr />

### Where to Get?

Parts of Anaconda Accelerate
[https://store.continuum.io/cshop/accelerate/](https://store.continuum.io/cshop/accelerate/)


More Examples: [https://github.com/ContinuumIO/numbapro-examples](https://github.com/ContinuumIO/numbapro-examples)

