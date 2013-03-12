===============================
Compute Unit (OpenCL-like) API
===============================

The compute unit (CU) API provides a portable interface for heterogeneous
(CPU+GPU) parallel programming.  This API is similar to OpenCL.  Users write
data parallel compute kernels which are executed in parallel with
no guarantee of execution order of each thread of the kernel.  The kernels are 
executed asynchronously and, therefore, it must be synchronized to ensure the 
execution of the kernels are completed and the outputs of the kernels are 
observable to the launching thread.

Kernels must be written in a restricted subset of Python that can be translated
to low-level representation without the Python runtime:

- does not return any value;
- does not call other python functions except a set of special math functions
  in `math` or `numpy`; such as: sin, cos, log, exp...

Kernels are "enqueue"-ed into a CU.  This means the kernels are scheduled to
execute asynchronously.  To synchronize the execution, the launching thread
calls the "wait" method on the CU.

A Kernel
---------

A kernel function is a python function with the following signature::

    def an_example_kernel(tid, arg0, arg1, ..., argN):
        pass
        
A kernel never returns any value (except None).  The first argument is the 
id of the current logical thread.  The id always starts with 0 and ends at 
`ntid - 1`, where `ntid` is passed to `cu.enqueue` (which is described in the next
session).  The remaining arguments are passed into the kernel through 
`cu.enqueue`.  A kernel can have no additional argument but it will not be 
useful, because the only way for the kernel to have side-effect is to store
output data to an array passed in as an argument.

Logically the execution is similar to::

    for tid in range(ntid):
        an_example_kernel(tid, arg0, arg1, ..., argN)

Except for the fact that there is no guarantee for the execution order of the 
loop body.  The kernel should be data parallel so that we executes correctly 
when running in parallel or sequentially.

The CU Object
--------------

A CU object represents a compute unit on the machine.

::

    from numbapro import CU
    cu = CU(target)

where target is a string name of target; either 'cpu' or 'gpu'.

The 'cpu' target uses multiple OS threads.  The 'gpu' target uses the CUDA card 
on the machine.

cu.enqueue(kernel, ntid, args=())
=================================

Enqueue a kernel as an asynchronous task to the CU object.

Parameters:

    **kernel**
    
        `function`
        
        A python function representing the kernel
        
    **ntid**
        
        `int`
        
        Number of logical threads to launch.  A logical thread does not imply
        a hardware thread or a OS thread.  The kernel will receive tid from

        
    **args**
    
        `sequence, optional`
        
        A sequence (usually tuple) of arguments for the kernel.  Does not
        include the first (`tid`) argument.
        

cu.wait()
=========

Wait until all previously enqueued asynchronous tasks to complete.

cu.close()
==========

Release the resources of the CU object.  This is especially important for
CPU target to terminate cleanly.  The best practive is to use 
`contextlib.closing`.

::

    from contextlib import closing
    
    cu = CU('cpu')
    with closing(cu):
        ...


Data Arrays
------------
The CU API is tightly integrated with the numpy array.  To pass arrays to 
the CU object, use the `input`, `output`, `inout`, `scratch` and `scratch_like`
methods of the CU object.  

For example::

    inary = np.arange(10)
    outary = np.empty(10)
    inoutary = np.arange((20, 10))

    d_inary = cu.input(inary)       # tag as input
    d_outary = cu.output(outary)    # tag as output
    d_inoutary = cu.inout(inoutary) # tag as both input and output
    d_scratch1 = cu.scratch(5, dtype=numpy.double)  # scratchpad memory of double type
    d_scratch2 = cu.scratch_like(outary) # scratchpad memory with shape, strides and dtype of outary.

Except for `scratch` these methods take a numpy 
array object as argument.  The `scratch` method is similar to `numpy.empty`.
It has the following signature::

    scratch(shape, dtype=numpy.float, order='C')
    
User should treat the return value of these methods as an opaque handle to 
the memory buffer on the CU.  Only use these handles as arguments to kernel 
calls.


Math Support
--------------

The following math functions are supported on all targets:
::

    np.log(x)
    math.log(x)

    np.exp(x)
    math.exp(x)

    np.sqrt(x)
    math.sqrt(x)

    abs(x)
    
The following math functions are *only* supported on CPU target:
::

    np.sin(x)
    math.sin(x)

    np.cos(x)
    math.cos(x)


Builtins Kernels
---------------

There are a few builtin kernels supplied under the namespace 
`numbapro.parallel.kernel.builtins`.

Currently, NumbaPro only contains builtin random number generator kernels for
the GPU target.

builtins.random.seed
=======================

*Not available for CPU yet*

Configure the seed of the PRNG.

Uses cuRAND internally for GPU target.


Example::

    from numbapro.parallel.kernel import builtins
    cu.configure(builtins.random.seed, 12345)  # set the seed to 12345


builtins.random.uniform
=======================

*Not available for CPU yet*

The `builtins.random.uniform` kernel generates uniformly distributed random 
number in the half open internal [0, 1) and writing to `out[:ntid]`.

Uses cuRAND internally for GPU target.

Arguments:

    **out** 
        `array, output` 

        A 1-D contiguous array of 32-bit or 64-bit float only.

Example::

    from numbapro.parallel.kernel import builtins
    rnd = numpy.empty(123)
    d_rnd = cu.output(rnd)
    cu.enqueue(builtins.random.uniform,
               ntid = d_rnd.size,       # ntid controls the # of elements
               args = (d_rnd,))
               

builtins.random.normal
=======================

*Not available for CPU yet*

The `builtins.random.normal` kernel generates normally distributed random
number and writing to `out[:ntid]`.

Uses cuRAND internally for GPU target.

Arguments:

    **out**     
        `array, output`

        A 1-D contiguous array of 32-bit or 64-bit float only.

    **mean = 0**
        `number, optional`

        Center of the distribution.
    
    **sigma = 1**   
        `number, optional` 
        
        Standard deviation of the distribution.
        

Example::

    from numbapro.parallel.kernel import builtins
    rnd = numpy.empty(123)
    d_rnd = cu.output(rnd)
    cu.enqueue(builtins.random.normal,
               ntid = d_rnd.size,       # ntid controls the # of elements
               args = (d_rnd,))
               


Examples
---------


Saxpy
=====

Implement Saxpy in two kernels.

::

    from contextlib import closing
    import numpy as np
    from numbapro import CU

    def product(tid, A, B, Prod):
        Prod[tid] = A[tid] * B[tid]

    def sum(tid, A, B, Sum):
        Sum[tid] = A[tid] + B[tid]

    cu = CU('cpu') # or 'gpu' if you have CUDA
    with closing(cu):
        n = 100
        # input arrays
        A = np.arange(n)
        B = np.arange(n)
        C = np.arange(n)

        # output arrays
        D = np.empty(n)
        
        # tag the arrays
        dA = cu.input(A)
        dB = cu.input(B)
        dC = cu.input(C)
        dProd = cu.scratch_like(D)
        dSum  = cu.output(D)

        cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
        cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))

        cu.wait() # synchronize

        print(D)                         # print values
        print(np.allclose(A * B + C, D)) # verify

