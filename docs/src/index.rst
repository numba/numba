.. NumbaPro documentation master file, created by
   sphinx-quickstart on Wed Aug 29 09:01:25 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NumaPro
=========
NumbaPro is an enhanced version of Numba. 


Current Features
----------------


Fast Vectorize
--------------

There are 4 versions of vectorize for building native ufunc kernels from python functions.

They are:

1. BasicVectorize_ (`numbapro.vectorize.basic`) -- A simple wrapper for vectorize
2. ParallelVectorize (`numbapro.vectorize.parallel`) -- A multi-threaded version of vectorize.
3. CUDAVectorize	 (`numbapro.vectorize.cuda`) -- A vectorize that uses Nvidia CUDA GPU.
4. StreamVectorize (`numbapro.vectorize.stream`) -- A cache optimized version.  (Work in progress)


.. toctree::
   :maxdepth: 1

   BasicVectorize
   ParallelVectorize
   CUDAVectorize
   StreamVectorize



All these vectorize versions use Numba to compile the functions into LLVM.  Then, NumbaPro wraps and enhances these LLVM functions into ufuncs.  User may choose different version of vectorize depending on the complexity of the workload.  Cache optimization, multithreading and GPU computation have their associated overheads.  Therefore, there may not be much benefits for smaller workloads.


* normal --- this is a standard ufunc that just executes the inner loop calling the provided function each time
* parallel --- this one uses threads (pthreads and soon windows threads as appropriate) to execute parts of the inner loop in parallel
* stream --- this first copies memory to aligned, contiguous buffers (small enough presumably to fit in cache) and then executes the code on the chunk
* gpu (cuda) --- this is currently not technically a ufunc (it's a simulated ufunc).   It takes the translated code from Numba and converts it to PTX and then uses pycuda to load the PTX code on the device and then call it with array inputs.
        * the next release of Numba should remove the dependency on pycuda and make the call inside the ufunc wrapper (like parallel and stream) --- if someone is ambitious and wants to do this, it would be great because I'd really rather not have numbapro depend on pycuda, but I want to show the gpu stuff.

* general --- this creates a "generalized ufunc" from the numba-compiled code.
        * I presume we could create parallel, stream, and gpu versions of this as well eventually (perhaps even before shipping this week with some help).

These features are implemented using an expert-mode feature of NumbaPro called llvm_cbuilder.   The llvm_cbuilder classes allow one to translate to llvm code from C-like expressions using Python syntax.   It's enabled Siu to make rapid progress.    I'm not sure how much of this we will advertise to end-users, actually, but these very cool classes (that make clever use of context-managers) will be in NumbaPro.

The final feature is a relatively simple script that takes Numba-compiled code in modules and creates either .bc files, .o files or a shared-library out of the python code.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




