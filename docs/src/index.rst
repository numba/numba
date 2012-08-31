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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




