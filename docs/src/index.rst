.. NumbaPro documentation master file, created by
   sphinx-quickstart on Wed Aug 29 09:01:25 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NumaPro
=========
NumbaPro is an enhanced version of Numba.  With NumbaPro python developers can define NumPy ufuncs, in python, which are compiled and loaded on the fly.  Additionally, NumbaPro offers developers the ability to target mutlicore architecture for maximum efficiency.

*NumbaPro can also target NVIDIA GPUs. While this functionality is being actively developed; it is, however, at the moment still in the experimental stages.*

Current Features
----------------


There are 4 versions of vectorize for building native ufunc kernels from python functions.

.. toctree::
   :maxdepth: 1

   BasicVectorize
   ParallelVectorize
   CUDAVectorize
   StreamVectorize


Dependencies
------------

* `Numba <http://numba.pydata.org/>`_
* `llvm (3.1) <http://llvm.org/releases/index.html>`_
* `llvm-py <https://github.com/llvmpy/llvmpy>`_
* `meta <http://pypi.python.org/pypi/meta/>`_



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




