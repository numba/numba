Installation
============

Use Anaconda
------------

The easiest way to install numba and get updates is by using the `Anaconda
Distribution <https://store.continuum.io/cshop/anaconda/>`_::

    $ conda install numba

Install from Source
-------------------

Numba main dependency is NumPy, LLVM and llvmpy.  Please refer to
http://www.llvmpy.org/ for instructions on how to install LLVM and llvmpy.
Note that Numba now depends on LLVM 3.3.

Dependencies
~~~~~~~~~~~~

* LLVM 3.3
* llvmpy (from llvmpy/llvmpy fork)
* numpy (version 1.6 or higher)
* argparse (for pycc in python2.6)
