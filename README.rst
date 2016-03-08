=====
Numba
=====

Numba is an Open Source NumPy-aware optimizing compiler for Python
sponsored by Continuum Analytics, Inc.  It uses the
remarkable LLVM compiler infrastructure to compile Python syntax to
machine code.

It is aware of NumPy arrays as typed memory regions and so can speed-up
code using NumPy arrays.  Other, less well-typed code will be translated
to Python C-API calls effectively removing the "interpreter" but not removing
the dynamic indirection.

Numba is also not a tracing JIT.  It *compiles* your code before it gets
run either using run-time type information or type information you provide
in the decorator.

Numba is a mechanism for producing machine code from Python syntax and typed
data structures such as those that exist in NumPy.


Dependencies
============

* llvmlite
* numpy (version 1.7 or higher)
* funcsigs (for Python 2)


Installing
==========

The easiest way to install numba and get updates is by using the Anaconda
Distribution: https://store.continuum.io/cshop/anaconda/

::

   $ conda install numba

If you wanted to compile Numba from source,
it is recommended to use conda environment to maintain multiple isolated
development environments.  To create a new environment for Numba development::

   $ conda create -p ~/dev/mynumba python numpy llvmlite

To select the installed version, append "=VERSION" to the package name,
where, "VERSION" is the version number.  For example::

   $ conda create -p ~/dev/mynumba python=2.7 numpy=1.9 llvmlite

to use Python 2.7 and Numpy 1.9.

If you need CUDA support, you should also install the CUDA toolkit::

   $ conda install cudatoolkit

This installs the CUDA Toolkit version 6.0, which requires driver version 331.00
or later to be installed.

Custom Python Environments
--------------------------

If you're not using conda, you will need to build llvmlite yourself:

Building and installing llvmlite
''''''''''''''''''''''''''''''''

See https://github.com/numba/llvmlite for the most up-to-date instructions.
You will need a build of LLVM 3.7.

::

   $ git clone https://github.com/numba/llvmlite
   $ cd llvmlite
   $ python setup.py install

Installing Numba
''''''''''''''''

::

   $ git clone https://github.com/numba/numba.git
   $ cd numba
   $ pip install -r requirements.txt
   $ python setup.py build_ext --inplace
   $ python setup.py install

or simply

::

   $ pip install numba

If you want to enable CUDA support, you will need to install CUDA Toolkit 6.0.
After installing the toolkit, you might have to specify environment variables
in order to override the standard search paths:

NUMBAPRO_CUDA_DRIVER
  Path to the CUDA driver shared library
NUMBAPRO_NVVM
  Path to the CUDA libNVVM shared library file
NUMBAPRO_LIBDEVICE
  Path to the CUDA libNVVM libdevice directory which contains .bc files


Documentation
=============

http://numba.pydata.org/numba-doc/dev/index.html


Mailing Lists
=============

Join the numba mailing list numba-users@continuum.io:
https://groups.google.com/a/continuum.io/d/forum/numba-users

or access it through the Gmane mirror:
http://news.gmane.org/gmane.comp.python.numba.user

Some old archives are at: http://librelist.com/browser/numba/


Website
=======

See if our sponsor can help you (which can help this project): http://www.continuum.io

http://numba.pydata.org


Continuous Integration
======================

https://travis-ci.org/numba/numba
