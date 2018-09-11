*****
Numba
*****

.. image:: https://badges.gitter.im/numba/numba.svg
   :target: https://gitter.im/numba/numba?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
   :alt: Gitter

A Just-In-Time Compiler for Numerical Functions in Python
#########################################################

Numba is an open source, NumPy-aware optimizing compiler for Python sponsored
by Anaconda, Inc.  It uses the LLVM compiler project to generate machine code
from Python syntax.

Numba can compile a large subset of numerically-focused Python, including many
NumPy functions.  Additionally, Numba has support for automatic
parallelization of loops, generation of GPU-accelerated code, and creation of
ufuncs and C callbacks.

For more information about Numba, see the Numba homepage: 
http://numba.pydata.org

Dependencies
============

* llvmlite
* NumPy (version 1.9 or higher)
* funcsigs (for Python 2)


Installing
==========

The easiest way to install Numba and get updates is by using the Anaconda
Distribution: https://www.anaconda.com/download

::

   $ conda install numba

For more options, see the Installation Guide: http://numba.pydata.org/numba-doc/latest/user/installing.html

Documentation
=============

http://numba.pydata.org/numba-doc/latest/index.html


Mailing Lists
=============

Join the Numba mailing list numba-users@continuum.io:
https://groups.google.com/a/continuum.io/d/forum/numba-users

Some old archives are at: http://librelist.com/browser/numba/


Continuous Integration
======================

.. image:: https://travis-ci.org/numba/numba.svg?branch=master
    :target: https://travis-ci.org/numba/numba
    :alt: Travis CI

.. image:: https://ci.appveyor.com/api/projects/status/klm0dk21innm33mi?svg=true)
    :target: https://ci.appveyor.com/project/seibert/numba-j46mi
    :alt: AppVeyor
