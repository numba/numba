
Getting started
===============

Compatibility
-------------

Numba is compatible with Python 2.7 and 3.4 or later, and Numpy versions 1.7 to 1.11.

Our supported platforms are:

* Linux x86 (32-bit and 64-bit)
* Windows 7 and later (32-bit and 64-bit)
* OS X 10.9 and later (64-bit)
* NVIDIA GPUs of compute capability 2.0 and later
* AMD APUs supported by the HSA 1.0 final runtime (Kaveri, Carrizo)


Installing using Conda
----------------------

The easiest way to install numba and get updates is by using Conda,
a cross-platform package manager and software distribution maintained
by Continuum Analytics.  You can either use `Anaconda
<http://continuum.io/downloads.html>`_ to get the full stack in one download,
or `Miniconda <http://conda.pydata.org/miniconda.html>`_ which will install
the minimum packages needed to get started.

Once you have conda installed, just type::

   $ conda install numba

or::

   $ conda update numba

Installing from source
----------------------

We won't cover requirements in detail here, but you can get the bleeding-edge
source code from `Github <https://github.com/numba/numba>`_::

   $ git clone git://github.com/numba/numba.git

Source archives of the latest release can be found on
`PyPI <https://pypi.python.org/pypi/numba/>`_.

You will need a C compiler corresponding to your Python installation, as
well as the `Numpy <http://www.numpy.org/>`_ and
`llvmlite <https://github.com/numba/llvmlite>`_ packages.  See :ref:`buildenv`
for more information.

Checking your installation
--------------------------

You should be able to import Numba from the Python prompt::

   $ python
   Python 3.4.2 |Continuum Analytics, Inc.| (default, Oct 21 2014, 17:16:37)
   [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import numba
   >>> numba.__version__
   '0.16.0-82-g350c9d2'

You can also try executing the :ref:`pycc <pycc>` utility::

   $ pycc --help
   usage: pycc [-h] [-o OUTPUT] [-c | --llvm] [--linker LINKER]
               [--linker-args LINKER_ARGS] [--header] [--python] [-d]
               inputs [inputs ...]

