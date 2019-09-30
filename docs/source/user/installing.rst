
Installation
============

Compatibility
-------------

Numba is compatible with Python 2.7 and 3.5 or later, and Numpy versions 1.7 to
1.16.

Our supported platforms are:

* Linux x86 (32-bit and 64-bit)
* Linux ppcle64 (POWER8)
* Windows 7 and later (32-bit and 64-bit)
* OS X 10.9 and later (64-bit)
* NVIDIA GPUs of compute capability 2.0 and later
* AMD ROC dGPUs (linux only and not for AMD Carrizo or Kaveri APU)
* ARMv7 (32-bit little-endian, such as Raspberry Pi 2 and 3)
* ARMv8 (64-bit little-endian, such as the NVIDIA Jetson)

:ref:`numba-parallel` is only available on 64-bit platforms,
and is not supported in Python 2.7 on Windows.

Installing using conda on x86/x86_64/POWER Platforms
----------------------------------------------------

The easiest way to install Numba and get updates is by using ``conda``,
a cross-platform package manager and software distribution maintained
by Anaconda, Inc.  You can either use `Anaconda
<https://www.anaconda.com/download>`_ to get the full stack in one download,
or `Miniconda <https://conda.io/miniconda.html>`_ which will install
the minimum packages required for a conda environment.

Once you have conda installed, just type::

    $ conda install numba

or::

    $ conda update numba

Note that Numba, like Anaconda, only supports PPC in 64-bit little-endian mode.

To enable CUDA GPU support for Numba, install the latest `graphics drivers from
NVIDIA <https://www.nvidia.com/Download/index.aspx>`_ for your platform.
(Note that the open source Nouveau drivers shipped by default with many Linux
distributions do not support CUDA.)  Then install the ``cudatoolkit`` package::

    $ conda install cudatoolkit

You do not need to install the CUDA SDK from NVIDIA.


Installing using pip on x86/x86_64 Platforms
--------------------------------------------

Binary wheels for Windows, Mac, and Linux are also available from `PyPI
<https://pypi.org/project/numba/>`_.  You can install Numba using ``pip``::

    $ pip install numba

This will download all of the needed dependencies as well.  You do not need to
have LLVM installed to use Numba (in fact, Numba will ignore all LLVM
versions installed on the system) as the required components are bundled into
the llvmlite wheel.

To use CUDA with Numba installed by `pip`, you need to install the `CUDA SDK
<https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA.  Please refer to
:ref:`cudatoolkit-lookup` for details. Numba can also detect CUDA libraries
installed system-wide on Linux.

Enabling AMD ROCm GPU Support
-----------------------------

The `ROCm Platform <https://rocm.github.io/>`_ allows GPU computing with AMD
GPUs on Linux.  To enable ROCm support in Numba,  conda is required, so begin
with an Anaconda or Miniconda installation with Numba 0.40 or later installed.
Then:

1. Follow the `ROCm installation instructions <https://rocm.github.io/install.html>`_.
2. Install ``roctools`` conda package from the ``numba`` channel::

    $ conda install -c numba roctools

See the `roc-examples <https://github.com/numba/roc-examples>`_ repository for
sample notebooks.


.. _numba-install-armv7:

Installing on Linux ARMv7 Platforms
-----------------------------------

`Berryconda <https://github.com/jjhelmus/berryconda>`_ is a
conda-based Python distribution for the Raspberry Pi.  We are now uploading
packages to the ``numba`` channel on Anaconda Cloud for 32-bit little-endian,
ARMv7-based boards, which currently includes the Raspberry Pi 2 and 3,
but not the Pi 1 or Zero.  These can be installed using conda from the
``numba`` channel::

    $ conda install -c numba numba

Berryconda and Numba may work on other Linux-based ARMv7 systems, but this has
not been tested.


Installing on Linux ARMv8 (AArch64) Platforms
---------------------------------------------

We build and test conda packages on the `NVIDIA Jetson TX2
<https://www.nvidia.com/en-us/autonomous-machines/embedded-systems-dev-kits-modules/>`_,
but they are likely to work for other AArch64 platforms.  (Note that while the
Raspberry Pi CPU is 64-bit, Raspbian runs it in 32-bit mode, so look at
:ref:`numba-install-armv7` instead.)

Conda-forge support for AArch64 is still quite experimental and packages are limited,
but it does work enough for Numba to build and pass tests.  To set up the environment:

* Install `conda4aarch64 <https://github.com/jjhelmus/conda4aarch64/releases>`_.
  This will create a minimal conda environment.
* Add the ``c4aarch64`` and ``conda-forge`` channels to your conda
  configuration::

    $ conda config --add channels c4aarch64
    $ conda config --add channels conda-forge

* Then you can install Numba from the ``numba`` channel::

    $ conda install -c numba numba

On CUDA-enabled systems, like the Jetson, the CUDA toolkit should be
automatically detected in the environment.

.. _numba-source-install-instructions:

Installing from source
----------------------

Installing Numba from source is fairly straightforward (similar to other
Python packages), but installing `llvmlite
<https://github.com/numba/llvmlite>`_ can be quite challenging due to the need
for a special LLVM build.  If you are building from source for the purposes of
Numba development, see :ref:`buildenv` for details on how to create a Numba
development environment with conda.

If you are building Numba from source for other reasons, first follow the
`llvmlite installation guide <https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html>`_.
Once that is completed, you can download the latest Numba source code from
`Github <https://github.com/numba/numba>`_::

    $ git clone git://github.com/numba/numba.git

Source archives of the latest release can also be found on
`PyPI <https://pypi.org/project/numba/>`_.  In addition to ``llvmlite``, you will also need:

* A C compiler compatible with your Python installation.  If you are using
  Anaconda, you can use the following conda packages:

  * Linux ``x86``: ``gcc_linux-32`` and ``gxx_linux-32``
  * Linux ``x86_64``: ``gcc_linux-64`` and ``gxx_linux-64``
  * Linux ``POWER``: ``gcc_linux-ppc64le`` and ``gxx_linux-ppc64le``
  * Linux ``ARM``: no conda packages, use the system compiler
  * Mac OSX: ``clang_osx-64`` and ``clangxx_osx-64`` or the system compiler at
    ``/usr/bin/clang`` (Mojave onwards)
  * Windows: a version of Visual Studio appropriate for the Python version in
    use

* `NumPy <http://www.numpy.org/>`_

Then you can build and install Numba from the top level of the source tree::

    $ python setup.py install

.. _numba-source-install-check:

Dependency List
---------------

Numba has numerous required and optional dependencies which additionally may
vary with target operating system and hardware. The following lists them all
(as of September 2019).

* Required build time:

  * ``setuptools``
  * ``numpy``
  * ``llvmlite``
  * ``funcsigs`` (Python 2)
  * ``singledispatch`` (Python 2)
  * Compiler toolchain mentioned above

* Optional build time:

  * ``llvm-openmp`` (OSX) - provides headers for compiling OpenMP support into
    Numba's threading backend
  * ``intel-openmp`` (OSX) - provides OpenMP library support for Numba's
    threading backend.
  * ``tbb-devel`` - provides TBB headers/libraries for compiling TBB support
    into Numba's threading backend

* Required run time:

  * ``numpy``
  * ``llvmlite``
  * ``funcsigs`` (Python 2)
  * ``singledispatch`` (Python 2)

* Optional runtime are:

  * ``scipy`` - provides cython bindings used in Numba's ``np.linalg.*``
    support
  * ``tbb`` - provides the TBB runtime libraries used by Numba's TBB threading
    backend
  * ``jinja2`` - for "pretty" type annotation output (HTML) via the ``numba``
    CLI
  * ``cffi`` - permits use of CFFI bindings in Numba compiled functions
  * ``intel-openmp`` - (OSX) provides OpenMP library support for Numba's OpenMP
    threading backend
  * ``ipython`` - if in use, caching will use IPython's cache
    directories/caching still works
  * ``pyyaml`` - permits the use of a ``.numba_config.yaml``
    file for storing per project configuration options
  * ``colorama`` - makes error message highlighting work
  * ``icc_rt`` - (numba channel) allows Numba to use Intel SVML for extra
    performance
  * ``pygments`` - for "pretty" type annotation
  * ``gdb`` as an executable on the ``$PATH`` - if you would like to use the gdb
    support
  * Compiler toolchain mentioned above, if you would like to use ``pycc`` for
    Ahead-of-Time (AOT) compilation

* To build the documentation:

  * ``sphinx``
  * ``pygments``
  * ``sphinx-bootstrap``
  * ``numpydoc``
  * ``make`` as an executable on the ``$PATH``

Checking your installation
--------------------------

You should be able to import Numba from the Python prompt::

    $ python
    Python 2.7.15 |Anaconda custom (x86_64)| (default, May  1 2018, 18:37:05)
    [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numba
    >>> numba.__version__
    '0.39.0+0.g4e49566.dirty'

You can also try executing the ``numba --sysinfo`` (or ``numba -s`` for short)
command to report information about your system capabilities. See :ref:`cli` for
further information.

::

    $ numba -s
    System info:
    --------------------------------------------------------------------------------
    __Time Stamp__
    2018-08-28 15:46:24.631054

    __Hardware Information__
    Machine                             : x86_64
    CPU Name                            : haswell
    CPU Features                        :
    aes avx avx2 bmi bmi2 cmov cx16 f16c fma fsgsbase lzcnt mmx movbe pclmul popcnt
    rdrnd sse sse2 sse3 sse4.1 sse4.2 ssse3 xsave xsaveopt

    __OS Information__
    Platform                            : Darwin-17.6.0-x86_64-i386-64bit
    Release                             : 17.6.0
    System Name                         : Darwin
    Version                             : Darwin Kernel Version 17.6.0: Tue May  8 15:22:16 PDT 2018; root:xnu-4570.61.1~1/RELEASE_X86_64
    OS specific info                    : 10.13.5   x86_64

    __Python Information__
    Python Compiler                     : GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)
    Python Implementation               : CPython
    Python Version                      : 2.7.15
    Python Locale                       : en_US UTF-8

    __LLVM information__
    LLVM version                        : 6.0.0

    __CUDA Information__
    Found 1 CUDA devices
    id 0         GeForce GT 750M                              [SUPPORTED]
                          compute capability: 3.0
                               pci device id: 0
                                  pci bus id: 1

(output truncated due to length)


