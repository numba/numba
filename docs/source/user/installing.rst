
Installation
============

Compatibility
-------------

Numba is compatible with Python 3.7--3.10, and Numpy versions 1.18 up
to 1.22.

Our supported platforms are:

* Linux x86 (32-bit and 64-bit)
* Linux ppcle64 (POWER8, POWER9)
* Windows 7 and later (32-bit and 64-bit)
* OS X 10.9 and later (64-bit and unofficial support on M1/Arm64)
* \*BSD (unofficial support only)
* NVIDIA GPUs of compute capability 5.3 and later

  * Compute capabilities 3.0 - 5.2 are supported, but deprecated.
* ARMv7 (32-bit little-endian, such as Raspberry Pi 2 and 3)
* ARMv8 (64-bit little-endian, such as the NVIDIA Jetson)

:ref:`numba-parallel` is only available on 64-bit platforms.

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

* Install `miniforge <https://github.com/conda-forge/miniforge>`_.
  This will create a minimal conda environment.

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
  * Mac OSX (M1): ``clang_osx-arm64`` and ``clangxx_osx-arm64``
  * Windows: a version of Visual Studio appropriate for the Python version in
    use

* `NumPy <http://www.numpy.org/>`_

Then you can build and install Numba from the top level of the source tree::

    $ python setup.py install

.. _numba-source-install-env_vars:

Build time environment variables and configuration of optional components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are environment variables that are applicable to altering how Numba would
otherwise build by default along with information on configuration options.

.. envvar:: NUMBA_DISABLE_OPENMP (default: not set)

  To disable compilation of the OpenMP threading backend set this environment
  variable to a non-empty string when building. If not set (default):

  * For Linux and Windows it is necessary to provide OpenMP C headers and
    runtime  libraries compatible with the compiler tool chain mentioned above,
    and for these to be accessible to the compiler via standard flags.
  * For OSX the conda package ``llvm-openmp`` provides suitable C headers and
    libraries. If the compilation requirements are not met the OpenMP threading
    backend will not be compiled.

.. envvar:: NUMBA_DISABLE_TBB (default: not set)

  To disable the compilation of the TBB threading backend set this environment
  variable to a non-empty string when building. If not set (default) the TBB C
  headers and libraries must be available at compile time. If building with
  ``conda build`` this requirement can be met by installing the ``tbb-devel``
  package. If not building with ``conda build`` the requirement can be met via a
  system installation of TBB or through the use of the ``TBBROOT`` environment
  variable to provide the location of the TBB installation. For more
  information about setting ``TBBROOT`` see the `Intel documentation <https://software.intel.com/content/www/us/en/develop/documentation/advisor-user-guide/top/appendix/adding-parallelism-to-your-program/adding-the-parallel-framework-to-your-build-environment/defining-the-tbbroot-environment-variable.html>`_.

.. _numba-source-install-check:

Dependency List
---------------

Numba has numerous required and optional dependencies which additionally may
vary with target operating system and hardware. The following lists them all
(as of July 2020).

* Required build time:

  * ``setuptools``
  * ``numpy``
  * ``llvmlite``
  * Compiler toolchain mentioned above

* Required run time:

  * ``setuptools``
  * ``numpy``
  * ``llvmlite``

* Optional build time:

  See :ref:`numba-source-install-env_vars` for more details about additional
  options for the configuration and specification of these optional components.

  * ``llvm-openmp`` (OSX) - provides headers for compiling OpenMP support into
    Numba's threading backend
  * ``tbb-devel`` - provides TBB headers/libraries for compiling TBB support
    into Numba's threading backend (2021 <= version < 2021.6 required).
  * ``importlib_metadata`` (for Python versions < 3.9)

* Optional runtime are:

  * ``scipy`` - provides cython bindings used in Numba's ``np.linalg.*``
    support
  * ``tbb`` - provides the TBB runtime libraries used by Numba's TBB threading
    backend (version >= 2021 required).
  * ``jinja2`` - for "pretty" type annotation output (HTML) via the ``numba``
    CLI
  * ``cffi`` - permits use of CFFI bindings in Numba compiled functions
  * ``llvm-openmp`` - (OSX) provides OpenMP library support for Numba's OpenMP
    threading backend.
  * ``intel-openmp`` - (OSX) provides an alternative OpenMP library for use with
    Numba's OpenMP threading backend.
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
  * ``r2pipe`` - required for assembly CFG inspection.
  * ``radare2`` as an executable on the ``$PATH`` - required for assembly CFG
    inspection. `See here <https://github.com/radareorg/radare2>`_ for
    information on obtaining and installing.
  * ``graphviz`` - for some CFG inspection functionality.
  * ``pickle5`` - provides Python 3.8 pickling features for faster pickling in
    Python 3.7.
  * ``typeguard`` - used by ``runtests.py`` for
    :ref:`runtime type-checking <type_anno_check>`.
  * ``cuda-python`` - The NVIDIA CUDA Python bindings. See :ref:`cuda-bindings`.
    Numba requires Version 11.6 or greater.

* To build the documentation:

  * ``sphinx``
  * ``pygments``
  * ``sphinx_rtd_theme``
  * ``numpydoc``
  * ``make`` as an executable on the ``$PATH``

Checking your installation
--------------------------

You should be able to import Numba from the Python prompt::

    $ python
    Python 3.10.2 | packaged by conda-forge | (main, Jan 14 2022, 08:02:09) [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numba
    >>> numba.__version__
    '0.55.1'

You can also try executing the ``numba --sysinfo`` (or ``numba -s`` for short)
command to report information about your system capabilities. See :ref:`cli` for
further information.

::

    $ numba -s
    System info:
    --------------------------------------------------------------------------------
    __Time Stamp__
    Report started (local time)                   : 2022-01-18 10:35:08.981319

    __Hardware Information__
    Machine                                       : x86_64
    CPU Name                                      : skylake-avx512
    CPU Count                                     : 12
    CPU Features                                  :
    64bit adx aes avx avx2 avx512bw avx512cd avx512dq avx512f avx512vl bmi bmi2
    clflushopt clwb cmov cx16 cx8 f16c fma fsgsbase fxsr invpcid lzcnt mmx
    movbe pclmul pku popcnt prfchw rdrnd rdseed rtm sahf sse sse2 sse3 sse4.1
    sse4.2 ssse3 xsave xsavec xsaveopt xsaves

    __OS Information__
    Platform Name                                 : Linux-5.4.0-94-generic-x86_64-with-glibc2.31
    Platform Release                              : 5.4.0-94-generic
    OS Name                                       : Linux
    OS Version                                    : #106-Ubuntu SMP Thu Jan 6 23:58:14 UTC 2022

    __Python Information__
    Python Compiler                               : GCC 9.4.0
    Python Implementation                         : CPython
    Python Version                                : 3.10.2
    Python Locale                                 : en_GB.UTF-8

    __LLVM information__
    LLVM Version                                  : 11.1.0

    __CUDA Information__
    Found 1 CUDA devices
    id 0      b'Quadro RTX 8000'                              [SUPPORTED]
                          Compute Capability: 7.5
                               PCI Device ID: 0
                                  PCI Bus ID: 21
                                        UUID: GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643
                                    Watchdog: Enabled
                 FP32/FP64 Performance Ratio: 32

(output truncated due to length)
