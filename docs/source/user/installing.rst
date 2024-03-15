
Installation
============

Compatibility
-------------

For software compatibility, please see the section on :ref:`version support
information<numba_support_info>` for details.

Our supported platforms are:

* Linux x86_64
* Linux ppcle64 (POWER8, POWER9)
* Windows 10 and later (64-bit)
* OS X 10.9 and later (64-bit and unofficial support on M1/Arm64)
* \*BSD (unofficial support only)
* NVIDIA GPUs of compute capability 5.0 and later

  * Compute capabilities 3.5 and 3.7 are supported, but deprecated.
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

To use CUDA with Numba installed by ``pip``, you need to install the `CUDA SDK
<https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA.  Please refer to
:ref:`cudatoolkit-lookup` for details. Numba can also detect CUDA libraries
installed system-wide on Linux.


Installing on Linux ARMv8 (AArch64) Platforms
---------------------------------------------

We build and test conda packages on the `NVIDIA Jetson TX2
<https://www.nvidia.com/en-us/autonomous-machines/embedded-systems-dev-kits-modules/>`_,
but they are likely to work for other AArch64 platforms.  (Note that while the
CPUs in the Raspberry Pi 3, 4, and Zero 2 W are 64-bit, Raspberry Pi OS may be
running in 32-bit mode depending on the OS image in use).

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

    $ git clone https://github.com/numba/numba.git

Source archives of the latest release can also be found on
`PyPI <https://pypi.org/project/numba/>`_.  In addition to ``llvmlite``, you will also need:

* A C compiler compatible with your Python installation.  If you are using
  Anaconda, you can use the following conda packages:

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

If you wish to run the test suite, see the instructions in the
:ref:`developer documentation <running-tests>`.

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

  * ``numpy``
  * ``llvmlite``

* Optional build time:

  See :ref:`numba-source-install-env_vars` for more details about additional
  options for the configuration and specification of these optional components.

  * ``llvm-openmp`` (OSX) - provides headers for compiling OpenMP support into
    Numba's threading backend
  * ``tbb-devel`` - provides TBB headers/libraries for compiling TBB support
    into Numba's threading backend (version >= 2021.6 required).
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
  * ``intel-cmplr-lib-rt`` - allows Numba to use Intel SVML for extra
    performance
  * ``pygments`` - for "pretty" type annotation
  * ``gdb`` as an executable on the ``$PATH`` - if you would like to use the gdb
    support
  * ``setuptools`` - permits the use of ``pycc`` for Ahead-of-Time (AOT)
    compilation
  * Compiler toolchain mentioned above, if you would like to use ``pycc`` for
    Ahead-of-Time (AOT) compilation
  * ``r2pipe`` - required for assembly CFG inspection.
  * ``radare2`` as an executable on the ``$PATH`` - required for assembly CFG
    inspection. `See here <https://github.com/radareorg/radare2>`_ for
    information on obtaining and installing.
  * ``graphviz`` - for some CFG inspection functionality.
  * ``typeguard`` - used by ``runtests.py`` for
    :ref:`runtime type-checking <type_anno_check>`.
  * ``cuda-python`` - The NVIDIA CUDA Python bindings. See :ref:`cuda-bindings`.
    Numba requires Version 11.6 or greater.
  * ``cubinlinker`` and ``ptxcompiler`` to support
    :ref:`minor-version-compatibility`.


* To build the documentation:

  * ``sphinx``
  * ``pygments``
  * ``sphinx_rtd_theme``
  * ``numpydoc``
  * ``make`` as an executable on the ``$PATH``

.. _numba_support_info:

Version support information
---------------------------

This is the canonical reference for information concerning which versions of
Numba's dependencies were tested and known to work against a given version of
Numba. Other versions of the dependencies (especially NumPy) may work reasonably
well but were not tested. The use of ``x`` in a version number indicates all
patch levels supported. The use of ``?`` as a version is due to missing
information.

+----------++--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| Numba     | Release date | Python                    | NumPy                      | llvmlite                     | LLVM              | TBB                         |
+===========+==============+===========================+============================+==============================+===================+=============================+
| 0.59.0    | 2024-01-31   | 3.9.x <= version < 3.13   | 1.22 <= version < 1.27     | 0.42.x                       | 14.x              | 2021.6 <= version           |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.58.1    | 2023-10-17   | 3.8.x <= version < 3.12   | 1.22 <= version < 1.27     | 0.41.x                       | 14.x              | 2021.6 <= version           |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.58.0    | 2023-09-20   | 3.8.x <= version < 3.12   | 1.22 <= version < 1.26     | 0.41.x                       | 14.x              | 2021.6 <= version           |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.57.1    | 2023-06-21   | 3.8.x <= version < 3.12   | 1.21 <= version < 1.25     | 0.40.x                       | 14.x              | 2021.6 <= version           |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.57.0    | 2023-05-01   | 3.8.x <= version < 3.12   | 1.21 <= version < 1.25     | 0.40.x                       | 14.x              | 2021.6 <= version           |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.56.4    | 2022-11-03   | 3.7.x <= version < 3.11   | 1.18 <= version < 1.24     | 0.39.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.56.3    | 2022-10-13   | 3.7.x <= version < 3.11   | 1.18 <= version < 1.24     | 0.39.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.56.2    | 2022-09-01   | 3.7.x <= version < 3.11   | 1.18 <= version < 1.24     | 0.39.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.56.1    | NO RELEASE   |                           |                            |                              |                   |                             |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.56.0    | 2022-07-25   | 3.7.x <= version < 3.11   | 1.18 <= version < 1.23     | 0.39.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.55.2    | 2022-05-25   | 3.7.x <= version < 3.11   | 1.18 <= version < 1.23     | 0.38.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.55.{0,1}| 2022-01-13   | 3.7.x <= version < 3.11   | 1.18 <= version < 1.22     | 0.38.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.54.x    | 2021-08-19   | 3.6.x <= version < 3.10   | 1.17 <= version < 1.21     | 0.37.x                       | 11.x              | 2021.x                      |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.53.x    | 2021-03-11   | 3.6.x <= version < 3.10   | 1.15 <= version < 1.21     | 0.36.x                       | 11.x              | 2019.5 <= version < 2021.4  |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.52.x    | 2020-11-30   | 3.6.x <= version < 3.9    | 1.15 <= version < 1.20     | 0.35.x                       | 10.x              | 2019.5 <= version < 2020.3  |
|           |              |                           |                            |                              | (9.x for aarch64) |                             |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.51.x    | 2020-08-12   | 3.6.x <= version < 3.9    | 1.15 <= version < 1.19     | 0.34.x                       | 10.x              | 2019.5 <= version < 2020.0  |
|           |              |                           |                            |                              | (9.x for aarch64) |                             |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.50.x    | 2020-06-10   | 3.6.x <= version < 3.9    | 1.15 <= version < 1.19     | 0.33.x                       | 9.x               | 2019.5 <= version < 2020.0  |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.49.x    | 2020-04-16   | 3.6.x <= version < 3.9    | 1.15 <= version < 1.18     | 0.31.x <= version < 0.33.x   | 9.x               | 2019.5 <= version < 2020.0  |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.48.x    | 2020-01-27   | 3.6.x <= version < 3.9    | 1.15 <= version < 1.18     | 0.31.x                       | 8.x               | 2018.0.5 <= version < ?     |
|           |              |                           |                            |                              | (7.x for ppc64le) |                             |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+
| 0.47.x    | 2020-01-02   | 3.5.x <= version < 3.9;   | 1.15 <= version < 1.18     | 0.30.x                       | 8.x               | 2018.0.5 <= version < ?     |
|           |              | version == 2.7.x          |                            |                              | (7.x for ppc64le) |                             |
+-----------+--------------+---------------------------+----------------------------+------------------------------+-------------------+-----------------------------+

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
