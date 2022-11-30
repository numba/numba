.. _cli:

Command line interface
======================

Numba is a Python package, usually you ``import numba`` from Python and use the
Python application programming interface (API). However, Numba also ships with a
command line interface (CLI), i.e. a tool ``numba`` that is installed when you
install Numba.

Currently, the only purpose of the CLI is to allow you to quickly show some
information about your system and installation, or to quickly get some debugging
information for a Python script using Numba.

.. _cli_usage:

Usage
-----

To use the Numba CLI from the terminal, use ``numba`` followed by the options
and arguments like ``--help`` or ``-s``, as explained below.

Sometimes it can happen that you get a "command not found" error when you type
``numba``, because your ``PATH`` isn't configured properly. In that case you can
use the equivalent command ``python -m numba``. If that still gives "command
not found", try to ``import numba`` as suggested here:
:ref:`numba-source-install-check`.

The two versions ``numba`` and ``python -m numba`` are the same. The first is
shorter to type, but if you get a "command not found" error because your
``PATH`` doesn't contain the location where ``numba`` is installed, having the
``python -m numba`` variant is useful.

To use the Numba CLI from IPython or Jupyter, use ``!numba``, i.e. prefix the
command with an exclamation mark. This is a general IPython/Jupyter feature to
execute shell commands, it is not available in the regular ``python`` terminal.

.. _cli_help:

Help
----

To see all available options, use ``numba --help``::

    $ numba --help
    usage: numba [-h] [--annotate] [--dump-llvm] [--dump-optimized]
                 [--dump-assembly] [--annotate-html ANNOTATE_HTML] [-s]
                 [--sys-json SYS_JSON]
                 [filename]

    positional arguments:
    filename              Python source filename

    optional arguments:
    -h, --help            show this help message and exit
    --annotate            Annotate source
    --dump-llvm           Print generated llvm assembly
    --dump-optimized      Dump the optimized llvm assembly
    --dump-assembly       Dump the LLVM generated assembly
    --annotate-html ANNOTATE_HTML
                            Output source annotation as html
    -s, --sysinfo         Output system information for bug reporting
    --sys-json SYS_JSON   Saves the system info dict as a json file


.. _cli_sysinfo:

System information
------------------

The ``numba -s`` (or the equivalent ``numba --sysinfo``) command prints a lot of
information about your system and your Numba installation and relevant
dependencies.

Remember: you can use ``!numba -s`` with an exclamation mark to see this
information from IPython or Jupyter.

Example output::

    $ numba -s

    System info:
    --------------------------------------------------------------------------------
    __Time Stamp__
    Report started (local time)                   : 2022-11-30 15:40:42.368114
    UTC start time                                : 2022-11-30 15:40:42.368129
    Running time (s)                              : 2.563586

    __Hardware Information__
    Machine                                       : x86_64
    CPU Name                                      : ivybridge
    CPU Count                                     : 3
    Number of accessible CPUs                     : ?
    List of accessible CPUs cores                 : ?
    CFS Restrictions (CPUs worth of runtime)      : None

    CPU Features                                  : 64bit aes avx cmov cx16 cx8 f16c
                                                    fsgsbase fxsr mmx pclmul popcnt
                                                    rdrnd sahf sse sse2 sse3 sse4.1
                                                    sse4.2 ssse3 xsave

    Memory Total (MB)                             : 14336
    Memory Available (MB)                         : 11540

    __OS Information__
    Platform Name                                 : macOS-10.16-x86_64-i386-64bit
    Platform Release                              : 20.6.0
    OS Name                                       : Darwin
    OS Version                                    : Darwin Kernel Version 20.6.0: Thu Sep 29 20:15:11 PDT 2022; root:xnu-7195.141.42~1/RELEASE_X86_64
    OS Specific Version                           : 10.16   x86_64
    Libc Version                                  : ?

    __Python Information__
    Python Compiler                               : Clang 14.0.6
    Python Implementation                         : CPython
    Python Version                                : 3.10.8
    Python Locale                                 : en_US.UTF-8

    __Numba Toolchain Versions__
    Numba Version                                 : 0+untagged.gb91eec710
    llvmlite Version                              : 0.40.0dev0+43.g7783803

    __LLVM Information__
    LLVM Version                                  : 11.1.0

    __CUDA Information__
    CUDA Device Initialized                       : False
    CUDA Driver Version                           : ?
    CUDA Runtime Version                          : ?
    CUDA NVIDIA Bindings Available                : ?
    CUDA NVIDIA Bindings In Use                   : ?
    CUDA Detect Output:
    None
    CUDA Libraries Test Output:
    None

    __NumPy Information__
    NumPy Version                                 : 1.23.4
    NumPy Supported SIMD features                 : ('MMX', 'SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C')
    NumPy Supported SIMD dispatch                 : ('SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2', 'AVX512F', 'AVX512CD', 'AVX512_KNL', 'AVX512_SKX', 'AVX512_CLX', 'AVX512_CNL', 'AVX512_ICL')
    NumPy Supported SIMD baseline                 : ('SSE', 'SSE2', 'SSE3')
    NumPy AVX512_SKX support detected             : False

    __SVML Information__
    SVML State, config.USING_SVML                 : False
    SVML Library Loaded                           : False
    llvmlite Using SVML Patched LLVM              : True
    SVML Operational                              : False

    __Threading Layer Information__
    TBB Threading Layer Available                 : True
    +-->TBB imported successfully.
    OpenMP Threading Layer Available              : True
    +-->Vendor: Intel
    Workqueue Threading Layer Available           : True
    +-->Workqueue imported successfully.

    __Numba Environment Variable Information__
    None found.

    __Conda Information__
    Conda Build                                   : not installed
    Conda Env                                     : 4.12.0
    Conda Platform                                : osx-64
    Conda Python Version                          : 3.9.12.final.0
    Conda Root Writable                           : True

    __Installed Packages__
    appnope                   0.1.2           py310hecd8cb5_1001    defaults
    asttokens                 2.0.5              pyhd3eb1b0_0    defaults
    backcall                  0.2.0              pyhd3eb1b0_0    defaults
    blas                      1.0                         mkl    defaults
    bzip2                     1.0.8                h1de35cc_0    defaults
    ca-certificates           2022.10.11           hecd8cb5_0    defaults
    cctools_osx-64            949.0.1             hc7db93f_25    defaults
    certifi                   2022.9.24       py310hecd8cb5_0    defaults
    cffi                      1.15.1          py310h6c40b1e_2    defaults
    clang                     14.0.6               hecd8cb5_0    defaults
    clang-14                  14.0.6          default_h32c6d10_0    defaults
    clang_osx-64              14.0.6               hb1e4b1b_0    defaults
    clangxx                   14.0.6          default_h32c6d10_0    defaults
    clangxx_osx-64            14.0.6               hd8b9576_0    defaults
    compiler-rt               14.0.6               hda8b6b8_0    defaults
    compiler-rt_osx-64        14.0.6               h8d5cb93_0    defaults
    debugpy                   1.5.1           py310he9d5cce_0    defaults
    decorator                 5.1.1              pyhd3eb1b0_0    defaults
    entrypoints               0.4             py310hecd8cb5_0    defaults
    executing                 0.8.3              pyhd3eb1b0_0    defaults
    fftw                      3.3.9                h9ed2024_1    defaults
    gitdb                     4.0.7              pyhd3eb1b0_0    defaults
    gitpython                 3.1.18             pyhd3eb1b0_1    defaults
    importlib-metadata        4.11.3          py310hecd8cb5_0    defaults
    importlib_metadata        4.11.3               hd3eb1b0_0    defaults
    intel-openmp              2021.4.0          hecd8cb5_3538    defaults
    ipykernel                 6.15.2          py310hecd8cb5_0    defaults
    ipython                   8.6.0           py310hecd8cb5_0    defaults
    jedi                      0.18.1          py310hecd8cb5_1    defaults
    jinja2                    3.1.2           py310hecd8cb5_0    defaults
    jupyter_client            7.4.7           py310hecd8cb5_0    defaults
    jupyter_core              4.11.2          py310hecd8cb5_0    defaults
    ld64_osx-64               530                 h70f3046_25    defaults
    ldid                      2.1.2                h2d21305_2    defaults
    libclang-cpp14            14.0.6          default_h32c6d10_0    defaults
    libcxx                    14.0.6               h9765a3e_0    defaults
    libffi                    3.4.2                hecd8cb5_6    defaults
    libgfortran               5.0.0           11_3_0_hecd8cb5_28    defaults
    libgfortran5              11.3.0              h9dfd629_28    defaults
    libllvm14                 14.0.6               h91fad77_1    defaults
    libsodium                 1.0.18               h1de35cc_0    defaults
    llvm-openmp               14.0.6               h0dcd299_0    defaults
    llvm-tools                14.0.6               h91fad77_1    defaults
    llvmlite                  0.40.0dev0             py310_43    numba/label/dev
    markupsafe                2.1.1           py310hca72f7f_0    defaults
    matplotlib-inline         0.1.6           py310hecd8cb5_0    defaults
    mkl                       2021.4.0           hecd8cb5_637    defaults
    mkl-service               2.4.0           py310hca72f7f_0    defaults
    mkl_fft                   1.3.1           py310hf879493_0    defaults
    mkl_random                1.2.2           py310hc081a56_0    defaults
    ncurses                   6.3                  hca72f7f_3    defaults
    nest-asyncio              1.5.5           py310hecd8cb5_0    defaults
    numba                     0+untagged.gb91eec710           dev_0    <develop>
    numpy                     1.23.4          py310h9638375_0    defaults
    numpy-base                1.23.4          py310ha98c3c9_0    defaults
    openssl                   1.1.1s               hca72f7f_0    defaults
    packaging                 21.3               pyhd3eb1b0_0    defaults
    parso                     0.8.3              pyhd3eb1b0_0    defaults
    pexpect                   4.8.0              pyhd3eb1b0_3    defaults
    pickleshare               0.7.5           pyhd3eb1b0_1003    defaults
    pip                       22.2.2          py310hecd8cb5_0    defaults
    prompt-toolkit            3.0.20             pyhd3eb1b0_0    defaults
    psutil                    5.9.0           py310hca72f7f_0    defaults
    ptyprocess                0.7.0              pyhd3eb1b0_2    defaults
    pure_eval                 0.2.2              pyhd3eb1b0_0    defaults
    pycparser                 2.21               pyhd3eb1b0_0    defaults
    pygments                  2.11.2             pyhd3eb1b0_0    defaults
    pyparsing                 3.0.9           py310hecd8cb5_0    defaults
    python                    3.10.8               h218abb5_1    defaults
    python-dateutil           2.8.2              pyhd3eb1b0_0    defaults
    pyyaml                    6.0                      pypi_0    pypi
    pyzmq                     23.2.0          py310he9d5cce_0    defaults
    readline                  8.2                  hca72f7f_0    defaults
    scipy                     1.9.3           py310h09290a1_0    defaults
    setuptools                65.5.0          py310hecd8cb5_0    defaults
    six                       1.16.0             pyhd3eb1b0_1    defaults
    smmap                     4.0.0              pyhd3eb1b0_0    defaults
    sqlite                    3.40.0               h880c91c_0    defaults
    stack_data                0.2.0              pyhd3eb1b0_0    defaults
    tapi                      1000.10.8            ha1b3eb9_0    defaults
    tbb                       2021.5.0             haf03e11_0    defaults
    tbb-devel                 2021.5.0             haf03e11_0    defaults
    tk                        8.6.12               h5d9f67b_0    defaults
    tornado                   6.2             py310hca72f7f_0    defaults
    traitlets                 5.1.1              pyhd3eb1b0_0    defaults
    typing-extensions         4.3.0           py310hecd8cb5_0    defaults
    typing_extensions         4.3.0           py310hecd8cb5_0    defaults
    tzdata                    2022f                h04d1e81_0    defaults
    wcwidth                   0.2.5              pyhd3eb1b0_0    defaults
    wheel                     0.37.1             pyhd3eb1b0_0    defaults
    xz                        5.2.6                hca72f7f_0    defaults
    yaml                      0.2.5                haf1e3a3_0    defaults
    zeromq                    4.3.4                h23ab428_0    defaults
    zipp                      3.8.0           py310hecd8cb5_0    defaults
    zlib                      1.2.13               h4dc903c_0    defaults

    No errors reported.


    __Warning log__
    Warning (cuda): CUDA driver library cannot be found or no CUDA enabled devices are present.
    Exception class: <class 'numba.cuda.cudadrv.error.CudaSupportError'>

.. _cli_debug:

Debugging
---------

As shown in the help output above, the ``numba`` command includes options that
can help you to debug Numba compiled code.

To try it out, create an example script called ``myscript.py``::

    import numba

    @numba.jit
    def f(x):
        return 2 * x

    f(42)

and then execute one of the following commands::

    $ numba myscript.py --annotate
    $ numba myscript.py --annotate-html myscript.html
    $ numba myscript.py --dump-llvm
    $ numba myscript.py --dump-optimized
    $ numba myscript.py --dump-assembly
