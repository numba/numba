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
    (output truncated due to length)

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
