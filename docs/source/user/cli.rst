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
                 [--dump-assembly] [--dump-cfg] [--dump-ast]
                 [--annotate-html ANNOTATE_HTML] [-s]
                 [filename]

    positional arguments:
      filename              Python source filename

    optional arguments:
      -h, --help            show this help message and exit
      --annotate            Annotate source
      --dump-llvm           Print generated llvm assembly
      --dump-optimized      Dump the optimized llvm assembly
      --dump-assembly       Dump the LLVM generated assembly
      --dump-cfg            [Deprecated] Dump the control flow graph
      --dump-ast            [Deprecated] Dump the AST
      --annotate-html ANNOTATE_HTML
                            Output source annotation as html
      -s, --sysinfo         Output system information for bug reporting

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
    2019-05-07 14:15:39.733994

    __Hardware Information__
    Machine                                       : x86_64
    CPU Name                                      : haswell
    CPU count                                     : 8
    CPU Features                                  : 
    aes avx avx2 bmi bmi2 cmov cx16 f16c fma fsgsbase invpcid lzcnt mmx movbe pclmul
    popcnt rdrnd sahf sse sse2 sse3 sse4.1 sse4.2 ssse3 xsave xsaveopt

    __OS Information__
    Platform                                      : Darwin-18.5.0-x86_64-i386-64bit
    Release                                       : 18.5.0
    System Name                                   : Darwin
    Version                                       : Darwin Kernel Version 18.5.0: Mon Mar 11 20:40:32 PDT 2019; root:xnu-4903.251.3~3/RELEASE_X86_64
    OS specific info                              : 10.14.4   x86_64

    __Python Information__
    Python Compiler                               : Clang 4.0.1 (tags/RELEASE_401/final)
    Python Implementation                         : CPython
    Python Version                                : 3.7.3
    Python Locale                                 : en_US UTF-8

    __LLVM information__
    LLVM version                                  : 7.0.0

    __CUDA Information__
    CUDA driver library cannot be found or no CUDA enabled devices are present.
    Error class: <class 'numba.cuda.cudadrv.error.CudaSupportError'>

    __ROC Information__
    ROC available                                 : False
    Error initialising ROC due to                 : No ROC toolchains found.
    No HSA Agents found, encountered exception when searching:
    Error at driver init: 

    HSA is not currently supported on this platform (darwin).
    :

    __SVML Information__
    SVML state, config.USING_SVML                 : False
    SVML library found and loaded                 : False
    llvmlite using SVML patched LLVM              : True
    SVML operational                              : False

    __Threading Layer Information__
    TBB Threading layer available                 : False
    +--> Disabled due to                          : Unknown import problem.
    OpenMP Threading layer available              : False
    +--> Disabled due to                          : Unknown import problem.
    Workqueue Threading layer available           : True

    __Numba Environment Variable Information__
    None set.

    __Conda Information__
    conda_build_version                           : 3.17.8
    conda_env_version                             : 4.6.14
    platform                                      : osx-64
    python_version                                : 3.7.3.final.0
    root_writable                                 : True

    __Current Conda Env__
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
