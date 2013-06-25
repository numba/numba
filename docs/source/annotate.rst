Annotating Numba Source
=======================
Numba comes with a script to annotate numba source code. The tool can
dump generated code or types to the terminal or generate a webpage with
annotated code.

.. code-block:: bash

    $ numba -h
    usage: numba [-h] [--annotate] [--dump-llvm] [--dump-optimized] [--dump-cfg]
                 [--dump-ast] [--time-compile] [--fancy]
                 filename
    
    positional arguments:
      filename          Python source filename
    
    optional arguments:
      -h, --help        show this help message and exit
      --annotate        Annotate source
      --dump-llvm       Print generated llvm assembly
      --dump-optimized  Dump the optimized llvm assembly
      --dump-cfg        Dump the control flow graph
      --dump-ast        Dump the AST
      --time-compile    Time the compilation process
      --fancy           Try to output fancy files (.dot or .html)

For instance, it's easy to dump the generated llvm code to the terminal.

.. code-block:: bash

    $ numba --dump-llvm foo.py

    define double @__numba_specialized_0_myfunc(double %y) {
    entry:
      %return_value = alloca double
      %0 = fmul double %y, 2.000000e+00
      store double %0, double* %return_value
      br label %cleanup_label
    
    cleanup_label:                                    ; preds = %entry, %error_label
      %1 = load double* %return_value
      ret double %1
    
    error_label:                                      ; No predecessors!
      store double 0x7FF8000000000000, double* %return_value
      br label %cleanup_label
    }
    
    $ numba --dump-optimized foo.py

    define double @__numba_specialized_0_myfunc(double %y) nounwind readnone {
    entry:
      %0 = fmul double %y, 2.000000e+00
      ret double %0
    }

To get a clear idea of what the types of variables are and where parts of the code
may be slow (rely on Python), use the ``--annotate`` option. This can also be used
from the code itself::

    @jit(double(double), annotate=True)
    def square(x):
        return x * x

