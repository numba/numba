.. _live variable analysis:

======================
Live Variable Analysis
======================

(Releated issue https://github.com/numba/numba/pull/1611)

Numba uses reference-counting for garbage collection, a technique that
requires cooperation by the compiler.  The Numba IR encodes the location
where a decref must be inserted.  These locations are determined by live
variable analysis.  The corresponding source code is the ``_insert_var_dels()``
method in https://github.com/numba/numba/blob/master/numba/interpreter.py.


In Python semantic, once a variable is defined inside a function, it is alive
until the variable is explicitly deleted or the function scope is ended.
However, Numba analyzes the code to determine the minimum bound of the lifetime
of each variable by its definition and usages during compilation.
As soon as a variable is unreachable, a ``del`` instruction is inserted at the
closest basic-block (either at the start of the next block(s) or at the
end of the current block).  This means variables can be released earlier than in
regular Python code.

The behavior of the live variable analysis affects memory usage of the compiled
code.  Internally, Numba does not differentiate temporary variables and user
variables.  Since each operation generates at least one temporary variable,
a function can accumulate a high number of temporary variables if they are not
released as soon as possible.
Our generator implementation can benefit from early releasing of variables,
which reduces the size of the state to suspend at each yield point.


Notes on behavior of the live variable analysis
================================================


Variable deleted before definition
-----------------------------------

(Related issue: https://github.com/numba/numba/pull/1738)

When a variable lifetime is confined within the loop body (its definition and
usage does not escape the loop body), like:

.. code-block:: python

    def f(arr):
      # BB 0
      res = 0
      # BB 1
      for i in (0, 1):
          # BB 2
          t = arr[i]
          if t[i] > 1:
              # BB 3
              res += t[i]
      # BB 4
      return res


Variable ``t`` is never referenced outside of the loop.
A ``del`` instruction is emitted for ``t`` at the head of the loop (BB 1)
before a variable is defined.  The reason is obvious once we know the control
flow graph::

             +------------------------------> BB4
             |
             |
    BB 0 --> BB 1  -->  BB 2 ---> BB 3
             ^          |          |
             |          V          V
             +---------------------+


Variable ``t`` is defined in BB 1.  In BB 2, the evaluation of
``t[i] > 1`` uses ``t``, which is the last use if execution takes the false
branch and goto BB 1.  In BB 3, ``t`` is only used in ``res += t[i]``, which is
the last use if execution takes the true branch.  Because BB 3, an outgoing
branch of BB 2 uses ``t``, ``t`` must be deleted at the common predecessor.
The closest point is BB 1, which does not have ``t`` defined from the incoming
edge of BB 0.

Alternatively, if ``t`` is deleted at BB 4, we will still have to delete the
variable before its definition because BB4 can be executed without executing
the loop body (BB 2 and BB 3), where the variable is defined.
