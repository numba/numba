
=================
Notes on Inlining
=================

There are occasions where it is useful to be able to inline a function
at its call site, at the Numba IR level of representation. The decorators
:func:`numba.jit` and :func:`numba.extending.overload` both
support the keyword argument ``inline``, to facilitate this behaviour.

The ``inline`` keyword argument can be one of three values:

* The string ``'never'``, this is the default and results in the function not
  being inlined under any circumstances.
* The string ``'always'``, this results in the function being inlined at all
  call sites.
* A python function that takes three arguments. The first argument is always the
  ``ir.Expr`` node that is the ``call`` requesting the inline, this is present
  to allow the function to make call contextually aware decisions. The second
  and third arguments are:

  * In the case of an untyped inline, i.e. that which occurs when using the
    :func:`numba.jit` family of decorators, both arguments are
    ``numba.ir.FunctionIR`` instances. The second argument corresponding to the
    IR of the caller, the third argument corresponding to the IR of the callee.

  * In the case of a typed inline, i.e. that which occurs when using
    :func:`numba.extending.overload`, both arguments are instances of a
    ``namedtuple`` with fields (corresponding to their standard use in the
    compiler internals):

    * ``func_ir`` - the function's Numba IR.
    * ``typemap`` - the function's type map.
    * ``calltypes`` - the call types of any calls in the function.
    * ``signature`` - the function's signature.

    The second argument holds the information from the caller, the third holds
    the information from the callee.

  In all cases the function should return True to inline and return False to not
  inline, this essentially permitting custom inlining rules (typical use might
  be cost models).

.. note:: No guarantee is made about the order in which functions are assessed
          for inlining or about the order in which they are inlined.


Example using :func:`numba.jit`
===============================

An example of using all three options to ``inline`` in the :func:`numba.njit`
decorator:

.. literalinclude:: inline_example.py

which produces the following when executed (with a print of the IR after the
legalization pass, enabled via the environment variable
``NUMBA_DEBUG_PRINT_AFTER="ir_legalization"``):

.. code-block:: none
    :emphasize-lines: 2, 3, 9, 16, 17, 21, 22, 26, 35

    label 0:
        $0.1 = global(never_inline: CPUDispatcher(<function never_inline at 0x7f890ccf9048>)) ['$0.1']
        $0.2 = call $0.1(func=$0.1, args=[], kws=(), vararg=None) ['$0.1', '$0.2']
        del $0.1                                 []
        a = $0.2                                 ['$0.2', 'a']
        del $0.2                                 []
        $0.3 = global(always_inline: CPUDispatcher(<function always_inline at 0x7f890ccf9598>)) ['$0.3']
        del $0.3                                 []
        $const0.1.0 = const(int, 200)            ['$const0.1.0']
        $0.2.1 = $const0.1.0                     ['$0.2.1', '$const0.1.0']
        del $const0.1.0                          []
        $0.4 = $0.2.1                            ['$0.2.1', '$0.4']
        del $0.2.1                               []
        b = $0.4                                 ['$0.4', 'b']
        del $0.4                                 []
        $0.5 = global(maybe_inline1: CPUDispatcher(<function maybe_inline1 at 0x7f890ccf9ae8>)) ['$0.5']
        $0.6 = call $0.5(func=$0.5, args=[], kws=(), vararg=None) ['$0.5', '$0.6']
        del $0.5                                 []
        d = $0.6                                 ['$0.6', 'd']
        del $0.6                                 []
        $const0.7 = const(int, 13)               ['$const0.7']
        magic_const = $const0.7                  ['$const0.7', 'magic_const']
        del $const0.7                            []
        $0.8 = global(maybe_inline1: CPUDispatcher(<function maybe_inline1 at 0x7f890ccf9ae8>)) ['$0.8']
        del $0.8                                 []
        $const0.1.2 = const(int, 300)            ['$const0.1.2']
        $0.2.3 = $const0.1.2                     ['$0.2.3', '$const0.1.2']
        del $const0.1.2                          []
        $0.9 = $0.2.3                            ['$0.2.3', '$0.9']
        del $0.2.3                               []
        e = $0.9                                 ['$0.9', 'e']
        del $0.9                                 []
        $0.10 = global(maybe_inline2: CPUDispatcher(<function maybe_inline2 at 0x7f890ccf9b70>)) ['$0.10']
        del $0.10                                []
        $const0.1.4 = const(int, 37)             ['$const0.1.4']
        $0.2.5 = $const0.1.4                     ['$0.2.5', '$const0.1.4']
        del $const0.1.4                          []
        $0.11 = $0.2.5                           ['$0.11', '$0.2.5']
        del $0.2.5                               []
        c = $0.11                                ['$0.11', 'c']
        del $0.11                                []
        $0.14 = a + b                            ['$0.14', 'a', 'b']
        del b                                    []
        del a                                    []
        $0.16 = $0.14 + c                        ['$0.14', '$0.16', 'c']
        del c                                    []
        del $0.14                                []
        $0.18 = $0.16 + d                        ['$0.16', '$0.18', 'd']
        del d                                    []
        del $0.16                                []
        $0.20 = $0.18 + e                        ['$0.18', '$0.20', 'e']
        del e                                    []
        del $0.18                                []
        $0.22 = $0.20 + magic_const              ['$0.20', '$0.22', 'magic_const']
        del magic_const                          []
        del $0.20                                []
        $0.23 = cast(value=$0.22)                ['$0.22', '$0.23']
        del $0.22                                []
        return $0.23                             ['$0.23']


Things to note in the above:

1. The call to the function ``never_inline`` remains as a call.
2. The ``always_inline`` function has been inlined, note its
   ``const(int, 200)`` in the caller body.
3. There is a call to ``maybe_inline1`` before the ``const(int, 13)``
   declaration, the cost model prevented this from being inlined.
4. After the ``const(int, 13)`` the subsequent call to ``maybe_inline1`` has
   been inlined as shown by the ``const(int, 300)`` in the caller body.
5. The function ``maybe_inline2`` has been inlined as demonstrated by
   ``const(int, 37)`` in the caller body.
6. That dead code elimination has not been performed and as a result there are
   superfluous statements present in the IR.


Example using :func:`numba.extending.overload`
==============================================

An example of using inlining with the  :func:`numba.extending.overload`
decorator. It is most interesting to note that if a function is supplied as the
argument to ``inline`` a lot more information is available via the supplied
function arguments for use in decision making. Also that different
``@overload`` s can have different inlining behaviours, with multiple ways to
achieve this:

.. literalinclude:: inline_overload_example.py

which produces the following when executed (with a print of the IR after the
legalization pass, enabled via the environment variable
``NUMBA_DEBUG_PRINT_AFTER="ir_legalization"``):

.. code-block:: none
    :emphasize-lines: 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 30

    label 0:
        $const0.2 = const(tuple, (1, 2, 3))      ['$const0.2']
        x.0 = $const0.2                          ['$const0.2', 'x.0']
        del $const0.2                            []
        $const0.2.2 = const(int, 0)              ['$const0.2.2']
        $0.3.3 = getitem(value=x.0, index=$const0.2.2) ['$0.3.3', '$const0.2.2', 'x.0']
        del x.0                                  []
        del $const0.2.2                          []
        $0.4.4 = $0.3.3                          ['$0.3.3', '$0.4.4']
        del $0.3.3                               []
        $0.3 = $0.4.4                            ['$0.3', '$0.4.4']
        del $0.4.4                               []
        a = $0.3                                 ['$0.3', 'a']
        del $0.3                                 []
        $const0.5 = const(int, 100)              ['$const0.5']
        x.5 = $const0.5                          ['$const0.5', 'x.5']
        del $const0.5                            []
        $const0.2.7 = const(int, 1)              ['$const0.2.7']
        $0.3.8 = x.5 + $const0.2.7               ['$0.3.8', '$const0.2.7', 'x.5']
        del x.5                                  []
        del $const0.2.7                          []
        $0.4.9 = $0.3.8                          ['$0.3.8', '$0.4.9']
        del $0.3.8                               []
        $0.6 = $0.4.9                            ['$0.4.9', '$0.6']
        del $0.4.9                               []
        b = $0.6                                 ['$0.6', 'b']
        del $0.6                                 []
        $0.7 = global(bar: <function bar at 0x7f6c3710d268>) ['$0.7']
        $const0.8 = const(complex, 300j)         ['$const0.8']
        $0.9 = call $0.7($const0.8, func=$0.7, args=[Var($const0.8, inline_overload_example.py (56))], kws=(), vararg=None) ['$0.7', '$0.9', '$const0.8']
        del $const0.8                            []
        del $0.7                                 []
        c = $0.9                                 ['$0.9', 'c']
        del $0.9                                 []
        $0.12 = a + b                            ['$0.12', 'a', 'b']
        del b                                    []
        del a                                    []
        $0.14 = $0.12 + c                        ['$0.12', '$0.14', 'c']
        del c                                    []
        del $0.12                                []
        $0.15 = cast(value=$0.14)                ['$0.14', '$0.15']
        del $0.14                                []
        return $0.15                             ['$0.15']

Things to note in the above:

1. The first highlighted section is the always inlined overload for the
   ``UniTuple`` argument type.
2. The second highlighted section is the overload for the ``Number`` argument
   type that has been inlined as the cost model function decided to do so as the
   argument was an ``Integer`` type instance.
3. The third highlighted section is the overload for the ``Number`` argument
   type that has not inlined as the cost model function decided to reject it as
   the argument was an ``Complex`` type instance.
4. That dead code elimination has not been performed and as a result there are
   superfluous statements present in the IR.
