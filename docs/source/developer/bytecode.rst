==========================
Notes on Bytecode Handling
==========================


``LOAD_FAST_AND_CLEAR`` opcode, ``Expr.undef`` IR Node, ``UndefVar`` type
=========================================================================

Python 3.12 introduced a new bytecode ``LOAD_FAST_AND_CLEAR`` which is solely  
used in comprehensions. The common pattern is:

.. code-block:: python

    In [1]: def foo(x):
    ...:      # 6 LOAD_FAST_AND_CLEAR      0 (x)  # push x and clear from scope
    ...:      y = [x for x in (1, 2)]             # comprehension
    ...:      # 30 STORE_FAST              0 (x)  # restore x
    ...:      return x
    ...:

    In [2]: import dis

    In [3]: dis.dis(foo)
    1           0 RESUME                   0

    3           2 LOAD_CONST               1 ((1, 2))
                4 GET_ITER
                6 LOAD_FAST_AND_CLEAR      0 (x)
                8 SWAP                     2
                10 BUILD_LIST               0
                12 SWAP                     2
            >>   14 FOR_ITER                 4 (to 26)
                18 STORE_FAST               0 (x)
                20 LOAD_FAST                0 (x)
                22 LIST_APPEND              2
                24 JUMP_BACKWARD            6 (to 14)
            >>   26 END_FOR
                28 STORE_FAST               1 (y)
                30 STORE_FAST               0 (x)

    5          32 LOAD_FAST_CHECK          0 (x)
                34 RETURN_VALUE
            >>   36 SWAP                     2
                38 POP_TOP

    3          40 SWAP                     2
                42 STORE_FAST               0 (x)
                44 RERAISE                  0
    ExceptionTable:
    10 to 26 -> 36 [2]


Numba handles the ``LOAD_FAST_AND_CLEAR`` bytecode differently to CPython 
because it relies on static instead of dynamic semantics. 

In Python, comprehensions can shadow variables from the enclosing function 
scope. To handle this, ``LOAD_FAST_AND_CLEAR`` snapshots the value of a 
potentially shadowed variable and clears it from the scope. This gives the 
illusion that comprehensions execute in a new scope, even though they are fully 
inlined in Python 3.12. The snapshotted value is later restored with 
``STORE_FAST`` after the comprehension.

Since Numba uses static semantics, it cannot precisely model the dynamic 
behavior of ``LOAD_FAST_AND_CLEAR``. Instead, Numba checks if a variable is 
used in previous opcodes to determine if it must be defined. If so, Numba 
treats it like a regular ``LOAD_FAST``. Otherwise, Numba emits an `Expr.undef`_ 
IR node to mark the stack value as undefined. Type inference assigns the 
`UndefVar`_ type to this node, allowing the value to be zero-initialized and 
implicitly cast to other types. 

In object mode, Numba uses the `\_UNDEFINED`_ sentinel object to indicate 
undefined values.

Numba does not raise ``UnboundLocalError`` if an undefined value is used.

Special case 1: ``LOAD_FAST_AND_CLEAR`` may load an undefined variable
----------------------------------------------------------------------

.. code-block:: python

    In [1]: def foo(a, v):               
    ...:      if a:
    ...:          x = v
    ...:      y = [x for x in (1, 2)]
    ...:      return x


In the above example, the variable ``x`` may or may not be defined before the 
list comprehension, depending on the truth value of ``a``. If ``a`` is ``True``, 
then ``x`` is defined and execution proceeds as described in the common case. 
However, if ``a`` is ``False``, then ``x`` is undefined. 
In this case, the Python interpreter would raise an ``UnboundLocalError`` at 
the ``return x`` line. Numba cannot determine whether ``x`` was previously 
defined, therefore it assumes ``x`` is defined to avoid the error. 
This deviates from Python's official semantics, since Numba will use a 
zero-initialized ``x`` even if it was not defined earlier.

.. code-block:: python

    In [1]: from numba import njit

    In [2]: def foo(a, v):
    ...:     if a:
    ...:         x = v
    ...:     y = [x for x in (1, 2)]
    ...:     return x
    ...:

    In [3]: foo(0, 123)
    ---------------------------------------------------------------------------
    UnboundLocalError                         Traceback (most recent call last)
    Cell In[3], line 1
    ----> 1 foo(0, 123)

    Cell In[2], line 5, in foo(a, v)
        3     x = v
        4 y = [x for x in (1, 2)]
    ----> 5 return x

    UnboundLocalError: cannot access local variable 'x' where it is not associated with a value

    In [4]: njit(foo)(0, 123)
    Out[4]: 0

As shown in the above example, Numba does not raise ``UnboundLocalError`` and
allows the function to return normally.

Special case 2: ``LOAD_FAST_AND_CLEAR`` loads undefined variable
----------------------------------------------------------------

If Numba can statically determine that a variable must be undefined,  
the type system will raise a ``TypingError`` instead of raising a ``NameError`` 
like the Python interpreter does.

.. code-block:: python
            
    In [1]: def foo():
    ...:     y = [x for x in (1, 2)]
    ...:     return x
    ...:

    In [2]: foo()
    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    Cell In[2], line 1
    ----> 1 foo()

    Cell In[1], line 3, in foo()
        1 def foo():
        2     y = [x for x in (1, 2)]
    ----> 3     return x

    NameError: name 'x' is not defined

    In [3]: from numba import njit

    In [4]: njit(foo)()
    ---------------------------------------------------------------------------
    TypingError                               Traceback (most recent call last)
    Cell In[4], line 1
    ----> 1 njit(foo)()

    File /numba/numba/core/dispatcher.py:468, in _DispatcherBase._compile_for_args(self, *args, **kws)
        464         msg = (f"{str(e).rstrip()} \n\nThis error may have been caused "
        465                f"by the following argument(s):\n{args_str}\n")
        466         e.patch_message(msg)
    --> 468     error_rewrite(e, 'typing')
        469 except errors.UnsupportedError as e:
        470     # Something unsupported is present in the user code, add help info
        471     error_rewrite(e, 'unsupported_error')

    File /numba/numba/core/dispatcher.py:409, in _DispatcherBase._compile_for_args.<locals>.error_rewrite(e, issue_type)
        407     raise e
        408 else:
    --> 409     raise e.with_traceback(None)

    TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    NameError: name 'x' is not defined


.. _UndefVar: https://github.com/numba/numba/blob/db5f0a45fcccb359cba248c4767cd1caf16c4a85/numba/core/types/misc.py#L36-L44

.. _\_UNDEFINED: https://github.com/numba/numba/blob/db5f0a45fcccb359cba248c4767cd1caf16c4a85/numba/core/pylowering.py#L32

.. _Expr.undef: https://github.com/numba/numba/blob/db5f0a45fcccb359cba248c4767cd1caf16c4a85/numba/core/ir.py#L565-L572
