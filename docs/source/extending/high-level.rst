
.. _high-level-extending:

High-level extension API
========================

This extension API is exposed through the :mod:`numba.extending` module.

To aid debugging extensions to Numba, it's recommended to set the following
environment variable::

    NUMBA_CAPTURED_ERRORS="new_style"

this makes it easy to differentiate between errors in implementation and
acceptable errors that can take part in e.g. type inference. For more
information see :envvar:`NUMBA_CAPTURED_ERRORS`.

Implementing functions
----------------------

The ``@overload`` decorator allows you to implement arbitrary functions
for use in :term:`nopython mode` functions.  The function decorated with
``@overload`` is called at compile-time with the *types* of the function's
runtime arguments.  It should return a callable representing the
*implementation* of the function for the given types.  The returned
implementation is compiled by Numba as if it were a normal function
decorated with ``@jit``.  Additional options to ``@jit`` can be passed as
dictionary using the ``jit_options`` argument.

For example, let's pretend Numba doesn't support the :func:`len` function
on tuples yet.  Here is how to implement it using ``@overload``::

   from numba import types
   from numba.extending import overload

   @overload(len)
   def tuple_len(seq):
      if isinstance(seq, types.BaseTuple):
          n = len(seq)
          def len_impl(seq):
              return n
          return len_impl


You might wonder, what happens if :func:`len()` is called with something
else than a tuple? If a function decorated with ``@overload`` doesn't
return anything (i.e. returns None), other definitions are tried until
one succeeds.  Therefore, multiple libraries may overload :func:`len()`
for different types without conflicting with each other.

Implementing methods
--------------------

The ``@overload_method`` decorator similarly allows implementing a
method on a type well-known to Numba.

.. autofunction:: numba.core.extending.overload_method

Implementing classmethods
-------------------------

The ``@overload_classmethod`` decorator similarly allows implementing a
classmethod on a type well-known to Numba.

.. autofunction:: numba.core.extending.overload_classmethod


Implementing attributes
-----------------------

The ``@overload_attribute`` decorator allows implementing a data
attribute (or property) on a type.  Only reading the attribute is
possible; writable attributes are only supported through the
:ref:`low-level API <low-level-extending>`.

The following example implements the :attr:`~numpy.ndarray.nbytes` attribute
on Numpy arrays::

   @overload_attribute(types.Array, 'nbytes')
   def array_nbytes(arr):
      def get(arr):
          return arr.size * arr.itemsize
      return get

.. _cython-support:

Importing Cython Functions
--------------------------

The function ``get_cython_function_address`` obtains the address of a
C function in a Cython extension module. The address can be used to
access the C function via a :func:`ctypes.CFUNCTYPE` callback, thus
allowing use of the C function inside a Numba jitted function. For
example, suppose that you have the file ``foo.pyx``::

   from libc.math cimport exp

   cdef api double myexp(double x):
       return exp(x)

You can access ``myexp`` from Numba in the following way::

   import ctypes
   from numba.extending import get_cython_function_address

   addr = get_cython_function_address("foo", "myexp")
   functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
   myexp = functype(addr)

The function ``myexp`` can now be used inside jitted functions, for
example::

   @njit
   def double_myexp(x):
       return 2*myexp(x)

One caveat is that if your function uses Cython's fused types, then
the function's name will be mangled. To find out the mangled name of
your function you can check the extension module's ``__pyx_capi__``
attribute.

Implementing intrinsics
-----------------------

The ``@intrinsic`` decorator is used for marking a function *func* as typing and
implementing the function in ``nopython`` mode using the
`llvmlite IRBuilder API <http://llvmlite.pydata.org/en/latest/user-guide/ir/ir-builder.html>`_.
This is an escape hatch for expert users to build custom LLVM IR that will be
inlined into the caller, there is no safety net!

The first argument to *func* is the typing context.  The rest of the arguments
corresponds to the type of arguments of the decorated function. These arguments
are also used as the formal argument of the decorated function.  If *func* has
the signature ``foo(typing_context, arg0, arg1)``, the decorated function will
have the signature ``foo(arg0, arg1)``.

The return values of *func* should be a 2-tuple of expected type signature, and
a code-generation function that will passed to
:func:`~numba.targets.imputils.lower_builtin`. For an unsupported operation,
return ``None``.

Here is an example that cast any integer to a byte pointer::

    from numba import types
    from numba.extending import intrinsic

    @intrinsic
    def cast_int_to_byte_ptr(typingctx, src):
        # check for accepted types
        if isinstance(src, types.Integer):
            # create the expected type signature
            result_type = types.CPointer(types.uint8)
            sig = result_type(types.uintp)
            # defines the custom code generation
            def codegen(context, builder, signature, args):
                # llvm IRBuilder code here
                [src] = args
                rtype = signature.return_type
                llrtype = context.get_value_type(rtype)
                return builder.inttoptr(src, llrtype)
            return sig, codegen

it may be used as follows::

    from numba import njit

    @njit('void(int64)')
    def foo(x):
        y = cast_int_to_byte_ptr(x)

    foo.inspect_types()

and the output of ``.inspect_types()`` demonstrates the cast (note the
``uint8*``)::

    def foo(x):

        #   x = arg(0, name=x)  :: int64
        #   $0.1 = global(cast_int_to_byte_ptr: <intrinsic cast_int_to_byte_ptr>)  :: Function(<intrinsic cast_int_to_byte_ptr>)
        #   $0.3 = call $0.1(x, func=$0.1, args=[Var(x, check_intrin.py (24))], kws=(), vararg=None)  :: (uint64,) -> uint8*
        #   del x
        #   del $0.1
        #   y = $0.3  :: uint8*
        #   del y
        #   del $0.3
        #   $const0.4 = const(NoneType, None)  :: none
        #   $0.5 = cast(value=$const0.4)  :: none
        #   del $const0.4
        #   return $0.5

        y = cast_int_to_byte_ptr(x)


Implementing mutable structures
-------------------------------

.. warning:: This is an experimental feature, the API may change without warning.

The ``numba.experimental.structref`` module provides utilities for defining
mutable pass-by-reference structures, a ``StructRef``. The following example
demonstrates how to define a basic mutable structure:

Defining a StructRef
''''''''''''''''''''

.. literalinclude:: ../../../numba/tests/doc_examples/test_structref_usage.py
   :language: python
   :caption: from ``numba/tests/doc_examples/test_structref_usage.py``
   :start-after: magictoken.ex_structref_type_definition.begin
   :end-before: magictoken.ex_structref_type_definition.end
   :dedent: 0
   :linenos:

The following demonstrates using the above mutable struct definition:

.. literalinclude:: ../../../numba/tests/doc_examples/test_structref_usage.py
   :language: python
   :caption: from ``test_type_definition`` of ``numba/tests/doc_examples/test_structref_usage.py``
   :start-after: magictoken.ex_structref_type_definition_test.begin
   :end-before: magictoken.ex_structref_type_definition_test.end
   :dedent: 8
   :linenos:


Defining a method on StructRef
''''''''''''''''''''''''''''''

Methods and attributes can be attached using ``@overload_*`` as shown in the
previous sections.

The following demonstrates the use of ``@overload_method`` to insert a
method for instances of ``MyStructType``:

.. literalinclude:: ../../../numba/tests/doc_examples/test_structref_usage.py
   :language: python
   :caption: from ``test_overload_method`` of ``numba/tests/doc_examples/test_structref_usage.py``
   :start-after: magictoken.ex_structref_method.begin
   :end-before: magictoken.ex_structref_method.end
   :dedent: 8
   :linenos:


``numba.experimental.structref`` API Reference
''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: numba.experimental.structref
    :members:

Determining if a function is already wrapped by a ``jit`` family decorator
--------------------------------------------------------------------------

The following function is provided for this purpose.

.. automethod:: numba.extending.is_jitted
