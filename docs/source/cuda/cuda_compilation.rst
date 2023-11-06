
.. _cuda_compilation:

Compiling Python functions for use with other languages
=======================================================

Numba can compile Python code to PTX so that Python functions can be
incorporated into CUDA code written in other languages (e.g. C/C++).  It is
commonly used to support User-Defined Functions written in Python within the
context of a library or application.

The main API for compiling Python to PTX can be used without a GPU present, as
it uses no driver functions and avoids initializing CUDA in the process. It is
invoked through the following function:

.. autofunction:: numba.cuda.compile_ptx
   :noindex:

If a device is available and PTX for the compute capability of the current
device is required (for example when building a JIT compilation workflow using
Numba), the ``compile_ptx_for_current_device`` function can be used:

.. autofunction:: numba.cuda.compile_ptx_for_current_device
   :noindex:


Using the C ABI
---------------

Numba compiles functions with its ABI by default - this is as described in
:ref:`device-function-abi`, without the ``extern "C"`` modifier. Calling Numba
ABI device functions requires three issues to be addressed:

- The name of the function will be mangled according to Numba's ABI rules -
  these are based on the Itanium C++ ABI rules, but are extended beyond its
  specifications.
- The Python return value is expected to be stored into a pointer value passed
  in the first argument.
- The return value of the compiled function will contain a status code, instead
  of the return value of the function. For use of Numba-compiled functions
  outside of Numba, this can generally be ignored.

A simple way to address all these issues is to compile device functions with the
C ABI instead. This results in the following:

- The name of the compiled device function in PTX can be controlled. By default
  it will match the name of the function in Python, so it is easy to determine.
  This is the function's ``__name__``, rather than ``__qualname__``, because
  ``__qualname__`` encodes additional scoping information that would make the
  function name hard to predict, and in a lot of cases, an illegal identifier
  in C.
- The returned value of the Python code is placed in the return value of the
  compiled function.
- Status codes are ignored / unreported, so they do not need to be handled.

If the name of the compiled function needs to be specified, it can be controlled
by passing the name in the ``abi_info`` dict, under the key ``'abi_name'``.

C and Numba ABI examples
------------------------

The following function:

.. code:: python

   def add(x, y):
       return x + y

compiled for the Numba ABI using, for example:

.. code:: python

    ptx, resty = cuda.compile_ptx(add, int32(int32, int32), device=True)

results in PTX where the function prototype is:

.. code:: text

   .visible .func  (.param .b32 func_retval0) _ZN8__main__3addB2v1B94cw51cXTLSUwv1sCUt9Uw1VEw0NRRQPKzLTg4gaGKFsG2oMQGEYakJSQB1PQBk0Bynm21OiwU1a0UoLGhDpQE8oxrNQE_3dEii(
       .param .b64 _ZN8__main__3addB2v1B94cw51cXTLSUwv1sCUt9Uw1VEw0NRRQPKzLTg4gaGKFsG2oMQGEYakJSQB1PQBk0Bynm21OiwU1a0UoLGhDpQE8oxrNQE_3dEii_param_0,
       .param .b32 _ZN8__main__3addB2v1B94cw51cXTLSUwv1sCUt9Uw1VEw0NRRQPKzLTg4gaGKFsG2oMQGEYakJSQB1PQBk0Bynm21OiwU1a0UoLGhDpQE8oxrNQE_3dEii_param_1,
       .param .b32 _ZN8__main__3addB2v1B94cw51cXTLSUwv1sCUt9Uw1VEw0NRRQPKzLTg4gaGKFsG2oMQGEYakJSQB1PQBk0Bynm21OiwU1a0UoLGhDpQE8oxrNQE_3dEii_param_2
    )

Note that there are three parameters, for the pointer to the return value,
``x``, and ``y``. The name is mangled in a way that is hard to predict outside
of Numba internals.

Compiling for the C ABI with:

.. code:: python

   ptx, resty = cuda.compile_ptx(add, int32(int32, int32), device=True, abi="c")

instead results in the following PTX prototype:

.. code:: text

   .visible .func  (.param .b32 func_retval0) add(
       .param .b32 add_param_0,
       .param .b32 add_param_1
   )

The function name matches the Python source function name, and there are exactly
two parameters, for ``x`` and ``y``. The result of the function is directly
placed in the return value:

.. code:: text

   add.s32 	%r3, %r2, %r1;
   st.param.b32 	[func_retval0+0], %r3;

To distinguish one variant of the compiled ``add()`` function from another, the
following example specifies its ABI name in the ``abi_info`` dict:

.. code:: python

   ptx, resty = cuda.compile_ptx(add, float32(float32, float32), device=True,
                                 abi="c", abi_info={"abi_name": "add_f32"})

Resulting in the PTX prototype:

.. code:: text

   .visible .func  (.param .b32 func_retval0) add_f32(
       .param .b32 add_f32_param_0,
       .param .b32 add_f32_param_1
   )

which will not clash with definitions by other names (e.g. the variant for
``int32`` above).
