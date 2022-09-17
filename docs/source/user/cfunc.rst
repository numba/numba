.. _cfunc:

====================================
Creating C callbacks with ``@cfunc``
====================================

Interfacing with some native libraries (for example written in C or C++)
can necessitate writing native callbacks to provide business logic to the
library.  The :func:`numba.cfunc` decorator creates a compiled function
callable from foreign C code, using the signature of your choice.


Basic usage
===========

The ``@cfunc`` decorator has a similar usage to ``@jit``, but with an
important difference: passing a single signature is mandatory.
It determines the visible signature of the C callback::

   from numba import cfunc

   @cfunc("float64(float64, float64)")
   def add(x, y):
       return x + y


The C function object exposes the address of the compiled C callback as
the :attr:`~CFunc.address` attribute, so that you can pass it to any
foreign C or C++ library.  It also exposes a :mod:`ctypes` callback
object pointing to that callback; that object is also callable from
Python, making it easy to check the compiled code::

   @cfunc("float64(float64, float64)")
   def add(x, y):
       return x + y

   print(add.ctypes(4.0, 5.0))  # prints "9.0"


Example
=======

In this example, we are going to be using the ``scipy.integrate.quad``
function.  That function accepts either a regular Python callback or
a C callback wrapped in a :mod:`ctypes` callback object.

Let's define a pure Python integrand and compile it as a
C callback::

   >>> import numpy as np
   >>> from numba import cfunc
   >>> def integrand(t):
           return np.exp(-t) / t**2
      ...:
   >>> nb_integrand = cfunc("float64(float64)")(integrand)

We can pass the ``nb_integrand`` object's :mod:`ctypes` callback to
``scipy.integrate.quad`` and check that the results are the same as with
the pure Python function::

   >>> import scipy.integrate as si
   >>> def do_integrate(func):
           """
           Integrate the given function from 1.0 to +inf.
           """
           return si.quad(func, 1, np.inf)
      ...:
   >>> do_integrate(integrand)
   (0.14849550677592208, 3.8736750296130505e-10)
   >>> do_integrate(nb_integrand.ctypes)
   (0.14849550677592208, 3.8736750296130505e-10)


Using the compiled callback, the integration function does not invoke the
Python interpreter each time it evaluates the integrand.  In our case, the
integration is made 18 times faster::

   >>> %timeit do_integrate(integrand)
   1000 loops, best of 3: 242 µs per loop
   >>> %timeit do_integrate(nb_integrand.ctypes)
   100000 loops, best of 3: 13.5 µs per loop


Dealing with pointers and array memory
======================================

A less trivial use case of C callbacks involves doing operation on some
array of data passed by the caller.  As C doesn't have a high-level
abstraction similar to Numpy arrays, the C callback's signature will pass
low-level pointer and size arguments.  Nevertheless, the Python code for
the callback will expect to exploit the power and expressiveness of Numpy
arrays.

In the following example, the C callback is expected to operate on 2-d arrays,
with the signature ``void(double *input, double *output, int m, int n)``.
You can implement such a callback thusly::

   from numba import cfunc, types, carray

   c_sig = types.void(types.CPointer(types.double),
                      types.CPointer(types.double),
                      types.intc, types.intc)

   @cfunc(c_sig)
   def my_callback(in_, out, m, n):
       in_array = carray(in_, (m, n))
       out_array = carray(out, (m, n))
       for i in range(m):
           for j in range(n):
               out_array[i, j] = 2 * in_array[i, j]


The :func:`numba.carray` function takes as input a data pointer and a shape
and returns an array view of the given shape over that data.  The data is
assumed to be laid out in C order.  If the data is laid out in Fortran order,
:func:`numba.farray` should be used instead.


Handling C structures
=====================


With CFFI
---------

For applications that have a lot of state, it is useful to pass data in C
structures.  To simplify the interoperability with C code, numba can convert
a ``cffi`` type into a numba ``Record`` type using
``numba.core.typing.cffi_utils.map_type``::

   from numba.core.typing import cffi_utils

   nbtype = cffi_utils.map_type(cffi_type, use_record_dtype=True)

.. note:: **use_record_dtype=True** is needed otherwise pointers to C
    structures are returned as void pointers.

.. note:: From v0.49 the ``numba.cffi_support`` module has been phased out
    in favour of ``numba.core.typing.cffi_utils``


For example::

   from cffi import FFI

   src = """

   /* Define the C struct */
   typedef struct my_struct {
      int    i1;
      float  f2;
      double d3;
      float  af4[7]; // arrays are supported
   } my_struct;

   /* Define a callback function */
   typedef double (*my_func)(my_struct*, size_t);
   """

   ffi = FFI()
   ffi.cdef(src)

   # Get the function signature from *my_func*
   sig = cffi_utils.map_type(ffi.typeof('my_func'), use_record_dtype=True)

   # Make the cfunc
   from numba import cfunc, carray

   @cfunc(sig)
   def foo(ptr, n):
      base = carray(ptr, n)  # view pointer as an array of my_struct
      tmp = 0
      for i in range(n):
         tmp += base[i].i1 * base[i].f2 / base[i].d3
         tmp += base[i].af4.sum()  # nested arrays are like normal NumPy arrays
      return tmp


With ``numba.types.Record.make_c_struct``
-----------------------------------------

The ``numba.types.Record`` type can be created manually to follow a
C-structure's layout.  To do that, use ``Record.make_c_struct``, for example::

   my_struct = types.Record.make_c_struct([
      # Provides a sequence of 2-tuples i.e. (name:str, type:Type)
      ('i1', types.int32),
      ('f2', types.float32),
      ('d3', types.float64),
      ('af4', types.NestedArray(dtype=types.float32, shape=(7,))),
   ])

Due to ABI limitations, structures should be passed as pointers
using ``types.CPointer(my_struct)`` as the argument type.  Inside the ``cfunc``
body, the ``my_struct*`` can be accessed with ``carray``.

Full example
------------

See full example in ``examples/notebooks/Accessing C Struct Data.ipynb``.


Signature specification
=======================

The explicit ``@cfunc`` signature can use any :ref:`Numba types <numba-types>`,
but only a subset of them make sense for a C callback.  You should
generally limit yourself to scalar types (such as ``int8`` or ``float64``)
,pointers to them (for example ``types.CPointer(types.int8)``), or pointers
to ``Record`` type.


Compilation options
===================

A number of keyword-only arguments can be passed to the ``@cfunc``
decorator: ``nopython`` and ``cache``.  Their meaning is similar to those
in the ``@jit`` decorator.


Calling C code from Numba
=========================

It is also possible to call C code from Numba ``@jit`` functions. In this
example, we are going to be compiling a simple function ``sum`` that adds two
integers and calling it within Numba ``@jit`` code.

.. note::
   The example below was tested on Linux and will likely work on Unix-like
   operating systems.

.. code-block:: C

   #include <stdint.h>

   int64_t sum(int64_t a, int64_t b){
      return a + b;
   }


Compile the code with ``gcc lib.c -fPIC -shared -o shared_library.so`` to
generate a shared library.

.. code-block:: python

   from numba import njit
   from numba.core import types, typing
   from llvmlite import binding
   import os

   # load the library into LLVM
   path = os.path.abspath('./shared_library.so')
   binding.load_library_permanently(path)

   # Adds typing information
   c_func_name = 'sum'
   return_type = types.int64
   argty = types.int64
   c_sig = typing.signature(return_type, argty, argty)
   c_func = types.ExternalFunction(c_func_name, c_sig)

   @njit
   def example(x, y):
      return c_func(x, y)

   print(example(3, 4)) # 7


It is also possible to use ``ctypes`` as well to call C functions. The advantage
of using ``ctypes`` is that it is invariant to the usage of JIT decorators.

.. code-block:: python

   from numba import njit
   import ctypes
   DSO = ctypes.CDLL('./shared_library.so')

   # Add typing information
   c_func = DSO.sum
   c_func.restype = ctypes.c_int
   c_func.argtypes = [ctypes.c_int, ctypes.c_int]

   @njit
   def example(x, y):
      return c_func(x, y)

   print(example(3, 4)) # 7
   print(example.py_func(3, 4)) # 7

Implementing function pointer arguments
===============================
We can pass in other C compiled functions to a cfunc as a function pointer argument.
In order to do this, we include the type ExternalFunctionPointer in the function signature.

ExternalFunctionPointer is originally designed to handle functions imported from e.g. ctypes.
It expects a method which returns the value of the pointer to be passed to the constructor.
We can simply pass a lambda to zero, since using ExternalFunctionPointer in the function signature will result
in the value of the pointer coming from the function argument.

Here is a working (contrived) example which uses a callback function to apply a generic operation to every element of an array.

.. code-block:: python

   import numpy as np
   from numba import cfunc, float64, void, intc, types

   CPointer = types.CPointer
   ExternalFunctionPointer = types.ExternalFunctionPointer

We first define the Callback type function signature and a pointer type to the callback.

.. code-block:: python

   CallbackSignature = float64(float64)
   CallbackType = ExternalFunctionPointer(CallbackSignature, get_pointer=lambda x: 0)
   
.. note:: **get_pointer=lambda x: 0** is needed since numba will try to
    evaluate the value of this pointer and needs some stand-in value.
    The value will not actually be used since the type only appears in a signature.

We now can write the driver function which consumes the function pointer. Note the use of the CallbackType in the signature.

.. code-block:: python

   @cfunc(void(CPointer(float64), intc, CallbackType))
   def apply_to_array(x, n, callback):
      for i in range(n):
         x[i] = callback(x[i])

For this example, we will implement a callback that simply adds one to each array element.

.. code-block:: python

   @cfunc(CallbackSignature)
   def add_one(x):
      return x + 1.0

To demonstrate this example, we need to use ctypes to call the compiled function from Python.
Normally numba can provide the equivalent ctypes pointer interface using ``function.ctypes``.
However, this mechanism does not know how to translate the function pointer type.
Therefore, we must do the translation manually.

.. code-block:: python

   import ctypes

   p_double = ctypes.POINTER(ctypes.c_double)

   # type of the callback in ctypes
   CallbackCtypes = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

   # create ctypes function pointer of correct type from the compiled function address
   apply_to_array_ctypes = ctypes.CFUNCTYPE(
      None, p_double, ctypes.c_int, CallbackCtypes
   )(apply_to_array.address)

Now we can call this example function from Python

.. code-block:: python

   x = np.arange(6.0)
   apply_to_array_ctypes(x.ctypes.data_as(p_double), x.size, add_one.ctypes)
   print(x)  # [1. 2. 3. 4. 5. 6.]

Note this example could also have been handled with a decorator pattern thusly:

.. code-block:: python

   def get_apply_to_array(callback):
       @cfunc(void(CPointer(float64), intc))
       def apply_to_array(x, n):
           for i in range(n):
               x[i] = callback(x[i])

       return apply_to_array
       
   x = np.arange(6.0)
   get_apply_to_array(add_one)(x.ctypes.data_as(p_double), x.size)
   print(x)  # [1. 2. 3. 4. 5. 6.]

This design pattern eliminates the problem of the function pointer altogether.

The disadvantage of this approach is that the compilation of apply_to_array cannot be cached.
For a small function such as in this example, this is not an issue.
However, if the driver function were a complex algorithm, it would be desirable to cache the compiled implementation.

Dealing with void pointer arguments
===============================
It may be the case that optional parameters are passed to a callback function
as ``void *`` or ``void * []``.
For example, scipy.integrate.quad accepts a C function input (through LowLevelCallable object) that has the following signature::

   double func(double x, void *user_data)
   
It is useful to be able to cast these pointer arguments to arbitrary types for use.
The following intrinsic implementation will cast any pointer type to any other pointer type (including function pointers).

.. code-block:: python

   from numba.extending import intrinsic
   from numba import types
   
   @intrinsic
   def cast_ptr(_typingctx, input_ptr, ptr_type):
       def impl(context, builder, _signature, args):
           llvm_type = context.get_value_type(target_type)
           val = builder.bitcast(args[0], llvm_type)
           return val

       if isinstance(ptr_type, types.TypeRef):
           target_type = ptr_type.instance_type
       else:
           target_type = ptr_type

       sig = target_type(input_ptr, ptr_type)
       return sig, impl
       
You may now use this in a callback function like the scipy example.

.. code-block:: python

   from numba import cfunc, float64, types
   
   CPointer = types.CPointer
   voidptr = types.voidptr
   
   p_double = CPointer(float64)

   @cfunc(float64(float64, voidptr))
   def my_callback(x, user_data):
       y = cast_ptr(user_data, p_double)
       return x + y[0]

To demonstrate:

.. code-block:: python

   import ctypes

   y = ctypes.c_double(2.0)
   r = my_callback.ctypes(1.0, ctypes.pointer(y))
   print(r)  # 3.0

This cast_ptr intrinsic may be also used with the function pointer technique above.

.. code-block:: python

   ExternalFunctionPointer = types.ExternalFunctionPointer

   CallbackSignature = float64(float64)
   CallbackType = ExternalFunctionPointer(CallbackSignature, get_pointer=lambda x: 0)
   
   
   @cfunc(float64(float64, voidptr))
   def my_callback_fnptr(x, user_data):
       y = cast_ptr(user_data, CallbackType)
       return y(x)
       
   @cfunc(CallbackSignature)
   def add_one(x):
     return x + 1.0

   y = add_one.ctypes
   r = my_callback_fnptr.ctypes(1.0, y)
   print(r)  # 2.0
