
.. code:: python

    from __future__ import print_function
    import numba
    import numpy as np
    import math
    import llvm
    import ctypes
    print("numba version: %s \nNumPy version: %s\nllvm version: %s" % (numba.__version__,np.__version__, llvm.__version__))

.. parsed-literal::

    numba version: 0.12.0 
    NumPy version: 1.7.1
    llvm version: 0.12.1


Numba and types
===============


Introduction
------------


*Numba* translates *Python* code into fast executing native code. In
order to generate fast native code, many dynamic features of *Python*
need to be translated into static equivalents. This includes dynamic
typing as well as polymorphism. The approach taken in *numba* is using
*type inference* to generate *type information* for the code, so that it
is possible to translate into native code. If all the values in a
*numba* compiled function can be translated into native types, the
resulting code will be competitive with that generated with a low level
language.

The objective of *type inference* is assigning a *type* to every single
value in the function. The *type* of a value can either be:

-  *Implicit*, in the case of providing an object that will provide its
   *type*. This happens, for example, in literals.
-  *Explicit*, in the case of the programmer explicitly writing the
   *type* of a given value. This happens, for example, when a signature
   is given to *numba.jit*. That signature explicitly *types* the
   arguments.
-  *Inferred*, when the *type* is deduced from an operation and the
   types of the its operands. For example, inferring that the type of *a
   + b*, when *a* and *b* are of type *int* is going to be an *int*

*Type inference* is the process by which all the *types* that are
neither *implicit* nor *explicit* are deduced.

Type inference by example
-------------------------


Let's take a very simple sample function to illustrate these concepts:

.. code:: python

    def sample_func(n):
        tmp = n + 4;
        return tmp + 3j;

When translating to native code it is needed to provide *type
information* for every value involved in the sample function. This will
include:

-  The *literals* **4** and **3j**. These two have an implicit type.
-  The argument **n**. In the function, as is, it is yet untyped.
-  Some intermediate values, like **tmp** and the **return value**.
   Their type is not known yet.


Finding out the *types* of values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


You can use the function *numba.typeof* to find out the *numba type*
associated to a value.

.. code:: python

    print(numba.typeof.__doc__)

.. parsed-literal::

    
        Get the type of a variable or value.
    
        Used outside of Numba code, infers the type for the object.
        


Bear in mind that, when used from the *Python* interpreter,
*numba.typeof* will return the *numba type* associated to the object
passed as parameter. For example, let's try using it on the *literals*
found in our sample function:

.. code:: python

    numba.typeof(4)



.. parsed-literal::

    int32



.. code:: python

    numba.typeof(3j)



.. parsed-literal::

    complex128



Also note that the types of the results are *numba types*:

.. code:: python

    type(numba.typeof(4))



.. parsed-literal::

    numba.types.Integer



As a note, when used inside *numba* compiled code, *numba.typeof* will
return the type as inferred during *type inference*. This may be a more
general *type* than the one which would be returned when evaluating
using the *Python interpreter*.

*Type inference* in *numba.jit*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Let's illustrate how type inference works with *numba.jit*. In order to
illustrate this, we will use the *inspect\_types* method of a compiled
function and prints information about the types being used while
compiling. This will be the different native types when the function has
been compiled successfully in *nopython* mode. If object mode has been
used we will get plenty of *pyobject*\ s.

Note that *inspect\_types* is new to *numba 0.12*. Note also that the
behavior of object mode has changed quite a bit as well in this release.

.. code:: python

    def jit_sample_1(n):
        tmp = n + 4;
        return tmp + 3j;
.. code:: python

    numba.jit('c16(f8)', nopython=True)(jit_sample_1).inspect_types()

.. parsed-literal::

    jit_sample_1 (float64,) -> complex128
    --------------------------------------------------------------------------------
    # --- LINE 1 --- 
    
    def jit_sample_1(n):
    
        # --- LINE 2 --- 
        # label 0
        #   $0.1 = const(<type 'int'>, 4)  :: int32
        #   $0.2 = n + $0.1  :: float64
        #   tmp = $0.2  :: float64
    
        tmp = n + 4;
    
        # --- LINE 3 --- 
        #   $0.3 = const(<type 'complex'>, 3j)  :: complex128
        #   $0.4 = tmp + $0.3  :: complex128
        #   return $0.4
    
        return tmp + 3j;
    
    
    ================================================================================


The source code of the original function should be shown with lines
annotated with the values involved in that lines with its type annotated
following a couple of double periods. The form will look like "**value**
= **expression** :: **type**".

In this case, the resulting function will get a float64 argument and
return a complex128. The literal 4 will be of type int32 ($0.1), while
the result of adding the argument (n) to that literal will be a float64
($0.2). The variable in the source code named tmp will be just float64
(assigned from $0.2). In the same way we can trace the next expression
and see how **tmp+3j** results in a complex128 value that will be used
as return value. The values named \_$0.\*\_ are intermmediate values for
the expression, and do not have a named counterpart in the source code.

If we were in *object* mode we would get something quite different. In
order to illustrate, let's add the *forceobj* keyword to *numba.jit*.
This will force *numba* to use object mode when compiling. Usually you
don't want to use *forceobj* as *object* mode is slower than *nopython*
mode:

.. code:: python

    numba.jit('c16(f8)', forceobj=True)(jit_sample_1).inspect_types()

.. parsed-literal::

    jit_sample_1 (pyobject,) -> pyobject
    --------------------------------------------------------------------------------
    # --- LINE 1 --- 
    
    def jit_sample_1(n):
    
        # --- LINE 2 --- 
        # label 0
        #   $0.1 = const(<type 'int'>, 4)  :: pyobject
        #   tmp = n + $0.1  :: pyobject
    
        tmp = n + 4;
    
        # --- LINE 3 --- 
        #   $0.3 = const(<type 'complex'>, 3j)  :: pyobject
        #   $0.4 = tmp + $0.3  :: pyobject
        #   return $0.4
    
        return tmp + 3j;
    
    
    ================================================================================


As can be seen, everything is now a *pyobject*. That means that the
operations will be executed by the Python runtime in the generated code.

Going back to the *nopython* mode, we can see how changing the input
types will produced a different annotation for the code (and result in
different code generation):

.. code:: python

    numba.jit('c16(i1)')(jit_sample_1).inspect_types()

.. parsed-literal::

    jit_sample_1 (int8,) -> complex128
    --------------------------------------------------------------------------------
    # --- LINE 1 --- 
    
    def jit_sample_1(n):
    
        # --- LINE 2 --- 
        # label 0
        #   $0.1 = const(<type 'int'>, 4)  :: int32
        #   $0.2 = n + $0.1  :: int64
        #   tmp = $0.2  :: int64
    
        tmp = n + 4;
    
        # --- LINE 3 --- 
        #   $0.3 = const(<type 'complex'>, 3j)  :: complex128
        #   $0.4 = tmp + $0.3  :: complex128
        #   return $0.4
    
        return tmp + 3j;
    
    
    ================================================================================


In this case, the input is an int8, but tmp ends being and int64 as it
is added to an int32. Note that integer overflow of int64 is not handled
by *numba*. In case of overflow the int64 will wrap around in the same
way that it would happen in C.

Providing hints to the type inferrer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In most cases, the type inferrer will provide a type for your code.
However, sometimes you may want a given intermediate value to use a
specific type. This can be achieved by using the *locals* keyword in
*numba.jit*. In *locals* a dictionary can be passed that maps the name
of different local variables to a numba type. The compiler will assign
that type to that variable.

Let's make a version of out function where we force *tmp* to be a
*float*:

.. code:: python

    numba.jit('c16(i1)', locals={'tmp': numba.float64})(jit_sample_1).inspect_types()

.. parsed-literal::

    jit_sample_1 (int8,) -> complex128
    --------------------------------------------------------------------------------
    # --- LINE 1 --- 
    
    def jit_sample_1(n):
    
        # --- LINE 2 --- 
        # label 0
        #   $0.1 = const(<type 'int'>, 4)  :: int32
        #   $0.2 = n + $0.1  :: int64
        #   tmp = $0.2  :: float64
    
        tmp = n + 4;
    
        # --- LINE 3 --- 
        #   $0.3 = const(<type 'complex'>, 3j)  :: complex128
        #   $0.4 = tmp + $0.3  :: complex128
        #   return $0.4
    
        return tmp + 3j;
    
    
    ================================================================================


Note that as of numba 0.12, any type inference or type hints are ignored
if object mode ends being generated, as everything gets treated as an
object using the python runtime. This behavior may change in future
versions.

.. code:: python

    numba.jit('c16(i1)', forceobj=True, locals={ 'tmp': numba.float64 })(jit_sample_1).inspect_types()

.. parsed-literal::

    jit_sample_1 (pyobject,) -> pyobject
    --------------------------------------------------------------------------------
    # --- LINE 1 --- 
    
    def jit_sample_1(n):
    
        # --- LINE 2 --- 
        # label 0
        #   $0.1 = const(<type 'int'>, 4)  :: pyobject
        #   tmp = n + $0.1  :: pyobject
    
        tmp = n + 4;
    
        # --- LINE 3 --- 
        #   $0.3 = const(<type 'complex'>, 3j)  :: pyobject
        #   $0.4 = tmp + $0.3  :: pyobject
        #   return $0.4
    
        return tmp + 3j;
    
    
    ================================================================================


Importance of type inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


It must be emphasized how important it is type inference in *numba*. A
function where type inference is unable to provide a specific type for a
value (that is, any type other than the generic *pyobject*). Any
function that has a value fallback to *pyobject* will force the numba
compiler to use the object mode. Object mode is way less efficient thant
the *nopython*.

It is possible to know if a *numba* compiled function has fallen back to
object mode by calling *inspect\_types* on it. If there are values typed
as *pyobject* that means that the object mode was used to compile it.

Supported types in *numba*
--------------------------


Numba supports many different types. It also supports some composite
types as well as structures. Starting with numba 0.12 there is a
namespace for types (numba.types). The numba namespace also imports
these types.

In this section you can find a set of basic types you can use in numba.
Many of the types have a "short name" matching their equivalent NumPy
dtype. The list is not exahustive.

Integral types
~~~~~~~~~~~~~~


.. raw:: html

   <table><tr><th>

type

.. raw:: html

   </th><th>

numba type

.. raw:: html

   </th><th>

short name

.. raw:: html

   </th><th>

python equivalent

.. raw:: html

   </th></tr>
   <tr><td>

boolean

.. raw:: html

   </td><td>

numba.types.bool\_\_

.. raw:: html

   </td><td>

b1

.. raw:: html

   </td><td>

bool

.. raw:: html

   </td></tr>
   <tr><td>

signed integer

.. raw:: html

   </td><td>

numba.types.int\_\_

.. raw:: html

   </td><td></td><td>

int

.. raw:: html

   </td></tr>
   <tr><td>

signed integer (8 bit)

.. raw:: html

   </td><td>

numba.types.int8

.. raw:: html

   </td><td>

i1

.. raw:: html

   </td><td></td></tr>
   <tr><td>

signed integer (16 bit)

.. raw:: html

   </td><td>

numba.types.int16

.. raw:: html

   </td><td>

i2

.. raw:: html

   </td><td></td></tr>
   <tr><td>

signed integer (32 bit)

.. raw:: html

   </td><td>

numba.types.int32

.. raw:: html

   </td><td>

i4

.. raw:: html

   </td><td></td></tr>
   <tr><td>

signed integer (64 bit)

.. raw:: html

   </td><td>

numba.types.int64

.. raw:: html

   </td><td>

i8

.. raw:: html

   </td><td></td></tr>
   <tr><td>

unsigned integer

.. raw:: html

   </td><td>

numba.types.uint

.. raw:: html

   </td><td></td><td></td></tr>
   <tr><td>

unsigned integer (16 bit)

.. raw:: html

   </td><td>

numba.types.uint16

.. raw:: html

   </td><td>

u2

.. raw:: html

   </td><td></td></tr>
   <tr><td>

unsigned integer (32 bit)

.. raw:: html

   </td><td>

numba.types.uint32

.. raw:: html

   </td><td>

u4

.. raw:: html

   </td><td></td></tr>
   <tr><td>

unsigned integer (64 bit)

.. raw:: html

   </td><td>

numba.types.uint64

.. raw:: html

   </td><td>

u8

.. raw:: html

   </td><td></td></tr>
   </table>


Floating point types
~~~~~~~~~~~~~~~~~~~~


.. raw:: html

   <table><tr><th>

type

.. raw:: html

   </th><th>

numba type

.. raw:: html

   </th><th>

short name

.. raw:: html

   </th><th>

python equivalent

.. raw:: html

   </th></tr>
   <tr><td>

single precision floating point (32 bit)

.. raw:: html

   </td><td>

numba.float32

.. raw:: html

   </td><td>

f4

.. raw:: html

   </td><td></td></tr>
   <tr><td>

double precision floating point (64 bit)

.. raw:: html

   </td><td>

numba.float64

.. raw:: html

   </td><td>

f8

.. raw:: html

   </td><td>

float

.. raw:: html

   </td></tr>
   <tr><td>

single precision complex (2 x 32 bit)

.. raw:: html

   </td><td>

numba.complex64

.. raw:: html

   </td><td>

c8

.. raw:: html

   </td><td></td></tr>
   <tr><td>

double precison complex (2 x 64 bit)

.. raw:: html

   </td><td>

numba.complex128

.. raw:: html

   </td><td>

c16

.. raw:: html

   </td><td>

complex

.. raw:: html

   </td></tr>
   </table>


Array types
~~~~~~~~~~~


Array types are supported. An array type is built from a base type, a
number of dimensions and potentially a layout specification. Some
examples follow:

A one-dimensional array of float32

.. code:: python

    numba.types.float32[:]



.. parsed-literal::

    array(float32, 1d, A)



.. code:: python

    numba.typeof(np.zeros((12,2), dtype=np.float32)[:,0]) # slicing out the inner dimension to avoid defaulting to C array order in the result



.. parsed-literal::

    array(float32, 1d, A)



A two dimensional array of integers

.. code:: python

    numba.types.int_[:,:]



.. parsed-literal::

    array(uint32, 2d, A)



.. code:: python

    numba.typeof(np.zeros((12,2,2), dtype='i')[:,0]) # slicing out the inner dimension to avoid defaulting to C array order in the result



.. parsed-literal::

    array(int32, 2d, A)



A two dimensional array of type 'c8' (complex64) in C array order

.. code:: python

    numba.types.c8[:,::1]



.. parsed-literal::

    array(complex64, 2d, C)



.. code:: python

    numba.typeof(np.zeros((12,12), dtype='c8', order='C'))



.. parsed-literal::

    array(complex64, 2d, C)



A two dimensional array of type uint16 in FORTRAN array order

.. code:: python

    numba.types.uint16[::1,:]



.. parsed-literal::

    array(uint16, 2d, F)



.. code:: python

    numba.typeof(np.zeros((12,12), dtype='u2', order='F'))



.. parsed-literal::

    array(uint16, 2d, F)



Notice that the arity of the dimensions is not part of the types, only
the number of dimensions. In that sense, an array with a shape (4,4) has
the same numba type as another array with a shape (10, 12)

.. code:: python

    numba.typeof(np.zeros((4,4))) == numba.typeof(np.zeros((10,12)))



.. parsed-literal::

    True



Some extra types
~~~~~~~~~~~~~~~~


A type signature for a function (also known as a *function prototype*)
that returns a float64, taking a two dimensional float64 array as first
argument and a float64 argument

.. code:: python

    numba.types.float64(numba.types.float64[:,:], numba.types.float64)



.. parsed-literal::

    float64(array(float64, 2d, A), float64)



As can be seen the signature is just a type specification. In many
places that a *function signature* is expected a string can be used
instead. That string is in fact evaluated inside the numba.types
namespace in order to build the actual type. This allows specifying the
types in a compact way (as there is no need to fully qualify the base
types) without polluting the active namespace (as it would happen by
adding a \_\_from numba.types import \*\_\_.

In *numba* 0.12 this is performed by the
*numba.sigutils.parse\_signature* function. Note that this function is
likely to change or move in next versions, as it is just an
implementation detail, but it can be used to show how the string version
matches the other one, while keeping the syn

.. code:: python

    numba.sigutils.parse_signature('float64(float64[:,:], float64)')



.. parsed-literal::

    float64(array(float64, 2d, A), float64)



A generic Python object

.. code:: python

    numba.types.pyobject



.. parsed-literal::

    pyobject



Notes about changes in this tutorial
------------------------------------


In *numba* 0.12 there have been internal changes that have made material
previously found in this tutorial obsolete.

-  Some of the types previously supported in the *numba* type system
   have been dropped to be handled as *pyobjects*.

-  The numba command line tool is no longer supported, but its
   functionality to get insights on how type inference works is now
   present in the form of the *inspect\_types* method in the generated
   jitted function. This method is used in this tutorials to illustrate
   type inference.

-  In 0.12 the object mode of *numba* has been greatly modified. Before
   it was using a mix of Python run-time and native code. In 0.12 object
   mode forces all values into *pyobjects*. As conversion to a string
   forces *numba* into object mode, the approach used in the previous
   version of this tutorial to print from inside the compiled function
   is no longer useful, as it will not print the staticly inferred
   types.

A sample of the this last point follows:

.. code:: python

    def old_style_sample(n):
        print('arg n: '+ str(numba.typeof(n)))
        print('literal 4: ' + str(numba.typeof(4))) 
        tmp = n + 4;
        print('tmp: '+ str(numba.typeof(tmp)))
        print('literal 3j:' + str(numba.typeof(3j)))
        return tmp + 3j;
.. code:: python

    old_style_sample_jit = numba.jit('void(i1)')(old_style_sample)
.. code:: python

    numba.typeof(old_style_sample(42))

.. parsed-literal::

    arg n: int32
    literal 4: int32
    tmp: int32
    literal 3j:complex128




.. parsed-literal::

    complex128



.. code:: python

    numba.typeof(old_style_sample_jit(42))

.. parsed-literal::

    arg n: int32
    literal 4: int32
    tmp: int32
    literal 3j:complex128




.. parsed-literal::

    complex128



As can be seen, in both cases, Python and numba.jit, the results are the
same. This is because *numba.typeof* is being evaluated with using the
Python run-time.

If we use the inspect\_types method on the jitted version, we will see
that everything is in fact a *pyobject*

.. code:: python

    old_style_sample_jit.inspect_types()

.. parsed-literal::

    old_style_sample (pyobject,) -> pyobject
    --------------------------------------------------------------------------------
    # --- LINE 1 --- 
    
    def old_style_sample(n):
    
        # --- LINE 2 --- 
        # label 0
        #   $0.1 = global(print: <built-in function print>)  :: pyobject
        #   $0.2 = const(<type 'str'>, arg n: )  :: pyobject
        #   $0.3 = global(str: <type 'str'>)  :: pyobject
        #   $0.4 = global(numba: <module 'numba' from '/Users/jayvius/Projects/numba/numba/__init__.pyc'>)  :: pyobject
        #   $0.5 = getattr(attr=typeof, value=$0.4)  :: pyobject
        #   $0.6 = call $0.5(n, )  :: pyobject
        #   $0.7 = call $0.3($0.6, )  :: pyobject
        #   $0.8 = $0.2 + $0.7  :: pyobject
        #   $0.9 = call $0.1($0.8, )  :: pyobject
    
        print('arg n: '+ str(numba.typeof(n)))
    
        # --- LINE 3 --- 
        #   $0.10 = global(print: <built-in function print>)  :: pyobject
        #   $0.11 = const(<type 'str'>, literal 4: )  :: pyobject
        #   $0.12 = global(str: <type 'str'>)  :: pyobject
        #   $0.13 = global(numba: <module 'numba' from '/Users/jayvius/Projects/numba/numba/__init__.pyc'>)  :: pyobject
        #   $0.14 = getattr(attr=typeof, value=$0.13)  :: pyobject
        #   $0.15 = const(<type 'int'>, 4)  :: pyobject
        #   $0.16 = call $0.14($0.15, )  :: pyobject
        #   $0.17 = call $0.12($0.16, )  :: pyobject
        #   $0.18 = $0.11 + $0.17  :: pyobject
        #   $0.19 = call $0.10($0.18, )  :: pyobject
    
        print('literal 4: ' + str(numba.typeof(4)))
    
        # --- LINE 4 --- 
        #   $0.20 = const(<type 'int'>, 4)  :: pyobject
        #   tmp = n + $0.20  :: pyobject
    
        tmp = n + 4;
    
        # --- LINE 5 --- 
        #   $0.22 = global(print: <built-in function print>)  :: pyobject
        #   $0.23 = const(<type 'str'>, tmp: )  :: pyobject
        #   $0.24 = global(str: <type 'str'>)  :: pyobject
        #   $0.25 = global(numba: <module 'numba' from '/Users/jayvius/Projects/numba/numba/__init__.pyc'>)  :: pyobject
        #   $0.26 = getattr(attr=typeof, value=$0.25)  :: pyobject
        #   $0.27 = call $0.26(tmp, )  :: pyobject
        #   $0.28 = call $0.24($0.27, )  :: pyobject
        #   $0.29 = $0.23 + $0.28  :: pyobject
        #   $0.30 = call $0.22($0.29, )  :: pyobject
    
        print('tmp: '+ str(numba.typeof(tmp)))
    
        # --- LINE 6 --- 
        #   $0.31 = global(print: <built-in function print>)  :: pyobject
        #   $0.32 = const(<type 'str'>, literal 3j:)  :: pyobject
        #   $0.33 = global(str: <type 'str'>)  :: pyobject
        #   $0.34 = global(numba: <module 'numba' from '/Users/jayvius/Projects/numba/numba/__init__.pyc'>)  :: pyobject
        #   $0.35 = getattr(attr=typeof, value=$0.34)  :: pyobject
        #   $0.36 = const(<type 'complex'>, 3j)  :: pyobject
        #   $0.37 = call $0.35($0.36, )  :: pyobject
        #   $0.38 = call $0.33($0.37, )  :: pyobject
        #   $0.39 = $0.32 + $0.38  :: pyobject
        #   $0.40 = call $0.31($0.39, )  :: pyobject
    
        print('literal 3j:' + str(numba.typeof(3j)))
    
        # --- LINE 7 --- 
        #   $0.41 = const(<type 'complex'>, 3j)  :: pyobject
        #   $0.42 = tmp + $0.41  :: pyobject
        #   return $0.42
    
        return tmp + 3j;
    
    
    ================================================================================


Even more illustrating would be if *locals* was used to type an
intermediate value:

.. code:: python

    old_style_sample_jit_2 = numba.jit('void(i1)', locals={'tmp': numba.float32})(old_style_sample)
.. code:: python

    numba.typeof(old_style_sample_jit_2(42))

.. parsed-literal::

    arg n: int32
    literal 4: int32
    tmp: int32
    literal 3j:complex128




.. parsed-literal::

    complex128



The result seems to imply that *tmp* appears as an int32, but in fact is
a *pyobject* and the whole function is being evaluated using the python
run-time. So it is actually showing evaluating *typeof* at the runtime
on the run-time value of tmp, which happens to be a Python *int*,
translated into an int32 by *numba.typeof*. This can also be seen in the
dump caused by the call to inspect\_types.
