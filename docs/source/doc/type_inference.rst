.. _type_inference:

***************************************
Adding Type Inference for External Code
***************************************

Users can add type inference for non-numba functions, to specify to
numba what the result of an operation will look like. This can make
numba code much more efficient by knowing the lower-level types.

Type Functions
--------------

Users can write type functions to tell Numba what the return type of
a certain function call in numba code will be. The return type may
be always the same, or it may depend on the input types in a simple
or complicated way.

Below we will describe ways to handle all these cases.

Simple Signatures
+++++++++++++++++

Simple signatures can be easily handled by registering a callable with
a static signature::

    from numba import register_callable
    from numba import *

    @register_callable(double(double))
    def square(arg):
        return arg * 2

This specifies that the function takes only a double and returns a double.

If a function is statically polymorphic, a typeset can be used to
represent a variety of inputs::

    import numba as nb
    from numba import *

    @nb.register_callable(nb.numeric(nb.numeric))
    def square(...):
        ...

Here ``numba.numeric`` represents the set of numeric types. The set is
then bound in a similar way to Cython's fused types, in that the argument
types instantiate the sets to concrete types. This means the signature above
allows signatures like ``double(double)`` and ``int_(int_)``, but not for
instance ``double(int)``.

Type sets can be created explicitly::

    signatures = numba.typeset(object_(object_), numeric(numeric))

    @numba.register_callable(signatures)
    def square(...):
        ...

Parametric Polymorphism
-----------------------

Type sets are fine for small sets, but not general enough for a large
or infinite type universe. Instead, we allow parametric polymorphism through
simple templates and type functions (functions that return the type at compile
time given the input types). For an introduction to templates
we refer the reader to templates_.

Templates can be used as follows::

    # create a type variable that will be bound by an argument type
    a = numba.template("a") 

    @numba.register_callable(a(a[:]))
    def sum_1d(...):
        ...

More general type expressions are listed under templates_.

.. NOTE:: Templates for user-based type inference is not yet implemented.

For full control a user can write an explicit function that performs the type
inference based on the input types. The input types are passed in as objects
and inferred from the function's signature::

    from numba import typesystem
    from numba.typesystem import get_type

    def infer_reduce(a, axis, dtype, out):
        if out is not None:
            return out
    
        dtype_type = get_dtype(a, dtype, static_dtype)
    
        if axis is None:
            # Return the scalar type
            return dtype_type
    
        if dtype_type:
            # Handle the axis parameter
            if axis.is_tuple and axis.is_sized:
                # axis=(tuple with a constant size)
                return typesystem.array(dtype_type, a.ndim - axis.size)
            elif axis.is_int:
                # axis=1
                return typesystem.array(dtype_type, a.ndim - 1)
            else:
                # axis=(something unknown)
                return object_
   
    register_inferer(np, 'sum', infer_reduce)    # Register type inference for np.sum
    register_inferer(np, 'prod', infer_reduce)   # Register type inference for np.prod

The above works through introspection of the function using the ``inspect`` module. A call
in the user code to ``np.sum`` or ``np.prod`` will now ask the above function to resolve its
type at Numba compile time, passing in the types representing the arguments, or None
when absent::

    # Numba code
    np.sum(a)              # => infer_reduce(a=double[:, :]), axis=None, dtype=None, out=None)
    np.sum(a, axis=(1, 2)) # => infer_reduce(a=double[:, :, :]), axis=tuple(base_type=int_, size=2),
                           #                 dtype=None, out=None)

A shorthand function to register type functions is provided by ``numba.register``::

    @numba.register(np)
    def sum(a, axis, dtype, out):
        ...

This retrieves ``np.sum`` based on the name of the type inferring function (hence it must be called
``sum``).

Registering Unbound Methods
---------------------------

Unbound methods are transient, and hence can not be registered by value. Instead we register
a dotted path starting at a value, e.g.::

    numba.register_unbound(np, "add", "reduce", infer_reduce)

To allow type inference for ``np.add.reduce()``. The first string specifies the module (``np``), the
second the object (``"add"``), the third the dotted path (``"reduce"``) and the last the type
function (``infer_reduce``).

Future Directions
-----------------

The code above is clearly very verbose, which is partly due to the generality of the ``sum``
signature. In the future we hope to expose a more declarative way to specify parametrically
polymorphic signatures. Perhaps something like::

    sum(a)                                              => a
    sum(array(dtype, ndim), axis=integral)              => array(dtype, ndim - 1)
    sum(array(dtype, ndim), axis=tuple(integral, size)) => array(dtype, ndim - size)
    sum(in, axis=axis, out=out)                         => sum(out, axis=axis)
    sum(_, _)                                           => object_

