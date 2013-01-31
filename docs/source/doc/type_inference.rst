.. _type_inference:

***************************************
Adding Type Inference for External Code
***************************************

Users can add type inference for non-numba functions, to specify to
numba what the result of an operation will look like.

Static Signatures
-----------------

Static signatures can be easily handled by registering a callable with
a static signature::

    from numba import register_callable
    from numba import *

    @register_callable(double(double))
    def square(arg):
        return arg * 2

Or if your function is statically polymorphic, a typeset can be used to
represent a variety of inputs::

    import numba as nb
    from numba import *

    @nb.register_callable(nb.numeric(nb.numeric))
    def square(...):
        ...

Here ``numba.numeric`` represents the set of numeric types. The set is
then bound in a similar way to Cython's fused types, in that the argument
types instantiate the set to a concrete type. This means the signature above
allows signatures like ``double(double)`` but not for instance ``double(int)``.

Type sets can be created explicitly::

    signatures = numba.typeset(object_(object_), numeric(numeric))

    @numba.register_callable(signatures)
    def square(...):
        ...

Parametric Polymorphism
-----------------------

Type sets are fine for small typesets, but not general enough for a large
or infinite type universe. Instead, we allow parametric polymorphism through
simple templates and type infering callbacks. For an introduction to templates
we refer the reader to templates_.

Templates can be used as follows::

    # create a type variable that will be bound by an argument type
    a = numba.template("a") 

    @numba.register_callable(a(a[:]))
    def sum_1d(...):
        ...

More general type expressions are listed under templates_.

.. NOTE:: Type sets and templates for user exposed type inference is not yet implemented.

For full control a user can write an explicit function that performs the type
inference based on the input types::

    from numba import typesystem
    from numba.typesystem import get_type
    from numba.type_inference.modules import numpymodule

    def get_dtype(dtype, default_dtype):
        # Simple helper function to map an AST node dtype keyword argument => NumPy dtype
        ...

    def reduce_(a, axis, dtype, out):
        if out is not None:
            return get_type(out)
    
        # Get the type of the array argument AST node
        array_type = get_type(a)
        
        # Get the NumpyDtypeType that represents the type for a np.dtype(...) object 
        # The 'dtype' attribute of the type gives the concrete dtype type, i.e. for
        # an AST node `dt` representing `np.dtype(np.double)`, `get_dtype(dt).dtype` will return
        # the Numba type `double`
        dtype_type = numpymodule.get_dtype(dtype, default_dtype=array_type.dtype)
    
        if axis is None:
            # Return the scalar type
            return dtype_type.dtype
    
        # Handle the axis parameter
        axis_type = get_type(axis)
        if axis_type.is_tuple and axis_type.is_sized:
            # axis=(tuple with a constant size)
            return typesystem.array(dtype_type.dtype, array_type.ndim - axis_type.size)
        elif axis_type.is_int:
            # axis=1
            return typesystem.array(dtype_type.dtype, array_type.ndim - 1)
        else:
            # axis=(something unknown)
            return object_
    
    register_inferer(np, 'sum', reduce_)    # Register type inference for np.sum
    register_inferer(np, 'prod', reduce_)   # Register type inference for np.prod

The above works through introspection of the function using the ``inspect`` module. A call
in the user code to ``np.sum`` or ``np.prod`` will now ask this function to resolve its
type at Numba compile time, passing in the AST nodes representing the arguments, or None
when abscent::

    # Numba code
    np.sum(a)              # => reduce_(a=ast.Name(id='a', ctx=ast.Load(), type=double[:, :]),
                           #            axis=None, dtype=None, out=None)
    np.sum(a, axis=(1, 2)) # => reduce_(a=ast.Name(id='a', ctx=ast.Load(), type=double[:, :, :]),
                           #            axis=tuple(base_type=int_, size=2), dtype=None, out=None)

A shorthand function to register type inferering functions is provided by ``numba.register``::

    @numba.register(np)
    def sum(a, axis, dtype, out):
        ...

This retrieves ``np.sum`` based on the name of the type inferring function (hence it must be called
``sum``).

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

