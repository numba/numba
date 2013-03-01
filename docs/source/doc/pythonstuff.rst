********************
Python Functionality
********************

Extension Types (Classes)
=========================
Numba supports classes similar to Python classes and extension types.
Currently the methods have static signatures, but in the future they
we will likely support multiple inheritance and autojitting methods.

One can refer to the extension type as a type by accessing its ``exttype``
attribute::

    @jit
    class MyExtension(object):
        ...

    @jit(MyExtension.exttype(double))
    def create_ext(arg):
        return MyExtension(arg)

It is not yet possible to refer to the extension type in the class body or
methods of that extension type.

An example of extension classes and their capabilities and limitations
is shown below:

.. literalinclude:: /../../examples/numbaclasses.py

Closures
========

Numba supports closures (nested functions), and keeps the variable scopes
alive for the lifetime of the closures.
Variables that are closed over by the closures (``cell variables``) have
one consistent type throughout the entirety of the function. This means
differently typed variables can only be assigned if they are unifyable,
such as for instance ints and floats::

    @autojit
    def outer(arg1, arg2):
        arg1 = 0
        arg1 = 0.0      # This is fine
        arg1 = "hello"  # ERROR! arg1 must have a single type

        arg2 = 0
        arg2 = "hello"  # FINE! Not a cell variable

        def inner():
            print arg1

        return inner

Calling an inner function directly in the body of ``outer`` will result in
a direct, native call of the closure. In the future it is likely that passing
around the closure will still result in a native call in other places.

Like Python closures, closures can be arbitrarily nested, and follow the same
scoping rules.

