*******************
Classes
*******************
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
