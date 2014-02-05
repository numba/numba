********************
Numba Debugging Tips
********************

The most common problem users run into is slower than expected performance.
Usually this is because Numba is having trouble inferring a type or doesn't
have a compiled version of a function to call. An easy way to pinpoint the
source of this problem is to use the nopython flag. By default, Numba tries
to compile everything down to low level types, but if it runs into trouble it
will fall back to using Python objects. Setting nopython=True in the jit
decorator will tell Numba to never use objects, and instead bail out if it
runs into trouble::

    @jit(nopython=True)
    def foo()
        ...

which will result in an error if Numba can't figure out the type of a variable
or function.

Another more advanced method for figuring out what's going on is using the
inspect_types method. Calling inspect_types for a function compiled with Numba
like so::
    
    @jit
    def foo(x):
        return np.sin(x)

    foo.inspect_types()

will result in output similar to the following::

    foo (pyobject,) -> pyobject

    ---------------------------

    # --- LINE 25 ---

    @jit(nopython=False)

    # --- LINE 26 ---

    def foo(x):

        # --- LINE 27 --- 
        # label 0
        #   $0.1 = global(np: <module 'numpy' from '/Users/jayvius/anaconda/envs/dev/lib/python2.7/site-packages/numpy/__init__.pyc'>)  :: pyobject
        #   $0.2 = getattr(attr=sin, value=$0.1)  :: pyobject
        #   $0.3 = call $0.2(x, )  :: pyobject
        #   return $0.3

        return np.sin(x)


A few things to take note of here: First, every line of Python code is preceded
by several lines of Numba IR code that gives a glimpse into what Numba is doing
to your Python code behind the scenes. More helpful though, at the end of most
lines there are type annotations that show how Numba is treating variables and
function calls. In the example above, the 'pyobject' annotation indicates that
Numba doesn't know about the np.sin function so it has to fall back to the
Python object layer to call it.
