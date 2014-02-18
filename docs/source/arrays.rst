******
Arrays
******

Support for NumPy arrays is a key focus of Numba development and is currently
undergoing extensive refactorization and improvement. Most capabilities of NumPy
arrays are supported by Numba in object mode, and a few features are supported in
nopython mode too (with much more to come).

A few noteworthy limitations of arrays at this time:

* Arrays can be passed in to a function in nopython mode, but not returned.
  Arrays can only be returned in object mode.
* New arrays can only be created in object mode.
* Currently there are no bounds checking for array indexing and slicing.
* NumPy array ufunc support in nopython node is incomplete at this time. Most
  if not all ufuncs should work in object mode though.
* Array slicing only works in object mode.

Array Creation
--------------

NumPy array creation is not supported in nopython mode. Numba mitigates this by
automatically trying to "lift" loops out of an object mode function and compile
them in nopython mode. This allows for array creation at the top of a function
while still getting almost all the performance of nopython mode. For example,
the following simple function::

    # compiled in object mode
    @jit
    def sum(x, y):
        array = np.arange(x * y).reshape(x, y)
        sum = 0
        for i in range(x):
            for j in range(y):
                sum += array[i, j]
        return sum

looks like the equivalent of the following after being compiled by Numba::

    # compiled in nopython mode
    @njit
    def lifted_loop(array, x, y):
        sum = 0
        for i in range(x):
            for j in range(y):
                sum += array[i, j]
        return sum

    # compiled in object mode
    @jit
    def sum(x, y):
        array = np.arange(x * y).reshape(x, y)
        return lifted_loop(array, x, y)

Another consequence of array creation being restricted to object mode is that 
NumPy ufuncs that return the result as a new array are not allowed in nopython
mode. Fortunately we can declare an output array at the top of our function and
pass that in to the ufunc to store our result. For example, the following::

    @jit
    def foo():
        # initialize stuff

        # slow loop in object mode
        for i in range(x):
            result = np.multiply(a, b)

should be rewritten like the following to take advantage of loop lifting::

    @jit
    def foo():
        # initialize stuff

        # create output array
        result = np.zeros(a.size)

        # loop now compiled in nopython mode
        for i in range(x):
            np.multiply(a, b, result)


