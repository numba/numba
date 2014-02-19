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
* Currently there are no bounds checking for array indexing and slicing,
  although negative indices will wrap around correctly.
* NumPy array ufunc support in nopython node is incomplete at this time. Most
  if not all ufuncs should work in object mode though.
* Array slicing only works in object mode.

Array Creation & Loop-Jitting
------------------------------

NumPy array creation is not supported in nopython mode. Numba mitigates this by
automatically trying to jit loops in nopython mode. This allows for array
creation at the top of a function while still getting almost all the performance
of nopython mode. For example, the following simple function::

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
    def jitted_loop(array, x, y):
        sum = 0
        for i in range(x):
            for j in range(y):
                sum += array[i, j]
        return sum

    # compiled in object mode
    @jit
    def sum(x, y):
        array = np.arange(x * y).reshape(x, y)
        return jitted_loop(array, x, y)

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

should be rewritten like the following to take advantage of loop jitting::

    @jit
    def foo():
        # initialize stuff

        # create output array
        result = np.zeros(a.size)

        # jitted loop in nopython mode
        for i in range(x):
            np.multiply(a, b, result)


Loop Jitting Constraints
-------------------------

The current loop-jitting mechanism is very conservative.  A loop must satisfy
a set of constraints for loop-jitting to trigger. These constraints will be
relaxed in further development.

Currently, a loop is rejected if:

* the loop contains return statements;
* the loop binds a value to a variable that is read outside of the loop.

The following is rejected due to a return statement in the loop::

    @jit
    def foo(n):
        result = np.zeros(n)

        # Rejected loop-jitting candidate
        for i in range(n):
            result[i] = i   # setitem is accepted
            if i > 10:
                return      # return is not accepted

        return result

The following is rejected due to an assigning to a variable read outside of
the loop::

    @jit
    def foo(n):
        result = np.zeros(n)

        x = 1
        # Rejected loop-jitting candidate
        for i in range(n):
            x = result[i]           # assign to variable 'x'

        result += x                 # reading variable 'x'
        return result


The following is accepted::

    @jit
    def foo(n):
        result = np.zeros(n)
        x = 1
        # Accepted loop-jitting candidate
        for i in range(n):
            x = 2
        x = 3               # 'x' is only written to

        return result


The following is accepted::

    @jit
    def foo(n):
        result = np.zeros(n)
        x = 1
        # Accepted loop-jitting candidate
        for i in range(n):
            result[i] = x

        return result

User can inspect the loop-jitting by running `foo.inspect_types()`::

    foo (int32,) -> pyobject
    --------------------------------------------------------------------------------
    # File: somefile.py
    # --- LINE 1 ---

    @jit

    # --- LINE 2 ---

    def foo(n):

        # --- LINE 3 ---
        # label 0
        #   $0.1 = global(numpy: <module 'numpy' from '.../numpy/__init__.py'>)
          :: pyobject
        #   $0.2 = getattr(value=$0.1, attr=zeros)  :: pyobject
        #   result = call $0.2(n, )  :: pyobject

        result = numpy.zeros(n)

        # --- LINE 4 ---
        #   x = const(<class 'int'>, 1)  :: pyobject

        x = 1

        # --- LINE 5 ---
        #   jump 58
        # label 58
        #   $58.1 = global(foo__numba__loop21__: LiftedLoop(<function foo at 0x107781710>))  :: pyobject
        #   $58.2 = call $58.1(n, result, x, )  :: pyobject
        #   jump 54

        for i in range(n):

            # --- LINE 6 ---

            result[i] = x

        # --- LINE 7 ---
        # label 54
        #   return result

        return result

    # The function contains lifted loops
    # Loop at line 5
    # Has 1 overloads
    # File: somefile.py
    # --- LINE 1 ---

    @jit

    # --- LINE 2 ---

    def foo(n):

        # --- LINE 3 ---

        result = numpy.zeros(n)

        # --- LINE 4 ---

        x = 1

        # --- LINE 5 ---
        # label 34
        #   $34.1 = iternext(value=$21.3)  :: int32
        #   $34.2 = itervalid(value=$21.3)  :: bool
        #   branch $34.2, 37, 53
        # label 21
        #   $21.1 = global(range: <class 'range'>)  :: range
        #   $21.2 = call $21.1(n, )  :: (int32,) -> range_state32
        #   $21.3 = getiter(value=$21.2)  :: range_iter32
        #   jump 34
        # label 37
        #   $37.1 = $34.1  :: int32
        #   i = $37.1  :: int32

        for i in range(n):

            # --- LINE 6 ---
            # label 53
            #   del $21.3
            #   jump 54
            # label 54
            #   $54.1 = const(<class 'NoneType'>, None)  :: none
            #   return $54.1
            #   result[i] = x  :: (array(float64, 1d, C), int64, float64) -> none
            #   jump 34

            result[i] = x

        # --- LINE 7 ---

        return result



    ================================================================================

