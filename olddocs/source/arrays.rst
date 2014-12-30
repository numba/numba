******
Arrays
******

Support for NumPy arrays is a key focus of Numba development and is currently
undergoing extensive refactorization and improvement. Most capabilities of
NumPy arrays are supported by Numba in :term:`object mode`, and a few features
are supported in :term:`nopython mode` too (with much more to come).

A few noteworthy limitations of arrays at this time:

* Arrays can be passed in to a function in nopython mode, but not returned.
  Arrays can only be returned in object mode.
* New arrays can only be created in object mode.
* Currently there are no bounds checking for array indexing and slicing,
  although negative indices will wrap around correctly.
* A small number of NumPy array ufuncs are only supported in object mode, but 
  the vast majority work in nopython mode.
* Array slicing only works in object mode.

Array Creation & Loop-Jitting
------------------------------

NumPy array creation is not supported in nopython mode. Numba mitigates this by
automatically trying to JIT loops in nopython mode. This allows for array
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

    # force compilation in nopython mode
    @jit(nopython=True)
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

Loop-jitting will not be used by the compiler in this case because of the
return statement in the loop::

    @jit
    def foo(n):
        result = np.zeros(n)

        # Rejected loop-jitting candidate
        for i in range(n):
            result[i] = i   # setitem is accepted
            if i > 10:
                return      # return is not accepted

        return result

User can inspect the loop-jitting by running `foo.inspect_types()`.  For
example, this::

    @jit
    def foo(n):
        result = np.arange(n)

        x = 1
        for i in range(n):
            x = result[i]           # assign to variable 'x'

        result += x                 # reading variable 'x'
        return result


    foo(10) # trigger autojit
    foo.inspect_types()

prints the following output, indicating a lifted loop::

    foo (int64,)
    --------------------------------------------------------------------------------
    # File: <ipython-input-11-ffbcb34ecc02>
    # --- LINE 1 --- 

    @jit

    # --- LINE 2 --- 

    def foo(n):

    # --- LINE 3 --- 
    # label 0
    #   n.1 = n  :: pyobject
    #   del n
    #   $0.1 = global(np: <module 'numpy' from '/Users/stan/anaconda/envs/numba_dev/lib/python2.7/site-packages/numpy/__init__.pyc'>)  :: pyobject
    #   $0.2 = getattr(attr=arange, value=$0.1)  :: pyobject
    #   del $0.1
    #   $0.4 = call $0.2(n.1, )  :: pyobject
    #   del $0.2
    #   result = $0.4  :: pyobject
    #   del $0.4

    result = np.arange(n)

    # --- LINE 4 --- 



    # --- LINE 5 --- 
    #   $const0.5 = const(int, 1)  :: pyobject
    #   x = $const0.5  :: pyobject
    #   del x
    #   del $const0.5

    x = 1

    # --- LINE 6 --- 
    #   jump 24.1
    # label 24.1
    #   $const24.1.1 = const(LiftedLoop, LiftedLoop(<function foo at 0x10ae43410>))  :: pyobject
    #   $24.1.4 = call $const24.1.1(result, n.1, )  :: pyobject
    #   del n.1
    #   del $const24.1.1
    #   $24.1.6 = exhaust_iter(count=1, value=$24.1.4)  :: pyobject
    #   del $24.1.4
    #   $24.1.5 = static_getitem(index=0, value=$24.1.6)  :: pyobject
    #   del $24.1.6
    #   x.1 = $24.1.5  :: pyobject
    #   del $24.1.5
    #   jump 54

    for i in range(n):

        # --- LINE 7 --- 

        x = result[i]           # assign to variable 'x'

    # --- LINE 8 --- 



    # --- LINE 9 --- 
    # label 54
    #   $54.3 = inplace_binop(rhs=x.1, lhs=result, fn=+)  :: pyobject
    #   del x.1
    #   del result
    #   result.1 = $54.3  :: pyobject
    #   del $54.3

    result += x                 # reading variable 'x'

    # --- LINE 10 --- 
    #   $54.5 = cast(value=result.1)  :: pyobject
    #   del result.1
    #   return $54.5

    return result

    # The function contains lifted loops
    # Loop at line 6
    # Has 1 overloads
    # File: <ipython-input-11-ffbcb34ecc02>
    # --- LINE 1 --- 

    @jit

    # --- LINE 2 --- 

    def foo(n):

    # --- LINE 3 --- 

    result = np.arange(n)

    # --- LINE 4 --- 



    # --- LINE 5 --- 

    x = 1

    # --- LINE 6 --- 
    # label 34
    #   $34.2 = iternext(value=$phi34.1)  :: pair<int64, bool>
    #   $34.3 = pair_first(value=$34.2)  :: int64
    #   $34.4 = pair_second(value=$34.2)  :: bool
    #   del $34.2
    #   $phi37.1 = $34.3  :: int64
    #   del $34.3
    #   branch $34.4, 37, 53
    # label 21
    #   result.1 = result  :: array(int64, 1d, C, nonconst)
    #   del result
    #   n.1 = n  :: int64
    #   del n
    #   $21.1 = global(range: <built-in function range>)  :: range
    #   $21.3 = call $21.1(n.1, )  :: (int64,) -> range_state64
    #   del n.1
    #   del $21.1
    #   $21.4 = getiter(value=$21.3)  :: range_iter64
    #   del $21.3
    #   $phi34.1 = $21.4  :: range_iter64
    #   del $21.4
    #   jump 34
    # label 37
    #   i = $phi37.1  :: int64
    #   del $phi37.1

    for i in range(n):

        # --- LINE 7 --- 
        # label 53
        #   del result.1
        #   del $phi37.1
        #   del $phi34.1
        #   del $34.4
        #   jump 54
        # label 54
        #   $54.2 = build_tuple(items=[Var(x, <ipython-input-11-ffbcb34ecc02> (7))])  :: (int64 x 1)
        #   del x
        #   $54.3 = cast(value=$54.2)  :: (int64 x 1)
        #   del $54.2
        #   return $54.3
        #   $37.4 = getitem(index=i, value=result.1)  :: int64
        #   del i
        #   x = $37.4  :: int64
        #   del $37.4
        #   jump 34

        x = result[i]           # assign to variable 'x'

    # --- LINE 8 --- 



    # --- LINE 9 --- 

    result += x                 # reading variable 'x'

    # --- LINE 10 --- 

    return result


    ================================================================================

