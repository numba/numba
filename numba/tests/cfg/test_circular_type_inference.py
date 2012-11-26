from numba.tests.cfg.test_cfg_type_infer import *

@autojit
def test_simple_circular():
    """
    >>> test_simple_circular()
    Warning 12:16: local variable 'y' might be referenced before assignment
    """
    x = 2.0
    for i in range(10):
        if i > 5:
            x = y
        else:
            y = x

@autojit
def test_simple_circular2():
    """
    >>> test_simple_circular2()
    Warning 27:16: local variable 'x' might be referenced before assignment
    """
    y = 2.0
    for i in range(10):
        if i > 5:
            x = y
        else:
            y = x


@autojit
def test_simple_circular3():
    """
    >>> test_simple_circular3()
    (Value(5), Value(5))
    >>> sig, syms = infer(test_simple_circular3.py_func,
    ...                   functype(None, []))
    >>> types(syms, 'x', 'y')
    (object_, object_)
    """
    x = values[5]
    y = 2.0
    for i in range(10):
        if i > 5:
            x = y
        else:
            y = x

    return x, y

@autojit
def test_simple_circular_promotion():
    """
    >>> test_simple_circular_promotion()
    ((3-3j), (1-3j))
    >>> sig, syms = infer(test_simple_circular_promotion.py_func,
    ...                   functype(None, []))
    >>> types(syms, 'x', 'y')
    (complex128, complex128)
    """
    x = 1
    y = 2
    for i in range(10):
        if i > 5:
            x = y + 2.0
        else:
            y = x - 3.0j

    return x, y

@autojit
def test_simple_circular_binop_promotion():
    """
    >>> test_simple_circular_binop_promotion()
    ((3-3j), (3+0j))
    >>> sig, syms = infer(test_simple_circular_binop_promotion.py_func,
    ...                   functype(None, []))
    >>> types(syms, 'x', 'y')
    (complex128, complex128)
    """
    x = 1
    y = 2
    for i in range(10):
        if i > 5:
            x = y - 3.0j
        else:
            y = x + 2.0 # In pure python, y would always be a float

    return x, y

#
### Test delayed inference for unops and binops
#
@autojit(warn=False)
def test_circular_binop():
    """
    >>> test_circular_binop()
    (1.0, 2.0, 1.0, -3L)
    >>> sig, syms = infer(test_circular_binop.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'x', 'y', 'z', 'a')
    (double, double, double, int)
    """
    x = 1
    y = 2
    for i in range(10):
        if i > 5:
            x = y - z
            z = 1.0
        else:
            z = int(x + y)
            y = x + z - y
            a = -z

    return x, y, z, a

@autojit(warn=False)
def test_delayed_indexing():
    """
    >>> test_delayed_indexing()
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> sig, syms = infer(test_delayed_indexing.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'array', 'x')
    (double[:], int)
    """
    array = np.ones(10, dtype=np.double)
    x = 0
    for i in range(11):
        var = array[x]
        array[x] = var * x
        x = int(i * 1.0)

    return array

@autojit
def test_circular_error():
    """
    >>> try: test_circular_error()
    ... except error.NumbaError, e: print e
    Warning 61:16: local variable 'y' might be referenced before assignment
    Warning 63:16: local variable 'x' might be referenced before assignment
    61:12: Unable to infer type for assignment to x, insert a cast or initialize the variable.
    """
    for i in range(10):
        if i > 5:
            x = y
        else:
            y = x

#test_delayed_indexing()
testmod()