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

#
### Test delayed inference for comparisons and boolops
#
@autojit(warn=False)
def test_circular_compare():
    """
    >>> test_circular_compare()
    (5.0, 1.0)
    >>> sig, syms = infer(test_circular_compare.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'x', 'y')
    (double, double)
    """
    x = 1
    for i in range(10):
        if i == 0:
            y = float(x)
        if x < 5:
            x += y

    return x, y

@autojit(warn=False)
def test_circular_compare2():
    """
    >>> test_circular_compare2()
    (2.0, 1.0)
    >>> sig, syms = infer(test_circular_compare.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'x', 'y')
    (double, double)
    """
    x = 1
    for i in range(10):
        if i == 0:
            y = float(x)
        if x < 5 and (x > 2 or i == 0):
            x += y

    return x, y

@autojit(warn=False)
def test_circular_compare3():
    """
    >>> test_circular_compare3()
    1
    2
    3
    4
    (False, 10L)
    >>> sig, syms = infer(test_circular_compare3.py_func,
    ...                   functype(None, []), warn=False)
    >>> map(str, types(syms, 'cond', 'x'))
    ['bool', 'Py_ssize_t']
    """
    x = 1
    cond = True
    for i in range(10):
        if cond:
            x = i
        else:
            x = i + 1
        cond = x > 1 and x < 5
        if cond:
            x = cond or x < i
            cond = x
            x = i
            print i

    return cond, x

#
### Test indexing
#
@autojit(warn=False)
def test_delayed_array_indexing():
    """
    >>> test_delayed_array_indexing()
    (array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]), 1.0, 10L)
    >>> sig, syms = infer(test_delayed_array_indexing.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'array', 'var', 'x')
    (double[:], double, int)
    """
    array = np.ones(10, dtype=np.double)
    x = 0
    for i in range(11):
        var = array[x]
        array[x] = var * x
        x = int(i * 1.0)

    return array, var, x

@autojit(warn=False)
def test_delayed_array_slicing():
    """
    >>> test_delayed_array_slicing()
    [[ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  2.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  3.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  4.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  5.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  6.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  7.  1.  1.]]
    [ 1.  1.  1.  1.  1.  1.  1.  7.  1.  1.]
    >>> sig, syms = infer(test_delayed_array_slicing.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'array', 'row')
    (double[:, :], double[:])
    """
    array = np.ones((8, 10), dtype=np.double)
    for i in range(8):
        row = array[i, :]
        array[i, i] = row[i] * i
        array = array[:, :]

    print array
    print row


@autojit(warn=False)
def test_delayed_array_slicing2():
    """
    >>> test_delayed_array_slicing2()
    [[ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  2.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  3.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  4.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  5.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  6.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  7.  1.  1.]]
    [ 1.  1.  1.  1.  1.  1.  1.  7.  1.  1.]
    >>> sig, syms = infer(test_delayed_array_slicing.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 'array', 'row')
    (double[:, :], double[:])
    """
    for i in range(8):
        if i == 0:
            array = np.ones((8, 10), dtype=np.double)

        row = array[i, :]
        array[i, i] = row[i] * i
        array = array[:, :]

    print array
    print row

@autojit(warn=False)
def test_delayed_string_indexing():
    """
    >>> test_delayed_string_indexing()
    ('ham eggs', 3L)
    >>> sig, syms = infer(test_delayed_string_indexing.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 's', 'x')
    (const char *, Py_ssize_t)
    """
    s = "spam ham eggs"
    for i in range(4):
        if i < 3:
            x = i
            tmp1 = s[x:]
            tmp2 = tmp1
            s = tmp2
        elif i < 5:
            s = tmp1[x:]
        else:
            s = "hello"

        x = i

    return s, x

@autojit(warn=False)
def test_delayed_string_indexing2():
    """
    >>> test_delayed_string_indexing2()
    ('ham eggs', 3L)
    >>> sig, syms = infer(test_delayed_string_indexing2.py_func,
    ...                   functype(None, []), warn=False)
    >>> types(syms, 's', 'x')
    (const char *, Py_ssize_t)
    """
    for i in range(4):
        if i == 0:
            s = "spam ham eggs"

        if i < 3:
            x = i
            tmp1 = s[x:]
            tmp2 = tmp1
            s = tmp2
        elif i < 5:
            s = tmp1[x:]
        else:
            s = "hello"

        x = i

    return s, x

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

#testmod()
test_delayed_string_indexing()