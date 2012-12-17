import numpy as np

import numba
from numba import *
from numba import _numba_types as numba_types

from numba.tests.test_support import *
from numba.tests.cfg.test_cfg_type_infer import infer as _infer, types, functype

T = numba.template()

@autojit(T(T[:, :]), warn=False, locals=dict(scalar=T))
def test_simple_template(array):
    """
    >>> test_simple_template(np.arange(10, 12, dtype=np.float32))
    10.0

    #------------------------------------------------------------------------
    # Test type resolving
    #------------------------------------------------------------------------
    >>> infer(test_simple_template.py_func, double(double[:, :]), T(T[:, :]),
    ...       locals=dict(scalar=T))
    [('array', double[:, :]), ('scalar', double)]

    >>> infer(test_simple_template.py_func, double(double[:, :]), T(T[:, :]),
    ...       locals=dict(scalar=T.pointer()))
    Traceback (most recent call last):
        ...
    UnpromotableTypeError: (double *, double)

    #------------------------------------------------------------------------
    # Test type attributes
    #------------------------------------------------------------------------
    >>> infer(test_simple_template.py_func, double(double[:, :]), T.dtype(T),
    ...       locals=dict(scalar=T.dtype))
    [('array', double[:, :]), ('scalar', double)]
    """
    scalar = array[0, 0]
    return scalar

#------------------------------------------------------------------------
# Test type matching
#------------------------------------------------------------------------

T1 = numba.template("T1")
T2 = numba.template("T2")
T3 = numba.template("T3")
T4 = numba.template("T4")

A = T1[:, :]
F = void(T1)
S = numba.struct(a=T1, b=T2.pointer(), c=T3[:], d=void(T4))
P = T2.pointer()

type_context = { T1: int_, T2: float_, T3: double, T4: short, }

def test_type_matching(array, func, struct, pointer):
    """
    >>> infer(test_type_matching, template_signature=void(A, F, S, P),
    ...       type_context=type_context)
    [('array', int[:, :]), ('func', void (*)(int)), ('pointer', float *), ('struct', struct { float * b, double[:] c, int a, void (*)(short) d })]
    """
    func(array[0, 0])
    struct.b = pointer


#------------------------------------------------------------------------
# Test utilities
#------------------------------------------------------------------------

def infer(func, signature=None, template_signature=None,
          locals=None, type_context=None):

    if signature is None:
        signature = specialize(template_signature, type_context)

    if locals is not None:
        locals = dict(locals)

    sig, symbols = _infer(func, signature,
                          template_signature=template_signature,
                          locals=locals)

    if locals is not None:
        local_vars = sorted(locals.iteritems())
    else:
        local_vars = []

    vars = sorted((name, var.type) for name, var in symbols.iteritems())
    return vars + local_vars

def specialize(T, context):
    return numba_types.resolve_template_type(T, context)


if __name__ == '__main__':
    test_simple_template(np.arange(10, 12, dtype=np.float32))
    infer(test_type_matching, template_signature=void(A, F, S, P),
          type_context=type_context)

testmod()