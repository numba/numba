from pprint import pprint

import numpy as np

import numba
from numba import *

from numba.testing.test_support import *
from numba.control_flow.tests.test_cfg_type_infer import infer as _infer
from numba import typesystem

T = numba.template()

@autojit_py3doc(T(T[:, :]), warn=False, locals=dict(scalar=T))
def test_simple_template(array):
    """
    >>> test_simple_template(np.arange(10, 12, dtype=np.float32))
    Traceback (most recent call last):
        ...
    InvalidTemplateError: Template expects 2 dimensions, got 1

    >>> test_simple_template(np.arange(10, 12, dtype=np.float32).reshape(2, 1))
    10.0

    #------------------------------------------------------------------------
    # Test type resolving
    #------------------------------------------------------------------------
    >>> infer(test_simple_template.py_func, float64(float64[:, :]), T(T[:, :]),
    ...       locals=dict(scalar=T))
    [('array', float64[:, :]), ('scalar', float64)]

    >>> infer(test_simple_template.py_func, float64(float64[:, :]), T(T[:, :]),
    ...       locals=dict(scalar=T.pointer()))
    Traceback (most recent call last):
        ...
    UnpromotableTypeError: Cannot promote types float64 * and float64

    #------------------------------------------------------------------------
    # Test type attributes
    #------------------------------------------------------------------------
    >>> infer(test_simple_template.py_func, float64(float64[:, :]), T.dtype(T),
    ...       locals=dict(scalar=T.dtype))
    [('array', float64[:, :]), ('scalar', float64)]
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
S = numba.struct([('a', T1), ('b', T2.pointer()), ('c', T3[:]), ('d', void(T4))])
P = T2.pointer()

type_context1 = { T1: int_, T2: float_, T3: float64, T4: short, }
type_context2 = { T1: int_[:, :], T2: void(float32),
                  T3: numba.struct(a=float64, b=float_), T4: short.pointer(), }

def test_type_matching(array, func, struct, pointer):
    """
    >>> infer(test_type_matching, template_signature=void(A, F, S, P),
    ...       type_context=type_context1)
    [('array', int[:, :]), ('func', void (*)(int)), ('pointer', float32 *), ('struct', struct { int a, float32 * b, float64[:] c, void (*)(short) d })]
    """
    func(array[0, 0])
    struct.b = pointer

def test_type_attributes(array, func, struct, pointer):
    """
    >>> locals = dict(dtype=T1.dtype, arg=T2.args[0], field_a=T3.fielddict['a'],
    ...               field_b=T3.fielddict['b'], scalar=T4.base_type)
    >>> pprint(infer(test_type_attributes, template_signature=void(T1, T2, T3, T4),
    ...              type_context=type_context2, locals=locals))
    [('array', int[:, :]),
     ('func', void (*)(float32)),
     ('pointer', short *),
     ('struct', struct { float64 a, float32 b }),
     ('arg', float32),
     ('dtype', int),
     ('field_a', float64),
     ('field_b', float32),
     ('scalar', short)]
    """
    dtype = array[0, 0]
    arg = 0
    field_a = 0
    field_b = 0
    scalar = 0

@autojit_py3doc(T(T, float64), locals=None)
def test_template_with_concretes(a, b):
    """
    >>> test_template_with_concretes(1, 2)
    3
    """
    return a + b

@autojit(complex128(T, float64), locals=None)
def test_template_with_concretes2(a, b):
    """
    >>> test_template_with_concretes2(1, 2)
    (3+0j)
    >>> test_template_with_concretes2(1.0, 2.0)
    (3+0j)
    >>> test_template_with_concretes2(1+0j, 2)
    (3+0j)

    >>> test_template_with_concretes2(1+0j, 2+0j)
    Traceback (most recent call last):
        ...
    TypeError: can't convert complex to float
    """
    return a + b


@autojit_py3doc(T2(T1, float64), locals=None)
def test_unknown_template_error(a, b):
    """
    >>> test_unknown_template_error(1, 2)
    Traceback (most recent call last):
        ...
    InvalidTemplateError: Unknown template type: T2
    """
    return a + b

@autojit_py3doc(T(T, T), locals=None)
def test_template_inconsistent_types_error(a, b):
    """
    >>> test_template_inconsistent_types_error(1, 2)
    3
    >>> test_template_inconsistent_types_error(1, 2.0)
    Traceback (most recent call last):
        ...
    InvalidTemplateError: Inconsistent types found for template: int and float64
    """
    return a + b

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
                          locals=locals, warn=False)

    if locals is not None:
        local_vars = sorted(locals.iteritems())
    else:
        local_vars = []

    vars = sorted((name, var.type) for name, var in symbols.iteritems())
    return vars + local_vars

def specialize(T, context):
    return typesystem.resolve_template_type(T, context)


if __name__ == '__main__':
    testmod()
