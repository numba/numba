import numpy as np

from numba.testing.test_support import *
from numba import typesystem
from numba import pipeline, environment, functions, error


def construct_infer_pipeline():
    env = environment.NumbaEnvironment.get_environment()
    return env.get_pipeline('type_infer')

def functype(restype=None, argtypes=()):
    return typesystem.function(return_type=restype, args=list(argtypes))

def lookup(block, var_name):
    var = None
    try:
        var = block.symtab.lookup_most_recent(var_name)
    except (AssertionError, KeyError):
        if block.idom:
            var = lookup(block.idom, var_name)

    return var

def types(symtab, *varnames):
    return tuple(symtab[varname].type for varname in varnames)

def infer(func, signature=functype(), warn=True, **kwargs):
    func_ast = functions._get_ast(func)
    env = environment.NumbaEnvironment.get_environment(kwargs.get('env', None))
    infer_pipe = env.get_or_add_pipeline('infer', construct_infer_pipeline)
    kwargs.update(warn=warn, pipeline_name='infer')
    pipe, (signature, symtab, func_ast) = pipeline.run_pipeline2(
        env, func, func_ast, signature, **kwargs)
    last_block = func_ast.flow.blocks[-2]
    symbols = {}
    #for block in ast.flow.blocks: print block.symtab
    for var_name, var in symtab.iteritems():
        if not var.parent_var and not var.is_constant:
            var = lookup(last_block, var_name)
            if var:
                symbols[var_name] = var

    return signature, symbols

class Value(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Value(%s)" % self.value

    def __int__(self):
        return self.value

values = [Value(i) for i in range(10)]

@autojit
def test_reassign(obj):
    """
    >>> test_reassign(object())
    'hello'
    >>> sig, syms = infer(test_reassign.py_func, functype(None, [object_]))
    >>> sig
    string (*)(object)
    >>> syms['obj'].type
    string
    """
    obj = 1
    obj = 1.0
    obj = 1 + 4j
    obj = 2
    obj = "hello"
    return obj

@autojit
def test_if_reassign(obj1, obj2):
    """
    >>> test_if_reassign(*values[:2])
    (4.0, 5.0)
    >>> sig, syms = infer(test_if_reassign.py_func,
    ...                   functype(None, [object_] * 2))
    >>> types(syms, 'obj1', 'obj2')
    (float64, object)
    """
    x = 4.0
    y = 5.0
    z = 6.0
    if int(obj1) < int(obj2):
        obj1 = x
        obj2 = y
    else:
        obj1 = z

    return obj1, obj2

@autojit
def test_if_reassign2(value, obj1, obj2):
    """
    >>> test_if_reassign2(0, *values[:2])
    (4.0, 5.0, 'egel')
    >>> test_if_reassign2(1, *values[:2])
    ('hello', 'world', 'hedgehog')
    >>> test_if_reassign2(2, *values[:2])
    ([Value(0)], Value(12), 'igel')

    >>> sig, syms = infer(test_if_reassign2.py_func,
    ...                   functype(None, [int_, object_, object_]))
    >>> types(syms, 'obj1', 'obj2', 'obj3')
    (object, object, string)
    """
    x = 4.0
    y = 5.0
    z = "hedgehog"
    if value < 1:
        obj1 = x
        obj2 = y
        obj3 = "egel"
    elif value < 2:
        obj1 = "hello"
        obj2 = "world"
        obj3 = z
    else:
        obj1 = [obj1]
        obj2 = Value(12)
        obj3 = "igel"

    return obj1, obj2, obj3

@autojit_py3doc
def test_for_reassign(obj1, obj2, obj3, obj4):
    """
    >>> test_for_reassign(*values[:4])
    (9, Value(1), 2, 5)
    >>> sig, syms = infer(test_for_reassign.py_func,
    ...                   functype(None, [object_] * 4))
    >>> types(syms, 'obj1', 'obj2', 'obj3')
    (object, object, int)
    """
    for i in range(10):
        obj1 = i

    for i in range(0):
        obj2 = i

    for i in range(10):
        obj3 = i
    else:
        obj3 = 2 # This definition kills any previous definition

    for i in range(5, 10):
        obj4 = i
        break
    else:
        obj4 = 0

    return obj1, obj2, obj3, obj4

@autojit_py3doc
def test_while_reassign(obj1, obj2, obj3, obj4):
    """
    >>> test_while_reassign(*values[:4])
    (9, Value(1), 2, 5)
    >>> sig, syms = infer(test_while_reassign.py_func,
    ...                   functype(None, [object_] * 4))
    >>> types(syms, 'obj1', 'obj2', 'obj3', 'obj4')
    (object, object, int, int)
    """
    i = 0
    while i < 10:
        obj1 = i
        i += 1

    i = 0
    while i < 0:
        obj2 = i
        i += 1

    i = 0
    while i < 10:
        obj3 = i
        i += 1
    else:
        obj3 = 2 # This definition kills any previous definition

    i = 5
    while i < 10:
        obj4 = i
        i += 1
        break
    else:
        obj4 = 0

    return obj1, obj2, obj3, obj4

@autojit(warn=False)
def test_conditional_assignment(value):
    """
    >>> test_conditional_assignment(0)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)

    >>> test_conditional_assignment(1)
    Traceback (most recent call last):
        ...
    UnboundLocalError: 210:11: obj1
    """
    if value < 1:
        obj1 = np.ones(10, dtype=np.float32)

    return obj1


#
### Test for errors
#
# @autojit
# def test_error_array_variable1(value, obj1):
#     """
#     >>> test_error_array_variable1(0, object())
#     Traceback (most recent call last):
#         ...
#     TypeError: Arrays must have consistent types in assignment for variable 'obj1': 'float32[:]' and 'object_'
#     """
#     if value < 1:
#         obj1 = np.empty(10, dtype=np.float32)
#
#     return obj1

def test():
    from . import test_cfg_type_infer
    testmod(test_cfg_type_infer)

if __name__ == '__main__':
    testmod()
#else:
#    test()
