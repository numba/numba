from numba.tests.test_support import *
from numba.minivect import minitypes
from numba import pipeline, decorators, functions

order = pipeline.Pipeline.order
order = order[:order.index('dump_cfg') + 1]

def functype(restype=None, argtypes=()):
    return minitypes.FunctionType(return_type=restype, args=list(argtypes))

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

def infer(func, signature=functype()):
    ast = functions._get_ast(func)
    pipe, (signature, symtab, ast) = pipeline.run_pipeline(
                        decorators.context, func, ast, signature, order=order)

    last_block = ast.flow.blocks[-2]
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
    const char * (*)(object_)
    >>> syms['obj'].type
    const char *
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
    (double, object_)
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
def test_loop_reassign(obj1, obj2, obj3, obj4):
    """
    >>> test_loop_reassign(*values[:4])
    (9L, Value(1), 2L, 5L)
    >>> sig, syms = infer(test_loop_reassign.py_func,
    ...                   functype(None, [object_] * 4))
    >>> types(syms, 'obj1', 'obj2', 'obj3', 'obj4')
    (object_, object_, int, Py_ssize_t)
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

testmod()