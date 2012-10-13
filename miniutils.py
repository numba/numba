"""
Miscellaneous (convenience) utilities.
"""

import __builtin__

import treepath
from ctypes_conversion import get_data_pointer
import xmldumper
from xmldumper import etree, tostring

#
### Convenience utilities
#


def specialize(context, specializer_cls, ast, print_tree=False):
    "Specialize an AST with given specializer and compile"
    context = context or getcontext()
    specializers = [specializer_cls]
    result = iter(context.run(ast, specializers, print_tree=print_tree)).next()
    _, specialized_ast, _, code_result = result
    if not context.use_llvm:
        prototype, code_result = code_result
    return specialized_ast, code_result

class MiniFunction(object):
    """
    Convenience class to compile a function using LLVM and to invoke the
    function with ctypes given numpy arrays as input.
    """

    def __init__(self, context, specializer, variables, expr, name=None):
        self.b = context.astbuilder
        self.context = context
        self.specializer = specializer
        self.variables = variables
        self.minifunc = self.b.build_function(variables, expr, name)
        self.specialized_ast, (self.lfunc, self.ctypes_func) = specialize(
                                        context, specializer, self.minifunc)

    def get_ctypes_func_and_args(self, arrays):
        fist_array = arrays[0]
        shape = fist_array.shape
        for variable, array in zip(self.variables, arrays):
            for dim, extent in enumerate(array.shape):
                if extent != shape[dim] and extent != 1:
                    raise ValueError("Differing extents in dim %d (%s, %s)" %
                                     (dim, extent, shape[dim]))

        args = [fist_array.ctypes.shape]
        for variable, array in zip(self.variables, arrays):
            if variable.type.is_array:
                data_pointer = get_data_pointer(array, variable.type)
                args.append(data_pointer)
                if not self.specializer.is_contig_specializer:
                    args.append(array.ctypes.strides)
            else:
                raise NotImplementedError

        return args

    def __call__(self, *args, **kwargs):
        import numpy as np

        # print self.minifunc.ndim
        # self.minifunc.print_tree(self.context)
        # print self.context.debug_c(self.minifunc, self.specializer)

        out = kwargs.pop('out', None)
        assert not kwargs, kwargs

        if out is None:
            import minitypes
            dtype = minitypes.map_minitype_to_dtype(self.variables[0].type)
            broadcast = np.broadcast(*args)
            out = np.empty(broadcast.shape, dtype=dtype)

        arrays = [out]
        arrays.extend(args)
        assert len(arrays) == len(self.variables)

        args = self.get_ctypes_func_and_args(arrays)
        self.ctypes_func(*args)
        return out


def xpath(ast, expr):
    return treepath.find_all(ast, expr)


# Compatibility with Python 2.4
def any(it):
    for obj in it:
        if obj:
            return True
    return False

def all(it):
    for obj in it:
        if not obj:
            return False
    return True

def max(it, key=None):
    if key is not None:
        k, value = max((key(value), value) for value in it)
        return value
    return max(it)

def min(it, key=None):
    if key is not None:
        k, value = min((key(value), value) for value in it)
        return value
    return min(it)

class ComparableObjectMixin(object):
    "Make sure subclasses implement comparison and hashing methods"

    def __hash__(self):
        "Implement in subclasses"
        raise NotImplementedError

    def __eq__(self, other):
        "Implement in subclasses"
        return NotImplemented
