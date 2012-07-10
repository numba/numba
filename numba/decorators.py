import functools
import logging

import numba
from . import naming, utils
from . import ast_type_inference as type_inference
from .minivect import minitypes

import meta.decompiler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a new callable object
#  that creates a fast version of Python code using LLVM
# It maintains the generic function call for situations
#  where it cannot figure out a fast version, and specializes
#  based on the types that are passed in.
#  It maintains a dictionary keyed by python code +
#   argument-types with a tuple of either
#       (bitcode-file-name, function_name)
#  or (llvm mod object and llvm func object)

class CallSite(object):
    # Must support
    # func = CallSite(func)
    # func = CallSite()(func)
    # func = Callsite(*args, **kwds)(func)
    #  args[0] cannot be callable
    def __init__(self, *args, **kwds):
        # True if this instance is now a function
        self._isfunc = False
        self._args = args
        if len(args) > 1 and callable(args[0]):
            self._tocall = args[0]
            self._isfunc = True
            self._args = args[1:]

    def __call__(self, *args, **kwds):
        if self._isfunc:
            return self._tocall(*args, **kwds)
        else:
            if len(args) < 1 or not callable(args[0]):
                raise ValueError, "decorated object must be callable"
            self._tocall = args[0]
            self._isfunc = True
            return self

# A simple fast-vectorize example

#from translate import Translate
from ast_translate import LLVMCodeGenerator as ASTTranslate

# The __tr_map__ global maps from Python functions to a Translate
# object.  This added reference prevents the translator and its
# generated LLVM code from being garbage collected when we leave the
# scope of a decorator.

# See: https://github.com/ContinuumIO/numba/issues/5

__tr_map__ = {}

def vectorize(func):
    global __tr_map__
    try:
        if func not in __tr_map__:
            t = Translate(func)
            t.translate()
            __tr_map__[func] = t
        else:
            t = __tr_map__[func]
        return t.make_ufunc()
    except:
        print "Warning: Could not create fast version..."
        import numpy
        return numpy.vectorize(func)

context = utils.get_minivect_context()

def _get_ast(func):
    return meta.decompiler.decompile_func(func)

def _compile(func, ret_type=None, arg_types=None, **kwds):
    """
    Compile a numba annotated function.

        - decompile function into a Python ast
        - run type inference using the given input types
        - compile the function to LLVM
    """
    global __tr_map__

    ast = _get_ast(func)

    func_signature = minitypes.FunctionType(return_type=ret_type,
                                            args=arg_types)
    func_signature, symtab = type_inference._infer_types(context, func, ast, func_signature)

    func_name = naming.specialized_mangle(func.__name__, func_signature.args)

    if func in __tr_map__:
        print("Warning: Previously compiled version of %r may be "
              "garbage collected!" % (func,))

    t = ASTTranslate(context, func, ast, func_signature=func_signature,
                  func_name=func_name, symtab=symtab, **kwds)
    t.translate()
    logger.debug("Compiled function: %s" % t.lfunc)
    __tr_map__[func] = t
    return t.get_ctypes_func(kwds.get('llvm', True))

def function(f):
    """
    Defines a numba function, that, when called, specializes on the input
    types.
    """
    cache = {}
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        arguments = args + tuple(kwargs[k] for k in sorted(kwargs))
        types = tuple(context.typemapper.from_python(value)
                          for value in arguments)
        if types in cache:
            compiled_func = cache[types]
        else:

            compiled_func = _compile(f, ret_type=None, arg_types=types)
            cache[types] = compiled_func

        return compiled_func(*args, **kwargs)
    return wrapper


# XXX Proposed name; compile() would mask builtin of same name.
def numba_compile(*args, **kws):
    def _numba_compile(func):
        kws.setdefault('ret_type', numba.float64)
        kws.setdefault('arg_types', [numba.float64])
        return _compile(func, *args, **kws)
    return _numba_compile

from kerneltranslate import Translate as KernelTranslate

def numba_kompile(*args, **kws):
    def _numba_kompile(func):
        global __tr_map__
        llvm = kws.pop('llvm', True)
        if func not in __tr_map__:
            t = KernelTranslate(func)
            t.translate()
            __tr_map__[func] = t
        else:
            t = __tr_map__[func]
        return t.get_ctypes_func(llvm)
    return _numba_kompile
