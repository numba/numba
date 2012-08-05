import functools
import logging

import numba
from . import utils, functions, ast_translate as translate
from numba import translate as bytecode_translate
from .minivect import minitypes

logger = logging.getLogger(__name__)

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
    except Exception as msg:
        print "Warning: Could not create fast version...", msg
        import numpy
        return numpy.vectorize(func)

context = utils.get_minivect_context()
context.llvm_context = translate.LLVMContextManager()
function_cache = context.function_cache = functions.FunctionCache(context)

def function(f):
    """
    Defines a numba function, that, when called, specializes on the input
    types.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        arguments = args + tuple(kwargs[k] for k in sorted(kwargs))
        types = tuple(context.typemapper.from_python(value)
                          for value in arguments)
        result = function_cache.compile_function(f, types)
        _, _, ctypes_func = result
        return ctypes_func(*args, **kwargs)

    wrapper._is_numba_func = True
    wrapper._numba_func = f
    f._is_numba_func = True
    return wrapper

# XXX Proposed name; compile() would mask builtin of same name.
# NOTE: overridden below to use the bytecode translator
def numba_compile(ret_type, arg_types, **kws):
    def _numba_compile(func):
        func._is_numba_func = True
        _, _, ctypes_func = function_cache.compile_function(
                        func, arg_types=arg_types, ret_type=ret_type, **kws)
        return ctypes_func
    return _numba_compile

def numba_compile(*args, **kws):
    def _numba_compile(func):
        global __tr_map__
        llvm = kws.pop('llvm', True)
        if func in __tr_map__:
            print("Warning: Previously compiled version of %r may be "
                  "garbage collected!" % (func,))
        t = bytecode_translate.Translate(func, *args, **kws)
        t.translate()
        __tr_map__[func] = t
        return t.get_ctypes_func(llvm)
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
