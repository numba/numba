__all__ = ['autojit', 'jit2', 'jit', 'export']

import functools
import logging
import types

from numba import *
from . import _numba_types
from . import utils, functions, ast_translate as translate, ast_type_inference
from numba import translate as bytecode_translate
from numba import error
from .minivect import minitypes
from numba.utils import debugout

from numba import double
import llvm.core as _lc

default_module = _lc.Module.new('default')
translated = []
def export(restype=double, argtypes=[double], backend='bytecode', **kws):
    def _export(func, name=None):
        # XXX: need to implement ast backend.
        t = bytecode_translate.Translate(func, restype=restype,
                                         argtypes=argtypes,
                                         module=default_module,
                                         name=name, **kws)
        t.translate()
        t.mini_rettype = restype
        t.mini_argtypes = argtypes
        translated.append((t, name))
    return _export


logger = logging.getLogger(__name__)

# A simple fast-vectorize example was removed because it only supports one
#  use-case --- slower NumPy vectorize is included here instead.
#  The required code is still in _ext.c which is not compiled by default
#   and here is the decorator:
#def vectorize(func):
#    global __tr_map__
#    try:
#        if func not in __tr_map__:
#            t = Translate(func)
#            t.translate()
#            __tr_map__[func] = t
#        else:
#            t = __tr_map__[func]
#        return t.make_ufunc()
#    except Exception as msg:
#        print "Warning: Could not create fast version...", msg
#        import traceback
#        traceback.print_exc()
#        import numpy
#        return numpy.vectorize(func)

from numpy import vectorize

# The __tr_map__ global maps from Python functions to a Translate
# object.  This added reference prevents the translator and its
# generated LLVM code from being garbage collected when we leave the
# scope of a decorator.

# See: https://github.com/ContinuumIO/numba/issues/5

__tr_map__ = {}

context = utils.get_minivect_context()
context.llvm_context = translate.LLVMContextManager()
context.numba_pipeline = ast_type_inference.Pipeline
function_cache = context.function_cache = functions.FunctionCache(context)

class NumbaFunction(object):
    """
    Numba function.

        py_func: original Python function
        ctypes_func: LLVM function wrapper, callable from Python
        signature: minitype FunctionType signature
        lfunc: LLVM function
        methoddef: PyMethodDef ctypes structure for the wrapper function
    """

    def __init__(self, py_func, wrapper=None, ctypes_func=None, signature=None,
                 lfunc=None):
        self.py_func = py_func
        self.wrapper = wrapper
        self.ctypes_func = ctypes_func
        self.signature = signature
        self.lfunc = lfunc

        self.func_name = self.__name__ = py_func.__name__
        self.func_doc = self.__doc__ = py_func.__doc__
        self.__module__ = py_func.__module__

        if ctypes_func is None:
            self._is_numba_func = True
            self._numba_func = py_func

    def __repr__(self):
        if self.ctypes_func:
            compiled = 'compiled numba function (%s)' % self.signature
        else:
            compiled = 'specializing numba function'

        return '<%s %s>' % (compiled, self.py_func)

    def __call__(self, *args, **kwargs):
        if self.ctypes_func:
            return self.invoke_compiled(self.ctypes_func, *args, **kwargs)
        else:
            if kwargs:
                raise error.NumbaError("Cannot handle keyword arguments yet")

            nargs = self.py_func.func_code.co_argcount
            if len(args) != nargs:
                raise error.NumbaError("Expected %d arguments, got %d" % (
                                                        len(args), nargs))
            return self.wrapper(self, *args, **kwargs)

    def invoke_compiled(self, compiled_numba_func, *args, **kwargs):
        return compiled_numba_func(*args, **kwargs)


# TODO: make these two implementations the same
def _autojit2(target, nopython, **translator_kwargs):
    def _autojit2_decorator(f):
        """
        Defines a numba function, that, when called, specializes on the input
        types. Uses the AST translator backend. For the bytecode translator,
        use @autojit.
        """
        @functools.wraps(f)
        def wrapper(numba_func, *args, **kwargs):
            arguments = args + tuple(kwargs[k] for k in sorted(kwargs))
            types = tuple(context.typemapper.from_python(value)
                              for value in arguments)
            dec = jit2(argtypes=types, target=target, nopython=nopython,
                       **translator_kwargs)
            compiled_numba_func = dec(f)
            return numba_func.invoke_compiled(compiled_numba_func, *args, **kwargs)

        f.live_objects = []
        numba_func = numba_function_autojit_targets[target](f, wrapper=wrapper)
        return numba_func

    return _autojit2_decorator

_func_cache = {}

def _autojit(target, nopython):
    def _autojit_decorator(f):
        """
        Defines a numba function, that, when called, specializes on the input
        types. Uses the bytecode translator backend. For the AST backend use
        @autojit2
        """
        @functools.wraps(f)
        def wrapper(numba_func, *args, **kwargs):
            # Infer argument types
            arguments = args + tuple(kwargs[k] for k in sorted(kwargs))
            types = tuple(context.typemapper.from_python(value)
                              for value in arguments)
            if types in _func_cache:
                compiled_numba_func = _func_cache[types]
            else:
                # Infer the return type
                func_signature, symtab, ast = functions._infer_types(
                                            context, f, argtypes=types)

                decorator = jit(restype=func_signature.return_type,
                                argtypes=types, target=target)
                compiled_numba_func = decorator(f)
                _func_cache[types] = compiled_numba_func

            return numba_func.invoke_compiled(compiled_numba_func, *args, **kwargs)

        f.live_objects = []
        numba_func = numba_function_autojit_targets[target](f, wrapper=wrapper)
        return numba_func

    return _autojit_decorator

def autojit(backend='ast', target='cpu', nopython=False, locals=None):
    if backend not in ('bytecode', 'ast'):
        if callable(backend):
            func = backend
            return autojit(backend='ast', target=target,
                           nopython=nopython, locals=locals)(func)
        else:
            raise Exception("The autojit decorator should be called: "
                            "@autojit(backend='bytecode|ast')")

    if backend == 'bytecode':
        return _autojit(target, nopython)
    else:
        return _autojit2(target, nopython, locals=locals)

def _jit2(restype=None, argtypes=None, nopython=False,
          _llvm_module=None, _llvm_ee=None, **kwargs):
    def _jit2_decorator(func):
        argtys = argtypes
        if func.func_code.co_argcount == 0 and argtys is None:
            argtys = []

        assert argtys is not None

        if not hasattr(func, 'live_objects'):
            func.live_objects = []
        func._is_numba_func = True
        result = function_cache.compile_function(func, argtys,
                                                 nopython=nopython,
                                                 ctypes=False,
                                                 llvm_module=_llvm_module,
                                                 llvm_ee=_llvm_ee,
                                                 **kwargs)
        signature, lfunc, wrapper_func = result
        return NumbaFunction(func, ctypes_func=wrapper_func,
                             signature=signature, lfunc=lfunc)

    return _jit2_decorator

def _jit(restype=None, argtypes=None, backend='bytecode', **kws):
    assert 'arg_types' not in kws
    assert 'ret_type' not in kws
    def _jit(func):
        global __tr_map__

        llvm = kws.pop('llvm', True)
        if func in __tr_map__:
            logger.warning("Warning: Previously compiled version of %r may be "
                           "garbage collected!" % (func,))

        use_ast = False
        if backend == 'ast':
            use_ast = True
            if argtypes and restype:
                for arg_type in list(argtypes) + [restype]:
                    if not isinstance(arg_type, minitypes.Type):
                        use_ast = False
                        debugout("String type specified, using bytecode translator...")
                        break

        if use_ast:
            return jit2(argtypes=argtypes)(func)
        else:
            if argtypes is None:
                argtyps = [double]
            else:
                argtyps = argtypes
            t = bytecode_translate.Translate(func, restype=restype or double,
                                             argtypes=argtyps, **kws)
            t.translate()
            # print t.lfunc
            __tr_map__[func] = t
            ctypes_func = t.get_ctypes_func(llvm)
            return NumbaFunction(func, ctypes_func=ctypes_func, lfunc=t.lfunc)

    return _jit

jit_targets = {
    'cpu': _jit,
}

jit2_targets = {
    'cpu': _jit2,
}

numba_function_autojit_targets = {
    'cpu': NumbaFunction,
}

def jit(restype=None, argtypes=None, backend='bytecode', target='cpu',
        **kws):
    """
    Compile a function given the input and return types. If backend='bytecode'
    the bytecode translator is used, if backend='ast' the AST translator is
    used.
    """
    # Called with f8(f8) syntax which returns a dictionary of argtypes and restype
    if isinstance(restype, dict) and restype.has_key('argtypes') and restype.has_key('restype'):
        if argtypes is not None:
            raise TypeError, "Cannot use both calling syntax and argtypes keyword"
        argtypes = restype['argtypes']
        restype = restype['restype']
    # Called with a string like 'f8(f8)'
    elif isinstance(restype, str) and backend != 'bytecode':
        loc = {}
        types_dict = dict(globals(), d=double)
        signature = eval(restype, loc, types_dict)
        argtypes = signature['argtypes']
        restype = signature['restype']
    if restype is not None:
        kws['restype'] = restype
    if argtypes is not None:
        kws['argtypes'] = argtypes

    kws['backend'] = backend
    return jit_targets[target](**kws)

def jit2(restype=None, argtypes=None, _llvm_module=None, _llvm_ee=None,
          target='cpu', nopython=False, **kwargs):
    """
    Use the AST translator to translate the function.
    """
    return jit2_targets[target](restype, argtypes, nopython=nopython,
                                _llvm_module=_llvm_module, _llvm_ee=_llvm_ee,
                                **kwargs)
