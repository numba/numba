# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba.exttypes.entrypoints import  (jit_extension_class,
                                         autojit_extension_class,
                                         autojit_class_wrapper)

__all__ = ['autojit', 'jit', 'export', 'exportmany']

import types
import logging
import inspect

from numba import *
from numba import typesystem, numbawrapper
from numba import  functions
from numba.utils import  process_signature
from numba.codegen import llvmwrapper
from numba import environment
import llvm.core as _lc
from numba.wrapping import compiler

logger = logging.getLogger(__name__)

environment.NumbaEnvironment.get_environment().link_cbuilder_utilities()

if PY3:
    CLASS_TYPES = type
else:
    CLASS_TYPES = (type, types.ClassType)

#------------------------------------------------------------------------
# PyCC decorators
#------------------------------------------------------------------------

def _internal_export(env, function_signature, backend='ast', **kws):
    def _iexport(func):
        if backend == 'bytecode':
            raise NotImplementedError(
               'Bytecode translation has been removed for exported functions.')
        else:
            name = function_signature.name
            llvm_module = _lc.Module.new('export_%s' % name)
            if not hasattr(func, 'live_objects'):
                func.live_objects = []
            func._is_numba_func = True
            func_ast = functions._get_ast(func)
            # FIXME: Hacked "mangled_name" into the translation
            # environment.  Should do something else.  See comment in
            # codegen.translate.LLVMCodeGenerator.__init__().
            with environment.TranslationContext(
                    env, func, func_ast, function_signature,
                    name=name, llvm_module=llvm_module,
                    mangled_name=name,
                    link=False, wrap=False,
                    is_pycc=True) as func_env:
                pipeline = env.get_pipeline()
                func_ast.pipeline = pipeline
                pipeline(func_ast, env)
                exports_env = env.exports
                exports_env.function_signature_map[name] = function_signature
                exports_env.function_module_map[name] = llvm_module
                if not exports_env.wrap_exports:
                    exports_env.function_wrapper_map[name] = None
                else:
                    wrapper_tup = llvmwrapper.build_wrapper_module(env)
                    exports_env.function_wrapper_map[name] = wrapper_tup
        return func
    return _iexport

def export(signature, env_name=None, env=None, **kws):
    """
    Construct a decorator that takes a function and exports one

    A signature is a string with

    name ret_type(arg_type, argtype, ...)
    """
    if env is None:
        env = environment.NumbaEnvironment.get_environment(env_name)
    return _internal_export(env, process_signature(signature), **kws)

def exportmany(signatures, env_name=None, env=None, **kws):
    """
    A Decorator that exports many signatures for a single function
    """
    if env is None:
        env = environment.NumbaEnvironment.get_environment(env_name)
    def _export(func):
        for signature in signatures:
            tocall = _internal_export(env, process_signature(signature), **kws)
            tocall(func)
        return func
    return _export

#------------------------------------------------------------------------
# Compilation Entry Points
#------------------------------------------------------------------------

# TODO: Redo this entire module

def compile_function(env, func, argtypes, restype=None, func_ast=None, **kwds):
    """
    Compile a python function given the argument types. Compile only
    if not compiled already, and only if it is registered to the function
    cache.

    Returns a triplet of (signature, llvm_func, python_callable)
    `python_callable` is the wrapper function (NumbaFunction).
    """
    function_cache = env.specializations

    # For NumbaFunction, we get the original python function.
    func = getattr(func, 'py_func', func)

    # get the compile flags
    flags = None # stub

    # Search in cache
    result = function_cache.get_function(func, argtypes, flags)
    if result is not None:
        sig, lfunc, pycall = result
        return sig, lfunc, pycall

    # Compile the function
    from numba import pipeline

    compile_only = getattr(func, '_numba_compile_only', False)
    kwds['compile_only'] = kwds.get('compile_only', compile_only)

    assert kwds.get('llvm_module') is None, kwds.get('llvm_module')

    func_env = pipeline.compile2(env, func, restype, argtypes, func_ast=func_ast, **kwds)

    function_cache.register_specialization(func_env)
    return (func_env.func_signature,
            func_env.lfunc,
            func_env.numba_wrapper_func)


def _autojit(template_signature, target, nopython, env_name=None, env=None,
             **flags):
    if env is None:
        env = environment.NumbaEnvironment.get_environment(env_name)

    def _autojit_decorator(f):
        """
        Defines a numba function, that, when called, specializes on the input
        types. Uses the AST translator backend. For the bytecode translator,
        use @autojit.
        """

        if isinstance(f, CLASS_TYPES):
            compiler_cls = compiler.ClassCompiler
            wrapper = autojit_class_wrapper
        else:
            compiler_cls = compiler.FunctionCompiler
            wrapper = autojit_wrappers[(target, 'ast')]

        env.specializations.register(f)
        cache = env.specializations.get_autojit_cache(f)

        flags['target'] = target
        compilerimpl = compiler_cls(env, f, nopython, flags, template_signature)
        numba_func = wrapper(f, compilerimpl, cache)

        return numba_func

    return _autojit_decorator

def autojit(template_signature=None, backend='ast', target='cpu',
            nopython=False, locals=None, **kwargs):
    """
    Creates a function that dispatches to type-specialized LLVM
    functions based on the input argument types.  If no specialized
    function exists for a set of input argument types, the dispatcher
    creates and caches a new specialized function at call time.
    """
    if template_signature and not isinstance(template_signature, typesystem.Type):
        if callable(template_signature):
            func = template_signature
            return autojit(backend='ast', target=target,
                           nopython=nopython, locals=locals, **kwargs)(func)
        else:
            raise Exception("The autojit decorator should be called: "
                            "@autojit()")

    if backend == 'bytecode':
        return _not_implemented
    else:
        return _autojit(template_signature, target, nopython,
                        locals=locals, **kwargs)

def _jit(restype=None, argtypes=None, nopython=False,
         _llvm_module=None, env_name=None, env=None, func_ast=None, **kwargs):
    #print(ast.dump(func_ast))
    if env is None:
        env = environment.NumbaEnvironment.get_environment(env_name)
    def _jit_decorator(func):
        if isinstance(func, CLASS_TYPES):
            cls = func
            kwargs.update(env_name=env_name)
            return jit_extension_class(cls, kwargs, env)

        argtys = argtypes
        if argtys is None and restype:
            assert restype.is_function
            return_type = restype.return_type
            argtys = restype.args
        elif argtys is None:
            assert func.__code__.co_argcount == 0, func
            return_type = None
            argtys = []
        else:
            return_type = restype

        assert argtys is not None
        env.specializations.register(func)

        assert kwargs.get('llvm_module') is None # TODO link to user module
        assert kwargs.get('llvm_ee') is None, "Engine should never be provided"
        sig, lfunc, wrapper = compile_function(env, func, argtys,
                                               restype=return_type,
                                               nopython=nopython, func_ast=func_ast, **kwargs)
        return numbawrapper.create_numba_wrapper(func, wrapper, sig, lfunc)

    return _jit_decorator

def _not_implemented(*args, **kws):
    raise NotImplementedError('Bytecode backend is no longer supported.')

jit_targets = {
    ('cpu', 'bytecode') : _not_implemented,
    ('cpu', 'ast') : _jit,
}

autojit_wrappers = {
    ('cpu', 'bytecode') : _not_implemented,
    ('cpu', 'ast')      : numbawrapper.NumbaSpecializingWrapper,
}

def jit(restype=None, argtypes=None, backend='ast', target='cpu', nopython=False,
        **kws):
    """
    Compile a function given the input and return types.

    There are multiple ways to specify the type signature:

    * Using the restype and argtypes arguments, passing Numba types.

    * By constructing a Numba function type and passing that as the
      first argument to the decorator.  You can create a function type
      by calling an exisiting Numba type, which is the return type,
      and the arguments to that call define the argument types.  For
      example, ``f8(f8)`` would create a Numba function type that
      takes a single double-precision floating point value argument,
      and returns a double-precision floating point value.

    * As above, but using a string instead of a constructed function
      type.  Example: ``jit("f8(f8)")``.

    If backend='bytecode' the bytecode translator is used, if
    backend='ast' the AST translator is used.  By default, the AST
    translator is used.  *Note that the bytecode translator is
    deprecated as of the 0.3 release.*
    """
    kws.update(nopython=nopython, backend=backend)
    if isinstance(restype, CLASS_TYPES):
        cls = restype
        env = kws.pop('env', None) or environment.NumbaEnvironment.get_environment(
                                                         kws.get('env_name', None))
        return jit_extension_class(cls, kws, env)

    # Called with f8(f8) syntax which returns a dictionary of argtypes and restype
    if isinstance(restype, typesystem.function):
        if argtypes is not None:
            raise TypeError("Cannot use both calling syntax and argtypes keyword")
        argtypes = restype.args
        restype = restype.return_type
    # Called with a string like 'f8(f8)'
    elif isinstance(restype, str) and argtypes is None:
        signature = process_signature(restype, kws.get('name', None))
        name, restype, argtypes = (signature.name, signature.return_type,
                                   signature.args)
        if name is not None:
            kws['func_name'] = name
    if restype is not None:
        kws['restype'] = restype
    if argtypes is not None:
        kws['argtypes'] = argtypes

    return jit_targets[target, backend](**kws)

