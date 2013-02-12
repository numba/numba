__all__ = ['autojit', 'jit', 'export', 'exportmany']

import types
import functools
import logging
import inspect

from numba import *
from numba import typesystem, numbawrapper
from . import utils, functions, ast_translate as translate
from numba import  pipeline, extension_type_inference
from .minivect import minitypes
from numba.utils import debugout, process_signature
from numba.intrinsic import default_intrinsic_library
from numba.external import default_external_library
from numba.external.utility import default_utility_library
from numba import double, int_
from numba import environment
import llvm.core as _lc

context = utils.get_minivect_context()
context.llvm_context = translate.LLVMContextManager()
context.numba_pipeline = pipeline.Pipeline
function_cache = context.function_cache = functions.FunctionCache(context)
context.intrinsic_library = default_intrinsic_library(context)
context.external_library = default_external_library(context)
context.utility_library = default_utility_library(context)

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
            # ast_translate.LLVMCodeGenerator.__init__().
            with environment.TranslationContext(
                    env, func, func_ast, function_signature,
                    name=name, llvm_module=llvm_module,
                    mangled_name=name) as func_env:
                pipeline = env.get_pipeline()
                func_ast.pipeline = pipeline
                pipeline(func_ast, env)
                exports_env = env.exports
                exports_env.function_signature_map[name] = function_signature
                exports_env.function_module_map[name] = llvm_module
                if not exports_env.wrap_exports:
                    exports_env.function_wrapper_map[name] = None
                else:
                    wrapper_tup = func_env.translator.build_wrapper_module()
                    exports_env.function_wrapper_map[name] = wrapper_tup
        return func
    return _iexport

def export(signature, **kws):
    """
    Construct a decorator that takes a function and exports one

    A signature is a string with

    name ret_type(arg_type, argtype, ...)
    """
    env = environment.NumbaEnvironment.get_environment(kws.pop('env', None))
    return _internal_export(env, process_signature(signature), **kws)

def exportmany(signatures, **kws):
    """
    A Decorator that exports many signatures for a single function
    """
    env = environment.NumbaEnvironment.get_environment(kws.pop('env', None))
    def _export(func):
        for signature in signatures:
            tocall = _internal_export(env, process_signature(signature), **kws)
            tocall(func)
        return func
    return _export

logger = logging.getLogger(__name__)

def jit_extension_class(py_class, translator_kwargs):
    env = environment.NumbaEnvironment.get_environment(
        translator_kwargs.get('env', None))
    llvm_module = translator_kwargs.get('llvm_module', None)
    if llvm_module is None:
        llvm_module = _lc.Module.new('tmp.extension_class.%X' % id(py_class))
        translator_kwargs['llvm_module'] = llvm_module
    return extension_type_inference.create_extension(
        context, py_class, translator_kwargs)

def resolve_argtypes(numba_func, template_signature,
                     args, kwargs, translator_kwargs):
    """
    Given an autojitting numba function, return the argument types.
    These need to be resolved in order for the function cache to work.

    TODO: have a single entry point that resolved the argument types!
    """
    assert not kwargs, "Keyword arguments are not supported yet"

    locals_dict = translator_kwargs.get("locals", None)

    return_type = None
    argnames = inspect.getargspec(numba_func.py_func).args
    env = environment.NumbaEnvironment.get_environment(
        translator_kwargs.get('env', None))
    argtypes = map(env.context.typemapper.from_python, args)

    if template_signature is not None:
        template_context, signature = typesystem.resolve_templates(
                locals_dict, template_signature, argnames, argtypes)
        return_type = signature.return_type
        argtypes = list(signature.args)

    if locals_dict is not None:
        for i, argname in enumerate(argnames):
            if argname in locals_dict:
                new_type = locals_dict[argname]
                argtypes[i] = new_type

    return minitypes.FunctionType(return_type, tuple(argtypes))

def _autojit(template_signature, target, nopython, **translator_kwargs):
    def _autojit_decorator(f):
        """
        Defines a numba function, that, when called, specializes on the input
        types. Uses the AST translator backend. For the bytecode translator,
        use @autojit.
        """
        def compile_function(args, kwargs):
            "Compile the function given its positional and keyword arguments"
            signature = resolve_argtypes(numba_func, template_signature,
                                         args, kwargs, translator_kwargs)
            dec = _jit(restype=signature.return_type,
                       argtypes=signature.args,
                       target=target, nopython=nopython,
                       **translator_kwargs)

            compiled_function = dec(f)
            return compiled_function

        function_cache.register(f)
        cache = function_cache.get_autojit_cache(f)

        wrapper = autojit_wrappers[(target, 'ast')]
        numba_func = wrapper(f, compile_function, cache)
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
    if template_signature and not isinstance(template_signature, minitypes.Type):
        if callable(template_signature):
            func = template_signature
            return autojit(backend='ast', target=target,
                           nopython=nopython, locals=locals, **kwargs)(func)
        else:
            raise Exception("The autojit decorator should be called: "
                            "@autojit(backend='ast')")

    if backend == 'bytecode':
        return _not_implemented
    else:
        return _autojit(template_signature, target, nopython,
                        locals=locals, **kwargs)

def _jit(restype=None, argtypes=None, nopython=False,
         _llvm_module=None, **kwargs):
    def _jit_decorator(func):
        argtys = argtypes
        if func.func_code.co_argcount == 0 and argtys is None:
            argtys = []

        assert argtys is not None
        function_cache.register(func)

        assert kwargs.get('llvm_module') is None # TODO link to user module
        assert kwargs.get('llvm_ee') is None, "Engine should never be provided"
        result = function_cache.compile_function(func, argtys,
                                                 restype=restype,
                                                 nopython=nopython,
                                                 ctypes=False,
                                                 **kwargs)
        signature, lfunc, wrapper_func = result
        return numbawrapper.NumbaCompiledWrapper(func, wrapper_func,
                                                 signature, lfunc)

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
    if isinstance(restype, (type, types.ClassType)):
        cls = restype
        return jit_extension_class(cls, kws)

    # Called with f8(f8) syntax which returns a dictionary of argtypes and restype
    if isinstance(restype, minitypes.FunctionType):
        if argtypes is not None:
            raise TypeError, "Cannot use both calling syntax and argtypes keyword"
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
