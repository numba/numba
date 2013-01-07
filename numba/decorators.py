__all__ = ['autojit', 'jit', 'export', 'exportmany']

import functools
import logging
import types
import inspect

from numba import *
from numba import typesystem
from . import utils, functions, ast_translate as translate, ast_type_inference
from numba import translate as bytecode_translate
from numba import error, pipeline, extension_type_inference
from .minivect import minitypes
from numba.utils import debugout
from numba.intrinsic import default_intrinsic_library
from numba.external import default_external_library
from numba import double, int_
import llvm.core as _lc

context = utils.get_minivect_context()
context.llvm_context = translate.LLVMContextManager()
context.numba_pipeline = pipeline.Pipeline
function_cache = context.function_cache = functions.FunctionCache(context)
context.intrinsic_library = default_intrinsic_library(context)
context.external_library = default_external_library(context)
pipeline_env = pipeline.PipelineEnvironment.init_env(
    context, "Top-level compilation environment (in %s).\n" % __name__)

_EXPORTS_DOC = '''pycc environment

Includes flags, and modules for exported functions.

wrap - Boolean flag used to indicate that Python wrappers should be
generated for exported functions.

function_signature_map - Map from function names to type signatures
for the translated function (used for header generation).

function_module_map - Map from function names to LLVM modules that
define the translated function.

function_wrapper_map - Map from function names to tuples containing
LLVM wrapper functions and LLVM modules that define the wrapper
function.
'''

def _init_exports(environment=pipeline_env):
    '''
    Initialize the sub-environment specific to exporting functions.
    See environment documentation in decorators._EXPORTS_DOC.
    '''
    exports_env = pipeline.PipelineEnvironment(pipeline_env,
                                               _EXPORTS_DOC)
    environment.exports = exports_env
    exports_env.wrap = False
    exports_env.function_signature_map = {}
    exports_env.function_module_map = {}
    exports_env.function_wrapper_map = {}

_init_exports()

def _internal_export(name=None, restype=double, argtypes=[double],
                     backend='ast', **kws):
    def _iexport(func):
        if backend == 'bytecode':
            raise NotImplementedError(
               'Bytecode translation has been removed for exported functions.')
        else:
            # to reassign need to setup this variable
            # with no 'nonlocal'
            artypes = argtypes
            if func.func_code.co_argcount == 0 and artypes is None:
                artypes = []
            function_signature = minitypes.FunctionType(restype, artypes,
                                                        name=name)
            llvm_module = _lc.Module.new('export_%s' % name)
            func.live_objects = []
            func._is_numba_func = True
            func_ast = functions._get_ast(func)
            func_ast.pipeline = pipeline_env.pipeline
            # FIXME: Hacked "mangled_name" into the translation
            # environment.  Should do something else.  See comment in
            # ast_translate.LLVMCodeGenerator.__init__().
            pipeline_env.crnt.init_func(func, func_ast, function_signature,
                                        name=name, llvm_module=llvm_module,
                                        mangled_name=name)
            pipeline_env.pipeline(func_ast, pipeline_env)
            func_env = pipeline_env.crnt
            exports_env = pipeline_env.exports
            exports_env.function_signature_map[name] = function_signature
            exports_env.function_module_map[name] = llvm_module
            if not exports_env.wrap:
                exports_env.function_wrapper_map[name] = None
            else:
                wrapper_tup = func_env.translator.build_wrapper_module()
                exports_env.function_wrapper_map[name] = wrapper_tup
    return _iexport

def export(signature, **kws):
    """
    Construct a decorator that takes a function and exports one

    A signature is a string with

    name ret_type(arg_type, argtype, ...)
    """
    name, restype, argtypes = _process_sig(signature)
    return _internal_export(name=name, restype=restype, argtypes=argtypes, **kws)   

def exportmany(signatures, **kws):
    """
    A Decorator that exports many signatures for a single function
    """   
    def _export(func):
        for signature in signatures:
            name, restype, argtypes = _process_sig(signature)
            tocall = _internal_export(name=name, restype=restype, argtypes=argtypes, **kws)
            tocall(func)

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

        name = py_func.__name__
        if signature is not None:
            name = signature.name or name
        self.func_name = self.__name__ = name
        self.func_doc = self.__doc__ = py_func.__doc__
        self.__module__ = py_func.__module__

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
                                                        nargs, len(args)))
            return self.wrapper(self, *args, **kwargs)

    def invoke_compiled(self, compiled_numba_func, *args, **kwargs):
        return compiled_numba_func(*args, **kwargs)

def jit_extension_class(py_class, translator_kwargs):
    llvm_module = translator_kwargs.get('llvm_module')
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
    argtypes = map(context.typemapper.from_python, args)

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

# TODO: make these two implementations the same
def _autojit2(template_signature, target, nopython, **translator_kwargs):
    def _autojit2_decorator(f):
        """
        Defines a numba function, that, when called, specializes on the input
        types. Uses the AST translator backend. For the bytecode translator,
        use @autojit.
        """
        @functools.wraps(f)
        def wrapper(numba_func, *args, **kwargs):
            signature = resolve_argtypes(numba_func, template_signature,
                                         args, kwargs, translator_kwargs)
            dec = jit2(restype=signature.return_type,
                        argtypes=signature.args,
                        target=target, nopython=nopython,
                        **translator_kwargs)
            compiled_numba_func = dec(f)
            return numba_func.invoke_compiled(compiled_numba_func, *args, **kwargs)

        f.live_objects = []
        numba_func = numba_function_autojit_targets[target](f, wrapper=wrapper)
        function_cache.register(f)
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
                func_signature, symtab, ast = pipeline.infer_types(
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
                            "@autojit(backend='bytecode|ast')")

    if backend == 'bytecode':
        return _autojit(target, nopython, **kwargs)
    else:
        return _autojit2(template_signature, target, nopython,
                         locals=locals, **kwargs)

def _jit2(restype=None, argtypes=None, nopython=False,
          _llvm_module=None, **kwargs):
    def _jit2_decorator(func):
        argtys = argtypes
        if func.func_code.co_argcount == 0 and argtys is None:
            argtys = []

        assert argtys is not None

        if not hasattr(func, 'live_objects'):
            func.live_objects = []

        function_cache.register(func)

        assert kwargs.get('llvm_module') is None # TODO link to user module
        assert kwargs.get('llvm_ee') is None, "Engine should never be provided"
        result = function_cache.compile_function(func, argtys,
                                                 restype=restype,
                                                 nopython=nopython,
                                                 ctypes=False,
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
            return jit2(restype=restype, argtypes=argtypes)(func)
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
    ('cpu', 'bytecode') : _jit,
    ('cpu', 'ast') : _jit2
}

numba_function_autojit_targets = {
    'cpu': NumbaFunction,
}

def _process_sig(sigstr, name=None):
    sigstr = sigstr.replace('*', '.pointer()')    
    parts = sigstr.split()
    types_dict = dict(globals(), d=double, i=int_)    
    loc = {}
    if len(parts) < 2:
        signature = eval(sigstr, loc, types_dict)
        signature.name = None
    else:
        signature = eval(' '.join(parts[1:]), loc, types_dict)
        signature.name = parts[0]
    if name is not None:
        signature.name = name
    return signature.name, signature.return_type, signature.args

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
    if isinstance(restype, type):
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
        name, restype, argtypes = _process_sig(restype, kws.get('name', None))
        if name is not None:
            kws['func_name'] = name
    if restype is not None:
        kws['restype'] = restype
    if argtypes is not None:
        kws['argtypes'] = argtypes

    return jit_targets[target, backend](**kws)

def jit2(restype=None, argtypes=None, _llvm_module=None, _llvm_ee=None,
          target='cpu', nopython=False, **kwargs):
    """
    Use the AST translator to translate the function.
    """
    return jit_targets[target, 'ast'](restype, argtypes, nopython=nopython,
                                _llvm_module=_llvm_module, _llvm_ee=_llvm_ee,
                                **kwargs)
