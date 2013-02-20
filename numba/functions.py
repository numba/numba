import ast, inspect, os
import textwrap

from collections import defaultdict
from numba import *
from . import naming
from .minivect import minitypes
import logging
from numba import numbawrapper

logger = logging.getLogger(__name__)

try:
    from meta.decompiler import decompile_func
except Exception as exn:
    def decompile_func(*args, **kwargs):
        raise Exception("Could not import Meta -- Cannot recreate source "
                        "from bytecode")

def fix_ast_lineno(tree):
    # NOTE: A hack to fix assertion error in debug mode due to bad lineno.
    #       Lineno must increase monotonically for co_lnotab,
    #       the "line number table" to work correctly.
    #       This script just set all lineno to 1 and col_offset = to 0.
    #       This makes it impossible to do traceback, but it is not possible
    #       anyway since we are dynamically changing the source code.
    for node in ast.walk(tree):
        # only ast.expr and ast.stmt and their subclass has lineno and col_offset.
        # if isinstance(node,  ast.expr) or isinstance(node, ast.stmt):
        node.lineno = 1
        node.col_offset = 0

    return tree

## Fixme: 
##  This should be changed to visit the AST and fix-up where a None object
##  is present as this will likely not work for all AST.
def _fix_ast(myast):
    import _ast
    # Remove Pass nodes from the end of the ast
    while len(myast.body) > 0  and isinstance(myast.body[-1], _ast.Pass):
        del myast.body[-1]
    # Add a return node at the end of the ast if not present
    if len(myast.body) < 1 or not isinstance(myast.body[-1], _ast.Return):
        name = _ast.Name(id='None',ctx=_ast.Load(), lineno=0, col_offset=0)
        myast.body.append(ast.Return(name))
    # remove _decorator list which sometimes confuses ast visitor
    try:
        indx = myast._fields.index('decorator_list')
    except ValueError:
        return
    else:
        myast.decorator_list = []

def _get_ast(func):
    if int(os.environ.get('NUMBA_FORCE_META_AST', 0)):
        func_def = decompile_func(func)
        assert isinstance(func_def, ast.FunctionDef)
        return func_def
    try:
        source = inspect.getsource(func)
    except IOError:
        return decompile_func(func)
    else:
        source = textwrap.dedent(source)
        # Split off decorators
        decorators = 0
        while not source.startswith('def'): # decorator can have multiple lines
            decorator, sep, source = source.partition('\n')
            decorators += 1
        module_ast = ast.parse(source)

        # fix line numbering
        lineoffset = func.__code__.co_firstlineno + decorators
        ast.increment_lineno(module_ast, lineoffset)

        assert len(module_ast.body) == 1
        func_def = module_ast.body[0]
        _fix_ast(func_def)
        assert isinstance(func_def, ast.FunctionDef)
        return func_def

def _infer_types(context, func, restype=None, argtypes=None, **kwargs):
    import numba.type_inference.infer as type_inference

    ast = _get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
    return type_inference.run_pipeline(context, func, ast,
                                       func_signature, **kwargs)


def _compile(context, func, restype=None, argtypes=None, ctypes=False,
             compile_only=False, name=None, **kwds):
    """
    Compile a numba annotated function.

        - decompile function into a Python ast
        - run type inference using the given input types
        - compile the function to LLVM
    """
    from numba import codegen

    func_signature, symtab, ast = _infer_types(context, func,
                                               restype, argtypes, **kwds)
    func_name = name or naming.specialized_mangle(func.__name__, func_signature.args)
    func_signature.name = func_name

    t = codegen.LLVMCodeGenerator(
        context, func, ast, func_signature=func_signature,
        symtab=symtab, **kwds)
    t.translate()

    if compile_only:
        return func_signature, t.lfunc, None
    if ctypes:
        ctypes_func = t.get_ctypes_func(kwds.get('llvm', True))
        return func_signature, t.lfunc, ctypes_func
    else:
        return func_signature, t.lfunc, t.build_wrapper_function()

live_objects = [] # These are never collected

def keep_alive(py_func, obj):
    """
    Keep an object alive for the lifetime of the translated unit.

    This is a HACK. Make live objects part of the function-cache

    NOTE: py_func may be None, so we can't make it a function attribute
    """
    live_objects.append(obj)

class FunctionCache(object):
    """
    Cache for compiler functions, declared external functions and constants.
    """
    def __init__(self, context):
        self.context = context

        # All numba-compiled functions
        # (py_func) -> (arg_types, flags) -> (signature, llvm_func, ctypes_func)
        self.__compiled_funcs = defaultdict(dict)
        # Faster caches we use directly from autojit to determine the
        # specialization. (py_func) -> (NumbaFunction)
        self.__local_caches = defaultdict(numbawrapper.AutojitFunctionCache)

    def get_function(self, py_func, argtypes, flags):
        '''Get a compiled function in the the function cache.
        The function must not be an external function.
            
        For an external function, is_registered() must return False.
        '''
        result = None

        assert argtypes is not None
        flags = None # TODO: stub
        argtypes_flags = tuple(argtypes), flags
        if py_func in self.__compiled_funcs:
            result = self.__compiled_funcs[py_func].get(argtypes_flags)

        return result

    def get_autojit_cache(self, py_func):
        """
        Get the numbawrapper.AutojitFunctionCache that does a quick lookup
        for the cached case.
        """
        return self.__local_caches[py_func]

    def is_registered(self, func):
        '''Check if a function is registered to the FunctionCache instance.
        '''
        if isinstance(func, numbawrapper.NumbaWrapper):
            return func.py_func in self.__compiled_funcs
        return False

    def register(self, func):
        '''Register a function to the FunctionCache.  

        It is necessary before calling compile_function().
        '''
        return self.__compiled_funcs[func]

    def register_specialization(self, func, compiled, argtypes, flags):
        argtypes_flags = tuple(argtypes), flags
        self.__compiled_funcs[func][argtypes_flags] = compiled

    def compile_function(self, func, argtypes, restype=None,
                         ctypes=False, **kwds):
        """
        Compile a python function given the argument types. Compile only
        if not compiled already, and only if it is registered to the function
        cache.

        Returns a triplet of (signature, llvm_func, python_callable)
        `python_callable` may be the original function, or a ctypes callable
        if the function was compiled.
        """
        # For NumbaFunction, we get the original python function.
        func = getattr(func, 'py_func', func)
        assert func in self.__compiled_funcs, func

        # get the compile flags
        flags = None # stub

        # Search in cache
        result = self.get_function(func, argtypes, flags)
        if result is not None:
            sig, trans, pycall = result
            return sig, trans.lfunc, pycall

        # Compile the function
        from numba import pipeline

        compile_only = getattr(func, '_numba_compile_only', False)
        kwds['compile_only'] = kwds.get('compile_only', compile_only)

        assert kwds.get('llvm_module') is None, kwds.get('llvm_module')

        compiled = pipeline.compile(self.context, func, restype, argtypes,
                                    ctypes=ctypes, **kwds)
        func_signature, translator, ctypes_func = compiled
    
        self.register_specialization(func, compiled, func_signature.args, flags)
        return func_signature, translator.lfunc, ctypes_func
