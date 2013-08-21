# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast, inspect, linecache, os
import logging
import textwrap
from collections import defaultdict

from numba import *
from numba import typesystem
from numba import numbawrapper

import llvm.core

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

def _get_ast(func, flags=0):
    if (int(os.environ.get('NUMBA_FORCE_META_AST', 0)) or
                func.__name__ == '<lambda>'):
        func_def = decompile_func(func)
        if isinstance(func_def, ast.Lambda):
            func_def = ast.FunctionDef(name='<lambda>', args=func_def.args,
                                       body=[ast.Return(func_def.body)],
                                       decorator_list=[])
        assert isinstance(func_def, ast.FunctionDef)
        return func_def
    try:
        linecache.checkcache(inspect.getsourcefile(func))
        source = inspect.getsource(func)
        source_module = inspect.getmodule(func)
    except IOError:
        return decompile_func(func)
    else:
        # Split off decorators
        # TODO: This is not quite correct, we can have comments or strings
        # starting at column 0 and an indented function !
        source = textwrap.dedent(source)
        decorators = 0
        while not source.lstrip().startswith('def'): # decorator can have multiple lines
            assert source
            decorator, sep, source = source.partition('\n')
            decorators += 1
        if (hasattr(source_module, "print_function") and
                hasattr(source_module.print_function, "compiler_flag")):
            flags |= source_module.print_function.compiler_flag
        source_file = getattr(source_module, '__file__', '<unknown file>')
        module_ast = compile(source, source_file, "exec",
                             ast.PyCF_ONLY_AST | flags, True)

        lineoffset = func.__code__.co_firstlineno + decorators - 1
        ast.increment_lineno(module_ast, lineoffset)

        assert len(module_ast.body) == 1
        func_def = module_ast.body[0]
        _fix_ast(func_def)
        assert isinstance(func_def, ast.FunctionDef)
        return func_def

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
    def __init__(self, context=None, env=None):
        self.context = context
        self.env = env

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

    def register_specialization(self, func_env):
        func = func_env.func
        argtypes = func_env.func_signature.args
        compiled = (
            func_env.func_signature,
            func_env.lfunc,
            func_env.numba_wrapper_func,
        )

        # Sanity check
        assert isinstance(func_env.func_signature, typesystem.function)
        assert isinstance(func_env.lfunc, llvm.core.Function)

        argtypes_flags = tuple(argtypes), None
        self.__compiled_funcs[func][argtypes_flags] = compiled
