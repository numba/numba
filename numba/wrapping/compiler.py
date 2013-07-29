import inspect

from numba import typesystem
import numba.pipeline
from numba.exttypes import virtual
import numba.exttypes.entrypoints

import numba.decorators
from numba import functions

def resolve_argtypes(env, py_func, template_signature,
                     args, kwargs, translator_kwargs):
    """
    Given an autojitting numba function, return the argument types.
    These need to be resolved in order for the function cache to work.

    TODO: have a single entry point that resolves the argument types!
    """
    assert not kwargs, "Keyword arguments are not supported yet"

    locals_dict = translator_kwargs.get("locals", None)

    argcount = py_func.__code__.co_argcount
    if argcount != len(args):
        if argcount == 1:
            arguments = 'argument'
        else:
            arguments = 'arguments'
        raise TypeError("%s() takes exactly %d %s (%d given)" % (
                                py_func.__name__, argcount,
                                arguments, len(args)))

    return_type = None
    argnames = inspect.getargspec(py_func).args
    argtypes = [typesystem.numba_typesystem.typeof(x) for x in args]

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

    return typesystem.function(return_type, tuple(argtypes))

class Compiler(object):

    def __init__(self, env, py_func, nopython, flags, template_signature):
        self.env = env
        self.py_func = py_func
        self.nopython = nopython
        self.flags = flags
        self.target = flags.pop('target', 'cpu')
        self.template_signature = template_signature

    def resolve_argtypes(self, args, kwargs):
        signature = resolve_argtypes(self.env, self.py_func,
                                     self.template_signature,
                                     args, kwargs, self.flags)
        return signature

    def compile_from_args(self, args, kwargs):
        signature = self.resolve_argtypes(args, kwargs)
        return self.compile(signature)

    def compile(self, signature):
        "Compile the Python function with the given signature"

class FunctionCompiler(Compiler):

    def __init__(self, env, py_func, nopython, flags, template_signature):
        super(FunctionCompiler,self).__init__(env, py_func, nopython, flags, template_signature)
        self.ast = functions._get_ast(py_func)

    def compile(self, signature):
        jitter = numba.decorators.jit_targets[(self.target, 'ast')]

        dec = jitter(restype=signature.return_type,
                     argtypes=signature.args,
                     target=self.target, nopython=self.nopython,
                     env=self.env, func_ast=self.ast, **self.flags)

        compiled_function = dec(self.py_func)
        return compiled_function

class ClassCompiler(Compiler):

    def resolve_argtypes(self, args, kwargs):
        assert not kwargs
        # argtypes = map(self.env.crnt.typesystem.typeof, args)
        argtypes = map(numba.typeof, args) # TODO: allow registering a type system and using it here
        signature = typesystem.function(None, argtypes)
        return signature

    def compile(self, signature):
        py_class = self.py_func
        return numba.exttypes.entrypoints.autojit_extension_class(
            self.env, py_class, self.flags, signature.args)

#------------------------------------------------------------------------
# Autojit Method Compiler
#------------------------------------------------------------------------

class MethodCompiler(Compiler):

    def __init__(self, env, extclass, method, flags=None):
        super(MethodCompiler, self).__init__(env, method.py_func,
                                             method.nopython, flags or {},
                                             method.template_signature)
        self.extclass = extclass
        self.method = method

    def compile(self, signature):
        from numba.exttypes.autojitclass import autojit_method_compiler
        return autojit_method_compiler(
            self.env, self.extclass, self.method, signature)
