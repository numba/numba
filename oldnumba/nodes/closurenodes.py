# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *

# not really an expression, but used in an assignment
class ClosureNode(ExprNode):
    """
    Inner functions or closures.

    When coerced to an object, a wrapper PyMethodDef gets created, and at
    call time a function is dynamically created with the closure scope.

        func_def:
            AST FunctionDef of the function
        closure_type:
            numba.typesystem.ClosureType
        outer_py_func:
            Outer Python function (or None!)
    """

    _fields = []

    def __init__(self, env, func_def, closure_type, outer_py_func, **kwargs):
        super(ClosureNode, self).__init__(**kwargs)
        self.func_def = func_def
        self.type = closure_type
        self.outer_py_func = outer_py_func
        self.name = self.func_def.name

        func_env = env.translation.get_env(func_def)
        self.need_numba_func = not func_env or func_env.need_closure_wrapper
        self.lfunc = None
        self.wrapper_func = None
        self.wrapper_lfunc = None
        self.lfunc_pointer = None

        # FunctionEnvironment after type inference
        self.func_env = None
        # self.type_inferred_ast = None
        # self.symtab = None

        from numba import pipeline
        self.locals = pipeline.get_locals(func_def, None)

        # The Python extension type that must be instantiated to hold cellvars
        # self.scope_type = None
        self.ext_type = None
        self.need_closure_scope = False

    def make_pyfunc(self):
        d = self.outer_py_func.__globals__
#        argnames = tuple(arg.id for arg in self.func_def.args.args)
#        dummy_func_string = """
#def __numba_closure_func(%s):
#    pass
#        """ % ", ".join(argnames)
#        exec dummy_func_string in d, d

        # Something set a pure and original, unmodified, AST, use that
        # instead and reset it after the compile. This is a HACK
        func_body = self.func_def.body
        if hasattr(self.func_def, 'pure_ast_body'):
            self.func_def.body = self.func_def.pure_ast_body

        name = self.func_def.name
        self.func_def.name = '__numba_closure_func'
        ast_mod = ast.Module(body=[self.func_def])
        numba.functions.fix_ast_lineno(ast_mod)
        c = compile(ast_mod, '<string>', 'exec')
        exec(c, d, d)
        self.func_def.name = name

        self.py_func = d['__numba_closure_func']
        self.py_func.live_objects = []
        self.py_func.__module__ = self.outer_py_func.__module__
        self.py_func.__name__ = name

        if hasattr(self.func_def, 'pure_ast_body'):
            self.func_def.body = func_body

class InstantiateClosureScope(ExprNode):

    _fields = ['outer_scope']

    def __init__(self, func_def, scope_ext_type, scope_type, outer_scope, **kwargs):
        super(InstantiateClosureScope, self).__init__(**kwargs)
        self.func_def = func_def
        self.scope_type = scope_type
        self.ext_type = scope_ext_type
        self.outer_scope = outer_scope
        self.type = scope_type

class ClosureScopeLoadNode(ExprNode):
    "Load the closure scope for the function or NULL"

    type = void.pointer()

class ClosureCallNode(NativeCallNode):
    """
    Call to closure or inner function.
    """

    _fields = ['func', 'args']

    def __init__(self, closure_type, call_node, **kwargs):
        self.call_node = call_node
        self.func = call_node.func
        self.closure_type = closure_type

        self.argnames = [name.id for name in self.func_def.args.args[self.need_closure_scope:]]
        self.expected_nargs = len(self.argnames)

        args, keywords = call_node.args, call_node.keywords
        args = args + self._resolve_keywords(args, keywords)
        super(ClosureCallNode, self).__init__(
                closure_type.signature, args, llvm_func=None,
                skip_self=self.need_closure_scope, **kwargs)

    @property
    def need_closure_scope(self):
        return self.closure_type.closure.need_closure_scope

    @property
    def func_def(self):
        return self.closure_type.closure.func_def

    def _resolve_keywords(self, args, keywords):
        "Map keyword arguments to positional arguments"
        expected = self.expected_nargs - len(args)
        if len(keywords) != expected:
            raise error.NumbaError(
                    self.call_node,
                    "Expected %d arguments, got %d" % (self.expected_nargs,
                                                       len(args) + len(keywords)))

        argpositions = dict(zip(self.argnames, range(self.expected_nargs)))
        positional = [None] * (self.expected_nargs - len(args))

        for keyword in keywords:
            argname = keyword.arg
            pos = argpositions.get(argname, None)
            if pos is None:
                raise error.NumbaError(
                        keyword, "Not a valid keyword argument name: %s" % argname)
            elif pos < len(args):
                raise error.NumbaError(
                        keyword, "Got multiple values for positional "
                                 "argument %r" % argname)
            else:
                positional[pos] = keyword.value

        return positional
