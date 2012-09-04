
import ast
import __builtin__ as builtins
try:
    import numbers
except ImportError:
    # pre-2.6
    numbers = None

from numba.pymothoa import compiler_errors

class NumbaVisitorMixin(object):
    def __init__(self, context, func, ast):
        self.context = context
        self.ast = ast
        self.function_cache = context.function_cache

        self.func = func
        self.fco = func.func_code
        self.names = self.global_names = self.fco.co_names
        self.varnames = self.local_names = self.fco.co_varnames
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code
        self.argnames = self.fco.co_varnames[:self.fco.co_argcount]

        # Just the globals we will use
        self._myglobals = {}
        for name in self.names:
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                # Assumption here is that any name not in globals or
                # builtins is an attribtue.
                self._myglobals[name] = getattr(builtins, name, None)

    def error(self, node, msg):
        raise compiler_errors.CompilerError(node, msg)

    def visitlist(self, list):
        newlist = []
        for node in list:
            result = self.visit(node)
            if result is not None:
                newlist.append(result)

        list[:] = newlist
        return list

    def is_complex(self, n):
        if numbers:
            return isinstance(n, numbers.Complex)
        return isinstance(n, complex)

    def is_real(self, n):
        if numbers:
            return isinstance(n, numbers.Real)
        return isinstance(n, float)

    def is_int(self, n):
        if numbers:
            return isinstance(n, numbers.Int)
        return isinstance(n, (int, long))

class NumbaVisitor(ast.NodeVisitor, NumbaVisitorMixin):
    "Non-mutating visitor"

    def visitlist(self, list):
        return [self.visit(item) for item in list]

class NumbaTransformer(ast.NodeTransformer, NumbaVisitorMixin):
    "Mutating visitor"