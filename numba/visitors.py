import ast
import __builtin__ as builtins

class NumbaVisitorMixin(object):
    def __init__(self, context, func, ast):
        self.context = context
        self.ast = ast

        self.func = func
        self.fco = func.func_code
        self.names = self.global_names = self.fco.co_names
        self.varnames = self.local_names = self.fco.co_varnames
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code

        # Just the globals we will use
        self._myglobals = {}
        for name in self.names:
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                # Assumption here is that any name not in globals or
                # builtins is an attribtue.
                self._myglobals[name] = getattr(builtins, name, None)

    def visitlist(self, list):
        newlist = []
        for node in list:
            result = self.visit(node)
            if result is not None:
                newlist.append(result)

        list[:] = newlist
        return list

class NumbaVisitor(ast.NodeVisitor, NumbaVisitorMixin):
    "Non-mutating visitor"

class NumbaTransformer(ast.NodeTransformer, NumbaVisitorMixin):
    "Mutating visitor"