import ast
import __builtin__ as builtins
try:
    import numbers
except ImportError:
    # pre-2.6
    numbers = None

from numba import error

import logging
logger = logging.getLogger(__name__)

class CooperativeBase(object):
    def __init__(self, *args, **kwargs):
        pass

class NumbaVisitorMixin(CooperativeBase):
    _overloads = None
    def __init__(self, context, func, ast, func_signature=None, nopython=0,
                 symtab=None, **kwargs):
        super(NumbaVisitorMixin, self).__init__(
                                context, func, ast, func_signature,
                                nopython, symtab, **kwargs)
        self.context = context
        self.ast = ast
        self.function_cache = context.function_cache
        self.symtab = symtab
        self.func_signature = func_signature
        self.nopython = nopython

        self.func = func
        self.fco = func.func_code
        self.names = self.global_names = self.fco.co_names
        self.varnames = self.local_names = list(self.fco.co_varnames)
        if self.fco.co_cellvars:
            self.varnames.extend(cellvar for cellvar in self.fco.co_cellvars
                                     if cellvar not in self.varnames)
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code
        self.argnames = self.fco.co_varnames[:self.fco.co_argcount]

        if self.is_closure(func_signature):
            from numba import closure
            self.argnames = (closure.CLOSURE_SCOPE_ARG_NAME,) + self.argnames
            self.varnames.append(closure.CLOSURE_SCOPE_ARG_NAME)

        # Just the globals we will use
        self._myglobals = {}
        for name in self.names:
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                # Assumption here is that any name not in globals or
                # builtins is an attribtue.
                self._myglobals[name] = getattr(builtins, name, None)

        if self._overloads:
            self.visit = self._visit_overload

    def _visit_overload(self, node):
        assert self._overloads

        try:
            return super(NumbaVisitorMixin, self).visit(node)
        except error.NumbaError, e:
            # Try one of the overloads
            cls_name = type(node).__name__
            for i, cls_name in enumerate(self._overloads):
                for overload_name, func in self._overloads[cls_name]:
                    try:
                        return func(self, node)
                    except error.NumbaError, e:
                        if i == len(self._overloads) - 1:
                            raise

        assert False, "unreachable"

    def add_overload(self, visit_name, func):
        assert visit_name.startswith("visit_")
        if not self._overloads:
            self._overloads = {}

        self._overloads.setdefault(visit_name, []).append(func)

    def is_closure(self, func_signature):
        return (func_signature is not None and
                func_signature.args and
                func_signature.args[0].is_closure_scope)

    def error(self, node, msg):
        raise error.NumbaError(node, msg)

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

    def visit_CloneNode(self, node):
        return node


class NumbaVisitor(ast.NodeVisitor, NumbaVisitorMixin):
    "Non-mutating visitor"

    def visitlist(self, list):
        return [self.visit(item) for item in list]

class NumbaTransformer(NumbaVisitorMixin, ast.NodeTransformer):
    "Mutating visitor"

class NoPythonContextMixin(object):

    def visit_WithPythonNode(self, node):
        if not self.nopython:
            raise error.NumbaError(node, "Not in 'with nopython' context")

        self.nopython -= 1
        result = self.visitlist(node.body)
        self.nopython += 1

        return node

    def visit_WithNoPythonNode(self, node):
        if self.nopython:
            raise error.NumbaError(node, "Not in 'with python' context")

        self.nopython += 1
        result = self.visitlist(node.body)
        self.nopython -= 1

        return node
