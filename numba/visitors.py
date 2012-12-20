import ast
import ast as ast_module
import __builtin__ as builtins
try:
    import numbers
except ImportError:
    # pre-2.6
    numbers = None

from numba import error
from numba import _numba_types as numba_types

import logging
logger = logging.getLogger(__name__)

class CooperativeBase(object):
    def __init__(self, *args, **kwargs):
        pass

class NumbaVisitorMixin(CooperativeBase):

    _overloads = None

    def __init__(self, context, func, ast, func_signature=None, nopython=0,
                 symtab=None, locals=None, **kwargs):
        super(NumbaVisitorMixin, self).__init__(
                                context, func, ast, func_signature,
                                nopython, symtab, **kwargs)
        self.context = context
        self.ast = ast
        self.function_cache = context.function_cache
        self.symtab = symtab
        self.func_signature = func_signature
        self.nopython = nopython
        self.locals = locals or {}
        #self.local_scopes = [self.symtab]
        self.current_scope = symtab
        self.have_cfg = getattr(self.ast, 'flow', False)

        if self.have_cfg:
            self.flow_block = self.ast.flow.blocks[1]
        else:
            self.flow_block = None

        self.func = func
        if func is None:
            assert isinstance(ast, ast_module.FunctionDef)
            locals, cellvars, freevars = determine_variable_status(context, ast)
            self.names = self.global_names = freevars
            self.argnames = [arg.id for arg in ast.args.args]
            argnames = set(self.argnames)
            local_names = [local_name for local_name in locals
                                          if local_name not in argnames]
            self.varnames = self.local_names = self.argnames + local_names
            self.func_globals = kwargs.get('func_globals', {})

            self.cellvars = cellvars
            self.freevars = freevars
        else:
            f_code = self.func.func_code
            self.names = self.global_names = f_code.co_names
            self.varnames = self.local_names = list(f_code.co_varnames)

            if f_code.co_cellvars:
                self.varnames.extend(
                        cellvar for cellvar in f_code.co_cellvars
                                    if cellvar not in self.varnames)

            self.argnames = f_code.co_varnames[:f_code.co_argcount]
            self.func_globals = func.func_globals

            self.cellvars = set(f_code.co_cellvars)
            self.freevars = set(f_code.co_freevars)

        # Add variables declared in locals=dict(...)
        self.local_names.extend(
                local_name for local_name in self.locals
                               if local_name not in self.local_names)

        if self.is_closure(func_signature):
            from numba import closure
            self.argnames = (closure.CLOSURE_SCOPE_ARG_NAME,) + self.argnames
            self.varnames.append(closure.CLOSURE_SCOPE_ARG_NAME)

        # Just the globals we will use
        self._myglobals = {}
        for name in self.names:
            try:
                self._myglobals[name] = self.func_globals[name]
            except KeyError:
                # Assumption here is that any name not in globals or
                # builtins is an attribtue.
                self._myglobals[name] = getattr(builtins, name, None)

        if self._overloads:
            self.visit = self._visit_overload

        self.visitchildren = self.generic_visit

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

    def run_template(self, s, vars=None, **substitutions):
        from numba import templating

        templ = templating.TemplateContext(self.context, s)
        if vars:
            for name, type in vars.iteritems():
                templ.temp_var(name, type)

        symtab, tree = templ.template_type_infer(substitutions,
                                                 symtab=self.symtab)
        self.symtab.update(templ.get_vars_symtab())
        return tree

    def error(self, node, msg):
        raise error.NumbaError(node, msg)


    def have(self, t1, t2, p1, p2):
        """
        Return whether the two variables have the indicated properties:

            >>> have(int_, float_, "is_float", "is_int")
            float_

        If true, returns the type indicated by the first property.
        """
        return numba_types.have_properties(t1, t2, p1, p2)

    def have_types(self, v1, v2, p1, p2):
        return self.have(v1.type, v2.type, p1, p2)

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

    #@property
    #def current_scope(self):
    #    return self.local_scopes[-1]

    def visit_ControlBlock(self, node):
        #self.local_scopes.append(node.symtab)
        self.setblock(node)
        self.visitlist(node.phi_nodes)
        self.visitlist(node.body)
        #self.local_scopes.pop()
        return node

    def setblock(self, cfg_basic_block):
        if cfg_basic_block.is_fabricated:
            return

        old = self.flow_block
        self.flow_block = cfg_basic_block

        if old is not cfg_basic_block:
            self.current_scope = cfg_basic_block.symtab

        self.changed_block(old, cfg_basic_block)

    def changed_block(self, old_block, new_block):
        """
        Callback for when a new cfg block is encountered.
        """

    def handle_phis(self):
        for block in self.ast.flow.blocks:
            for phi_node in block.phi_nodes:
                self.handle_phi(phi_node)

    @property
    def type(self):
        assert self.is_expr and len(node.body) == 1
        return node.body[0].type

    @property
    def variable(self):
        assert self.is_expr and len(node.body) == 1
        return node.body[0].variable

x = 0
for i in range(10):
    if x > 1:
        print x
    y = 14.2
    x = x * y

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

class VariableFindingVisitor(NumbaVisitor):
    "Find referenced and assigned ast.Name nodes"

    function_level = 0

    def __init__(self, *args, **kwargs):
        self.referenced = {}
        self.assigned = {}
        self.func_defs = []

    def register_assignment(self, node, target):
        if isinstance(target, ast.Name):
            self.assigned[node.id] = node

    def visit_Assign(self, node):
        self.generic_visit(node)
        self.register_assignment(node, node.targets[0])

    def visit_For(self, node):
        self.generic_visit(node)
        self.register_assignment(node, node.target)

    def visit_Name(self, node):
        self.referenced[node.id] = node

    def visit_FunctionDef(self, node):
        if self.function_level == 0:
            self.generic_visit(node)
        else:
            self.func_defs.append(node)

        return node

def determine_variable_status(context, ast):
    """
    Determine what category referenced and assignment variables fall in:

        - local variables
        - free variables
        - cell variables
    """
    if hasattr(ast, 'variable_status_tuple'):
        return ast.variable_status_tuple

    v = VariableFindingVisitor(context, None, ast)
    v.visit(ast)

    locals = set(v.assigned)
    freevars = set(v.referenced) - locals
    cellvars = set()

    # Compure cell variables
    for func_def in v.func_defs:
        inner_locals, inner_cellvars, inner_freevars = \
                            determine_variable_status(context, func_def)
        cellvars.update(locals.intersection(inner_freevars))

    # Cache state
    ast.variable_status_tuple = locals, cellvars, freevars
    return locals, freevars