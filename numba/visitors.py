import ast
from pprint import pprint, pformat
import ast as ast_module
import __builtin__ as builtins
from numba import functions
from numba import nodes
from numba.typesystem.typemapper import have_properties

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
    func_level = 0

    def __init__(self, context, func, ast, locals=None,
                 func_signature=None, nopython=0,
                 symtab=None, **kwargs):

        assert locals is not None

        super(NumbaVisitorMixin, self).__init__(
            context, func, ast, func_signature=func_signature,
            nopython=nopython, symtab=symtab, **kwargs)

        self.context = context
        self.ast = ast
        self.function_cache = context.function_cache
        self.symtab = symtab
        self.func_signature = func_signature
        self.nopython = nopython
        self.llvm_module = kwargs.pop('llvm_module', None)
        self.locals = locals
        #self.local_scopes = [self.symtab]
        self.current_scope = symtab
        self.have_cfg = getattr(self.ast, 'flow', False)
        self.closures = kwargs.get('closures')
        self.is_closure = kwargs.get('is_closure', False)
        self.kwargs = kwargs

        if self.have_cfg:
            self.flow_block = self.ast.flow.blocks[1]
        else:
            self.flow_block = None

        self.func = func
        if func is None or not getattr(func, "__numba_valid_code_object", True):
            assert isinstance(ast, ast_module.FunctionDef)
            locals, cellvars, freevars = determine_variable_status(context, ast,
                                                                   self.locals)
            self.names = self.global_names = freevars
            self.argnames = tuple(arg.id for arg in ast.args.args)
            argnames = set(self.argnames)
            local_names = [local_name for local_name in locals
                                          if local_name not in argnames]
            self.varnames = self.local_names = list(self.argnames) + local_names

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

            self.cellvars = set(f_code.co_cellvars)
            self.freevars = set(f_code.co_freevars)

        if func is None:
            self.func_globals = kwargs.get('func_globals', {})
            self.module_name = self.func_globals.get("__name__", "")
        else:
            self.func_globals = func.func_globals
            self.module_name = self.func.__module__

        # Add variables declared in locals=dict(...)
        self.local_names.extend(
                local_name for local_name in self.locals
                               if local_name not in self.local_names)

        if self.is_closure_signature(func_signature) and func is not None:
            # If a closure is backed up by an actual Python function, the
            # closure scope argument is absent
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

    @property
    def func_name(self):
        if "func_name" in self.kwargs:
            return self.kwargs["func_name"]
        return self.ast.name

    @property
    def func_doc(self):
        if self.func is not None:
            return self.func.__doc__
        else:
            return ast.get_docstring(self.ast)

    @property
    def qualified_name(self):
        qname = self.kwargs.get("qualified_name", None)
        if qname is None:
            qname = "%s.%s" % (self.module_name, self.func_name)
        return qname

    def visit_func_children(self, node):
        self.func_level += 1
        self.generic_visit(node)
        self.func_level -= 1
        return node

    def invalidate_locals(self, ast=None):
        ast = ast or self.ast
        if hasattr(ast, "variable_status_tuple"):
            del ast.variable_status_tuple
        if self.func and ast is self.ast:
            setattr(self.func, "__numba_valid_code_object", False)

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

    def is_closure_signature(self, func_signature):
        return (func_signature is not None and
                func_signature.args and
                func_signature.args[0].is_closure_scope)

    def run_template(self, s, vars=None, **substitutions):
        from numba import templating

        templ = templating.TemplateContext(self.context, s)
        if vars:
            for name, type in vars.iteritems():
                templ.temp_var(name, type)

        symtab, tree = templ.template_type_infer(
                substitutions, symtab=self.symtab,
                closure_scope=getattr(self.ast, "closure_scope", None))
        self.symtab.update(templ.get_vars_symtab())
        return tree

    def error(self, node, msg):
        raise error.NumbaError(node, msg)

    def keep_alive(self, obj):
        """
        Keep an object alive for the lifetime of the translated unit.

        This is a HACK. Make live objects part of the function-cache
        """
        functions.keep_alive(self.func, obj)

    def have(self, t1, t2, p1, p2):
        """
        Return whether the two variables have the indicated properties:

            >>> have(int_, float_, "is_float", "is_int")
            float_

        If true, returns the type indicated by the first property.
        """
        return have_properties(t1, t2, p1, p2)

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

    def register_assignment(self, node, target, operator):
        if isinstance(target, nodes.MaybeUnusedNode):
            target = target.name_node
        if isinstance(target, ast.Name):
            self.assigned[target.id] = node

    def visit_Assign(self, node):
        self.generic_visit(node)
        op = getattr(node, "inplace_op", None)
        self.register_assignment(node, node.targets[0], op)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        self.register_assignment(node, node.target, node.op)

    def visit_For(self, node):
        self.generic_visit(node)
        self.register_assignment(node, node.target, None)

    def visit_Name(self, node):
        self.referenced[node.id] = node

    def visit_FunctionDef(self, node):
        if self.function_level == 0:
            self.function_level += 1
            self.generic_visit(node)
            self.function_level -= 1
        else:
            self.func_defs.append(node)

        return node

    def visit_ClosureNode(self, node):
        self.func_defs.append(node)
        self.generic_visit(node)
        return node

def determine_variable_status(context, ast, locals_dict):
    """
    Determine what category referenced and assignment variables fall in:

        - local variables
        - free variables
        - cell variables
    """
    from numba import pipeline

    if hasattr(ast, 'variable_status_tuple'):
        return ast.variable_status_tuple

    v = VariableFindingVisitor()
    v.visit(ast)

    locals = set(v.assigned)
    locals.update(locals_dict)
    locals.update(arg.id for arg in ast.args.args)
    locals.update(func_def.name for func_def in v.func_defs)

    freevars = set(v.referenced) - locals
    cellvars = set()

    # Compute cell variables
    for func_def in v.func_defs:
        inner_locals_dict = pipeline.get_locals(func_def, None)

        inner_locals, inner_cellvars, inner_freevars = \
                            determine_variable_status(context, func_def,
                                                      inner_locals_dict)
        cellvars.update(locals.intersection(inner_freevars))

#    print ast.name, "locals", pformat(locals),      \
#                    "cellvars", pformat(cellvars),  \
#                    "freevars", pformat(freevars),  \
#                    "locals_dict", pformat(locals_dict)
#    print ast.name, "locals", pformat(locals)

    # Cache state
    ast.variable_status_tuple = locals, cellvars, freevars
    return locals, cellvars, freevars
