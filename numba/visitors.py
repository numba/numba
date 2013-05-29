# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast
import ast as ast_module

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

import types

from numba.traits import traits, Delegate
from numba import functions, PY3
from numba import nodes
from numba.nodes.metadata import annotate, query
from numba.typesystem.typemapper import have_properties

from numba import error

import logging
logger = logging.getLogger(__name__)

@traits
class NumbaStatefulVisitor(object):

    func_level = 0

    func = Delegate("func_env")
    function_cache = Delegate("context")
    symtab = Delegate("func_env")
    func_signature = Delegate("func_env")
    locals = Delegate("func_env")
    llvm_module = Delegate("func_env")
    local_scopes = Delegate("func_env")
    current_scope = Delegate("func_env")
    closures = Delegate("func_env")
    is_closure = Delegate("func_env")
    kwargs = Delegate("func_env")
    func_globals = Delegate("func_env", "function_globals")
    module_name = Delegate("func_env")

    ir = ast_module

    def __init__(self, context, func, ast, env, **kwargs):
        self.env = env
        self.context = context
        self.ast = ast
        self.func_env = env.crnt
        self.nopython = env.translation.nopython

        # TODO: Hargh. Remove and track locals and cellvars explicitly
        if not self.valid_locals(func):
            assert isinstance(ast, ast_module.FunctionDef)
            locals, cellvars, freevars = determine_variable_status(
                self.env, ast, self.locals)

            self.argnames = tuple(name.id for name in ast.args.args)

            argnames = set(self.argnames)
            local_names = [local_name for local_name in locals
                                          if local_name not in argnames]
            self.varnames = self.local_names = list(self.argnames) + local_names

            self.cellvars = cellvars
            self.freevars = freevars
        else:
            f_code = self.func.__code__
            self.varnames = self.local_names = list(f_code.co_varnames)

            if PY3:
                def recurse_co_consts(fco):
                    for _var in fco.co_consts:
                        if not isinstance(_var, types.CodeType):
                            continue
                        self.varnames.extend((_name for _name in _var.co_varnames
                                              if not _name.startswith('.')))
                        recurse_co_consts(_var)

                recurse_co_consts(f_code)

            self.argnames = self.varnames[:f_code.co_argcount]

            if f_code.co_cellvars:
                self.varnames.extend(
                    cellvar for cellvar in f_code.co_cellvars
                    if cellvar not in self.varnames)

            self.cellvars = set(f_code.co_cellvars)
            self.freevars = set(f_code.co_freevars)

        # Add variables declared in locals=dict(...)
        self.local_names.extend(
                local_name for local_name in self.locals
                               if local_name not in self.local_names)

        if self.is_closure_signature(self.func_signature) and func is not None:
            # If a closure is backed up by an actual Python function, the
            # closure scope argument is absent
            from numba import closures
            self.argnames = (closures.CLOSURE_SCOPE_ARG_NAME,) + self.argnames
            self.varnames.append(closures.CLOSURE_SCOPE_ARG_NAME)

        self.visitchildren = self.generic_visit

    #------------------------------------------------------------------------
    # Visit methods
    #------------------------------------------------------------------------

    def visit(self, node):
        """Visit a node."""
        if node is not None:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            return visitor(node)

    #------------------------------------------------------------------------
    # Remove these
    #------------------------------------------------------------------------

    @property
    def func_name(self):
        if "func_name" in self.kwargs:
            return self.kwargs["func_name"]
        return self.ast.name

    @property
    def qualified_name(self):
        qname = self.kwargs.get("qualified_name", None)
        if qname is None:
            qname = "%s.%s" % (self.module_name, self.func_name)
        return qname

    #------------------------------------------------------------------------
    # AST annotations -- this needs more thought
    #------------------------------------------------------------------------

    def annotate(self, node, key, value):
        annotate(self.env, node, key, value)

    def query(self, node, key):
        return query(self.env, node, key)

    #------------------------------------------------------------------------
    # Error messages
    #------------------------------------------------------------------------

    def error(self, node, msg):
        "Issue a terminating error"
        raise error.NumbaError(node, msg)

    def deferred_error(self, node, msg):
        "Issue a deferred-terminating error"
        self.func_env.error_env.collection.error(node, msg)

    def warn(self, node, msg):
        "Issue a warning"
        self.func_env.error_env.collection.warning(node, msg)

    def visit_func_children(self, node):
        self.func_level += 1
        self.generic_visit(node)
        self.func_level -= 1
        return node

    #------------------------------------------------------------------------
    # Locals Invalidation -- remove
    #------------------------------------------------------------------------

    # TODO: Explicitly track locals, freevars, cellvars (and remove the below)

    def valid_locals(self, func):
        if self.ast is None or self.env is None:
            return True

        return (func is not None and
                query(self.env, self.ast, "__numba_valid_code_object",
                      default=True))

    def invalidate_locals(self, ast=None):
        ast = ast or self.ast
        if query(self.env, ast, "variable_status_tuple"):
            # Delete variable status of the function (local/free/cell status)
            annotate(self.env, ast, variable_status_tuple=None)

        if self.func and ast is self.ast:
            # Invalidate validity of code object
            annotate(self.env, ast, __numba_valid_code_object=False)

    def is_closure_signature(self, func_signature):
        from numba import closures
        return closures.is_closure_signature(func_signature)

    #------------------------------------------------------------------------
    # Templating -- outline
    #------------------------------------------------------------------------

    def run_template(self, s, vars=None, **substitutions):
        # TODO: make this not a method
        from numba import templating

        func = self.func
        if func is None:
            d = dict(self.func_globals)
            exec('def __numba_func(): pass', d, d)
            func = d['__numba_func']

        templ = templating.TemplateContext(self.context, s, env=self.env)

        if vars:
            for name, type in vars.iteritems():
                templ.temp_var(name, type)

        symtab, tree = templ.template_type_infer(
                substitutions, symtab=self.symtab,
                closure_scope=getattr(self.ast, "closure_scope", None),
                func=func)
        self.symtab.update(templ.get_vars_symtab())
        return tree

    #------------------------------------------------------------------------
    # This should go away
    #------------------------------------------------------------------------

    def visit_CloneNode(self, node):
        return node

    def visit_ControlBlock(self, node):
        self.setblock(node)
        self.visitlist(node.phi_nodes)
        self.visitlist(node.body)
        return node

    flow_block = None

    def setblock(self, cfg_basic_block):
        if cfg_basic_block.is_fabricated:
            return

        old = self.flow_block
        self.flow_block = cfg_basic_block

        if old is not cfg_basic_block:
            self.current_scope = cfg_basic_block.symtab

    #------------------------------------------------------------------------
    # Utilities
    #------------------------------------------------------------------------

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

    def handle_phis(self, reversed=False):
        # TODO: Remove this
        blocks = self.env.crnt.cfg.blocks
        if reversed:
            blocks = blocks[::-1]
        for block in blocks:
            for phi_node in block.phi_nodes:
                self.handle_phi(phi_node)

    #------------------------------------------------------------------------
    # nopython
    #------------------------------------------------------------------------

    def visit_WithPythonNode(self, node, errorcheck=True):
        if not self.nopython and errorcheck:
            raise error.NumbaError(node, "Not in 'with nopython' context")

        self.nopython -= 1
        self.visitlist(node.body)
        self.nopython += 1

        return node

    def visit_WithNoPythonNode(self, node, errorcheck=True):
        if self.nopython and errorcheck:
            raise error.NumbaError(node, "Not in 'with python' context")

        self.nopython += 1
        self.visitlist(node.body)
        self.nopython -= 1

        return node


class NumbaVisitor(NumbaStatefulVisitor):
    "Non-mutating visitor"

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visitlist(self, list):
        return [self.visit(item) for item in list]

class NumbaTransformer(NumbaStatefulVisitor):
    "Mutating visitor"

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visitlist(self, list):
        newlist = []
        for node in list:
            result = self.visit(node)
            if result is not None:
                newlist.append(result)

        list[:] = newlist
        return list

#---------------------------------------------------------------------------
# Track variables and mark assignments
#---------------------------------------------------------------------------

# TODO: Move this elsewhere

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

def determine_variable_status(env, ast, locals_dict):
    """
    Determine what category referenced and assignment variables fall in:

        - local variables
        - free variables
        - cell variables
    """
    variable_status = query(env, ast, 'variable_status_tuple')
    if variable_status:
        return variable_status

    v = VariableFindingVisitor()
    v.visit(ast)

    locals = set(v.assigned)
    locals.update(locals_dict)

    locals.update([name.id for name in ast.args.args])

    locals.update(func_def.name for func_def in v.func_defs)

    freevars = set(v.referenced) - locals
    cellvars = set()

    # Compute cell variables
    for func_def in v.func_defs:
        func_env = env.translation.make_partial_env(func_def, locals={})
        inner_locals_dict = func_env.locals

        inner_locals, inner_cellvars, inner_freevars = \
                            determine_variable_status(env, func_def,
                                                      inner_locals_dict)
        cellvars.update(locals.intersection(inner_freevars))

#    print ast.name, "locals", pformat(locals),      \
#                    "cellvars", pformat(cellvars),  \
#                    "freevars", pformat(freevars),  \
#                    "locals_dict", pformat(locals_dict)
#    print ast.name, "locals", pformat(locals)

    # Cache state
    annotate(env, ast, variable_status_tuple=(locals, cellvars, freevars))
    return locals, cellvars, freevars
