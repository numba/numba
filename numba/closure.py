"""
This module provides support for closures and inner functions.

@autojit
def outer():
    a = 10 # this is a cellvar

    @jit('void()')
    def inner():
        print a # this is a freevar

    inner()
    a = 12
    return inner

The 'inner' function closes over the outer scope. Each function with
cellvars packs them into a heap-allocated structure, the closure scope.

The closure scope is passed into 'inner' when called from within outer.

The execution of 'def' creates a NumbaFunction, which has itself as the
 m_self attribute. So when 'inner' is invoked from Python, the numba
 wrapper function gets called with NumbaFunction object and the args
 tuple. The closure scope is then set in NumbaFunction.func_closure.

The closure scope is an extension type with the cellvars as attributes.
Closure scopes are chained together, since multiple inner scopes may need
to share a single outer scope. E.g.

    def outer(a):
        def inner(b):
            def closure():
                print a, b
            return closure

        return inner(1), inner(2)

We have three closure scopes here:

    scope_outer = { 'a': a }
    scope_inner_1 = { 'scope_outer': scope_outer, 'b': 1 }
    scope_inner_2 = { 'scope_outer': scope_outer, 'b': 2 }

These scopes are instances of a numba extension class.
"""

import ast
import types
import ctypes

import numba.decorators
from numba import *
from numba import error, visitors, nodes
from numba.minivect import  minitypes
from numba import  _numba_types as numba_types
from numba.symtab import Variable

import logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class ClosureMixin(object):
    """
    Handles closures during type inference. Mostly performs error checking
    for closure signatures.

    Generates ClosureNodes that hold inner functions. When visited, they
    do not recurse into the inner functions themselves!
    """

    function_level = 0

    def _visit_func_children(self, node):
        self.function_level += 1
        self.generic_visit(node)
        self.function_level -= 1
        return node

    def _err_decorator(self, decorator):
        raise error.NumbaError(
                decorator, "Only @jit and @autojit decorators are supported")

    def _check_valid_argtype(self, argtype_node, argtype):
        if not isinstance(argtype, minitypes.Type):
            raise error.NumbaError(argtype_node, "Invalid type: %r" % (argtype,))

    def _assert_constant(self, decorator, result_node):
        result = self.visit(result_node)
        if not result.variable.is_constant:
            raise error.NumbaError(decorator, "Expected a constant")

        return result.variable.constant_value

    def _parse_argtypes(self, decorator, func_def, jit_args):
        argtypes_node = jit_args['argtypes']
        if argtypes_node is None:
            raise error.NumbaError(func_def.args[0],
                                   "Expected an argument type")

        argtypes = self._assert_constant(decorator, argtypes_node)

        if not isinstance(argtypes, (list, tuple)):
            raise error.NumbaError(argtypes_node,
                                   'Invalid argument for argtypes')
        for argtype in argtypes:
            self._check_valid_argtype(argtypes_node, argtype)

        return argtypes

    def _parse_restype(self, decorator, jit_args):
        restype_node = jit_args['restype']

        if restype_node is not None:
            restype = self._assert_constant(decorator, restype_node)
            if isinstance(restype, (str, unicode)):
                name, restype, argtypes = numba.decorators._process_sig(restype)
                self._check_valid_argtype(restype_node, restype)
                for argtype in argtypes:
                    self._check_valid_argtype(restype_node, argtype)
                restype = restype(*argtypes)
            else:
                self._check_valid_argtype(restype_node, restype)
        else:
            raise error.NumbaError(restype_node, "Return type expected")

        return restype

    def _handle_jit_decorator(self, func_def, decorator):
        from numba import ast_type_inference

        jit_args = ast_type_inference._parse_args(
                decorator, ['restype', 'argtypes', 'backend',
                            'target', 'nopython'])

        if decorator.args or decorator.keywords:
            restype = self._parse_restype(decorator, jit_args)
            if restype is not None and restype.is_function:
                signature = restype
            else:
                argtypes = self._parse_argtypes(decorator, func_def, jit_args)
                signature = minitypes.FunctionType(restype, argtypes,
                                                   name=func_def.name)
        else: #elif func_def.args:
            raise error.NumbaError(decorator,
                                   "The argument types and return type "
                                   "need to be specified")
        #else:
        #    signature = minitypes.FunctionType(None, [])

        # TODO: Analyse closure at call or outer function return time to
        # TODO:     infer return type
        # TODO: parse out nopython argument
        del func_def.decorator_list[:]
        return signature

    def _process_decorators(self, node):
        if not node.decorator_list:
            if hasattr(node, 'func_signature'):
                return node.func_signature

            raise error.NumbaError(
                node, "Closure must be decorated with 'jit' or 'autojit'")

        if len(node.decorator_list) > 1:
            raise error.NumbaError(
                        node, "Only one decorator may be specified for "
                              "closure (@jit/@autojit)")

        decorator, = node.decorator_list

        if isinstance(decorator, ast.Name):
            decorator_name = decorator.id
        elif (not isinstance(decorator, ast.Call) or not
                  isinstance(decorator.func, ast.Name)):
            self._err_decorator(decorator)
        else:
            decorator_name = decorator.func.id

        if decorator_name not in ('jit', 'autojit'):
            self._err_decorator(decorator)

        if decorator_name == 'autojit':
            raise error.NumbaError(
                decorator, "Dynamic closures not yet supported, use @jit")

        signature = self._handle_jit_decorator(node, decorator)
        return signature

    def visit_FunctionDef(self, node):
        if self.function_level == 0:
            return self._visit_func_children(node)

        signature = self._process_decorators(node)
        type = numba_types.ClosureType(signature)
        self.symtab[node.name] = Variable(type, is_local=True)

        closure = nodes.ClosureNode(node, type, self.func)
        type.closure = closure
        self.ast.closures.append(closure)

        return closure

def mangle(name, scope):
    return name

    if not scope.is_closure_type or name in scope.unmangled_symtab:
        return '__numba_scope%s_var_%s' % (scope.scope_prefix, name)
    else:
        scope_attr_name, scope_type = scope.attribute_struct.fields[0]
        return mangle(name, scope_type)

def outer_scope_field(scope_type):
    return scope_type.attribute_struct.fields[0]

def lookup_scope_attribute(cur_scope, var_name, ctx=None):
    """
    Look up a variable in the closure scope
    """
    ctx = ctx or ast.Load()
    scope_type = cur_scope.type
    outer_scope_name, outer_scope_type = outer_scope_field(scope_type)

    if var_name in scope_type.unmangled_symtab:
        return nodes.ExtTypeAttribute(value=cur_scope,
                                      attr=mangle(var_name, scope_type),
                                      ctx=ctx, ext_type=scope_type)
    elif outer_scope_type.is_closure_scope:
        scope = nodes.ExtTypeAttribute(value=cur_scope, attr=outer_scope_name,
                                       ctx=ctx, ext_type=scope_type)
        return lookup_scope_attribute(scope, var_name, ctx)
    else:
        # This indicates a bug
        raise error.InternalError(
                cur_scope, "Unable to look up attribute %s" % var_name)

CLOSURE_SCOPE_ARG_NAME = '__numba_closure_scope'

class ClosureBaseVisitor(object):

    @property
    def outer_scope(self):
        outer_scope = None
        if CLOSURE_SCOPE_ARG_NAME in self.symtab:
            outer_scope = ast.Name(id=CLOSURE_SCOPE_ARG_NAME, ctx=ast.Load())
            outer_scope.variable = self.symtab[CLOSURE_SCOPE_ARG_NAME]
            outer_scope.type = outer_scope.variable.type

        return outer_scope

class ClosureTypeInferer(ClosureBaseVisitor, visitors.NumbaTransformer):
    """
    Runs just after type inference after the outer variables types are
    resolved.

    1) run type inferencer on inner functions
    2) build scope extension types pre-order
    3) generate nodes to instantiate scope extension type at call time
    """

    func_level = 0

    def _visit_children(self, node):
        self.func_level += 1
        self.generic_visit(node)
        self.func_level -= 1
        return node

    def visit_FunctionDef(self, node):
        if node.closure_scope is None:
            # Process inner functions and determine cellvars and freevars
            # codes = [c for c in self.constants
            #                if isinstance(c, types.CodeType)]
            process_closures(self.context, node, self.symtab)

        # cellvars are the variables we own
        cellvars = dict((name, var) for name, var in self.symtab.iteritems()
                                        if var.is_cellvar)
        node.cellvars = cellvars

        logger.debug("Cellvars in function %s: %s", node.name, cellvars)

        outer_scope = self.outer_scope
        if outer_scope:
            outer_scope_type = outer_scope.type
        else:
            outer_scope_type = None

        if not cellvars:
            # No cellvars, so use parent closure scope if this is a closure
            if outer_scope:
                self.update_closures(node, outer_scope_type, None)

            return self._visit_children(node)

        # Create closure scope extension type
        cellvar_fields = [(name, var.type)
                              for name, var in cellvars.iteritems()]
        fields = numba.struct(cellvar_fields).fields

        if outer_scope:
            fields.insert(0, ('__numba_base_scope', outer_scope_type))

        class py_class(object):
            pass

        func_name = self.func.__name__
        py_class.__name__ = '%s_scope' % func_name
        scope_type = numba_types.ClosureScopeType(py_class, outer_scope_type)
        scope_type.unmangled_symtab = dict(fields)

        mangled_fields = [(mangle(name, scope_type), type)
                              for name, type in fields]
        scope_type.set_attributes(mangled_fields)

        ext_type = extension_types.create_new_extension_type(
                            func_name , (object,), {}, scope_type,
                            vtab=None, vtab_type=numba.struct(),
                            llvm_methods=[], method_pointers=[])

        # Instantiate closure scope
        logger.debug("Generate closure %s %s %s", node.name, scope_type,
                     outer_scope)

        cellvar_scope = nodes.InstantiateClosureScope(
                    node, ext_type, scope_type, outer_scope)
        node.body.insert(0, cellvar_scope)

        self.update_closures(node, scope_type, ext_type)
        return self._visit_children(node)

    def update_closures(self, func_def, scope_type, ext_type):
        # Patch closures to get the closure scope as the first argument
        for closure in func_def.closures:
            # closure.scope_type = scope_type
            closure.func_def.scope_type = scope_type
            closure.ext_type = ext_type

            # patch function parameters
            param = ast.Name(id=CLOSURE_SCOPE_ARG_NAME, ctx=ast.Param())
            param.variable = Variable(scope_type, is_local=True)
            param.type = param.variable.type

            closure.symtab[CLOSURE_SCOPE_ARG_NAME] = param.variable
            closure.func_def.args.args.insert(0, param)
            closure.need_closure_scope = True

            # patch closure signature
            closure.type.signature.args = (scope_type,) + closure.type.signature.args

#    def visit_ClosureNode(self, node):
#        node.func_def = self.visit(node.func_def)
#        return node


def get_locals(symtab):
    return dict((name, var) for name, var in symtab.iteritems()
                    if var.is_local)

def process_closures(context, outer_func_def, outer_symtab):
    """
    Process closures recursively and for each variable in each function
    determine whether it is a freevar, a cellvar, a local or otherwise.
    """
    import numba.pipeline

    outer_symtab = get_locals(outer_symtab)

    # closure_scope is set on the FunctionDef by TypeInferer
    if outer_func_def.closure_scope is not None:
        closure_scope = dict(outer_func_def.closure_scope, **outer_symtab)
    else:
        closure_scope = outer_symtab

    for closure in outer_func_def.closures:
        logger.debug("process closures: %s %s", outer_func_def.name,
                     closure.func_def.name)
        closure.make_pyfunc()
        symtab = {}
        p, result = numba.pipeline.infer_types_from_ast_and_sig(
                    context, closure.py_func, closure.func_def,
                    closure.type.signature,
                    closure_scope=closure_scope,
                    symtab=symtab)

        _, _, ast = result
        closure.symtab = symtab
        closure.type_inferred_ast = ast

        process_closures(context, closure.func_def, symtab)


class ClosureCompilingMixin(ClosureBaseVisitor):
    """
    Runs during late specialization.

        - Instantiates the closure scope and makes the necessary assignments
        - Rewrites local variable accesses to accesses on the instantiated scope
        - Instantiate function with closure scope
    """

    def __init__(self, *args, **kwargs):
        super(ClosureCompilingMixin, self).__init__(*args, **kwargs)
        if not self.ast.cellvars:
            self.ast.cur_scope = self.outer_scope

    def _load_name(self, var_name, is_cellvar=False):
        src = ast.Name(var_name, ast.Load())
        src.variable = Variable.from_variable(self.symtab[var_name])
        src.variable.is_cellvar = is_cellvar
        src.type = src.variable.type
        return src

    def visit_InstantiateClosureScope(self, node):
        """
        Instantiate a closure scope.

        After instantiation, assign the parent scope and all function
        arguments that belong in the scope to the scope.
        """
        ctor = nodes.objconst(node.ext_type.__new__)
        ext_type_arg = nodes.objconst(node.ext_type)
        create_scope = nodes.ObjectCallNode(
                    signature=node.scope_type(object_), func=ctor,
                    args=[ext_type_arg])

        create_scope = create_scope.cloneable
        scope = create_scope.clone
        stats = [create_scope]

        # Chain outer scope - if present - to current scope
        outer_scope = self.outer_scope
        if outer_scope:
            outer_scope_name, outer_scope_type = outer_scope_field(scope.type)
            dst = lookup_scope_attribute(scope, outer_scope_name,
                                         ctx=ast.Store())
            assmt = ast.Assign(targets=[dst], value=outer_scope)
            stats.append(assmt)

        # Assign function arguments that are cellvars
        for arg in self.ast.args.args:
            assert isinstance(arg, ast.Name)
            if arg.id in node.scope_type.unmangled_symtab:
                dst = lookup_scope_attribute(scope, arg.id, ast.Store())
                src = self._load_name(arg.id)
                assmt = ast.Assign(targets=[dst], value=src)
                stats.append(assmt)

        logger.debug("instantiating %s", scope.type)
        self.ast.cur_scope = scope
        return self.visit(nodes.ExpressionNode(stmts=stats, expr=scope))

    def visit_ClosureNode(self, node):
        """
        Compile the inner function.
        """
        closure_scope = self.ast.cur_scope

        if closure_scope is None:
            closure_scope = nodes.NULL
            scope_type = void.pointer()
        else:
            assert node.func_def.args.args[0].variable.type
            scope_type = closure_scope.type

        # Compile inner function, skip type inference
        order = numba.pipeline.Pipeline.order
        order = order[order.index('type_infer') + 1:]
        p, result = numba.pipeline.run_pipeline(
                    self.context, node.py_func, node.type_inferred_ast,
                    node.type.signature, symtab=node.symtab,
                    order=order, # skip type inference
                    )

        node.lfunc = p.translator.lfunc
        node.lfunc_pointer = p.translator.lfunc_pointer
        node.wrapper_func, node.wrapper_lfunc, methoddef = (
                    p.translator.build_wrapper_function(get_lfunc=True))

        # Keep methoddef alive
        assert methoddef in node.py_func.live_objects
        modname = self.func.__module__
        node.py_func.live_objects.append(modname)

        # Create function with closure scope at runtime
        create_numbafunc_signature = node.type(
            void.pointer(),     # PyMethodDef *ml
            object_,            # PyObject *module
            void.pointer(),     # PyObject *code
            scope_type,         # PyObject *closure
            void.pointer(),     # void *native_func
            object_,            # PyObject *native_signature
            object_,            # PyObject *keep_alive
        )
        create_numbafunc = nodes.ptrfromint(
                        extension_types.NumbaFunction_NewEx_pointer,
                        create_numbafunc_signature.pointer())

        methoddef_p = ctypes.cast(ctypes.byref(methoddef),
                                  ctypes.c_void_p).value

        args = [
            nodes.const(methoddef_p, void.pointer()),
            nodes.const(modname, object_),
            nodes.NULL,
            closure_scope,
            nodes.const(node.lfunc_pointer, void.pointer()),
            nodes.const(node.type.signature, object_),
            nodes.const(node.py_func, object_),
        ]

        func_call = nodes.NativeFunctionCallNode(
                            signature=create_numbafunc_signature,
                            function_node=create_numbafunc,
                            args=args)

        # Assign closure to its name
        func_name = node.func_def.name
        dst = self._load_name(func_name, self.symtab[func_name].is_cellvar)
        dst.ctx = ast.Store()
        result = ast.Assign(targets=[dst], value=func_call)

        #stats = [nodes.inject_print(nodes.const("calling...", c_string_type)),
        #         result]
        #result = ast.Suite(body=stats)
        result = self.visit(result)
        return result

    def visit_Name(self, node):
        "Resolve cellvars and freevars"
        if node.variable.is_cellvar or node.variable.is_freevar:
            logger.debug("Function %s, lookup %s in scope %s: %s",
                         self.ast.name, node.id, self.ast.cur_scope.type,
                         self.ast.cur_scope.type.attribute_struct)
            attr = lookup_scope_attribute(self.ast.cur_scope,
                                          var_name=node.id, ctx=node.ctx)
            return self.visit(attr)
        else:
            return node

    def visit_ClosureCallNode(self, node):
        if node.closure_type.closure.need_closure_scope:
            assert self.ast.cur_scope is not None
            node.args.insert(0, self.ast.cur_scope)

        self.generic_visit(node)
        return node

    def visit_ClosureScopeLoadNode(self, node):
        return self.ast.cur_scope or nodes.NULL_obj