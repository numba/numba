import ast
import ctypes

import numba.decorators
from numba import *
from numba import error, visitors, nodes
from numba.minivect import  minitypes
from numba import  _numba_types as numba_types
from numba.symtab import Variable

import logging
logger = logging.getLogger(__name__)


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
    return '__numba_scope%s_%s' % (scope.scope_prefix, name)

def lookup_scope_attribute(cur_scope, var_name, ctx=None):
    """
    Look up a variable in the closure scope
    """
    ctx = ctx or ast.Load()
    scope_type = cur_scope.type
    field_name, field_type = scope_type.attribute_struct.fields[0]

    var_name = mangle(var_name, scope_type)
    if var_name in scope_type.symtab:
        return nodes.ExtTypeAttribute(value=cur_scope, attr=var_name,
                                      ctx=ctx, ext_type=scope_type)
    elif field_type.is_closure_scope:
        return nodes.ExtTypeAttribute(value=cur_scope, attr=field_name,
                                      ctx=ctx, ext_type=scope_type)
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
    2) build scope extension type
    3) generate nodes to instantiate scope extension type at call time
    """

    def visit_func_children(self, node):
        for closure in node.closures:
            # Create fake python function that backs up the AST of inner
            # functions
            closure.make_pyfunc()

        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        # Process inner functions and determine cellvars and freevars
        process_closures(self.context, node, self.symtab)

        # cellvars are the variables we own
        cellvars = dict((name, var) for name, var in self.symtab.iteritems()
                                        if var.is_cellvar)
        node.cellvars = cellvars

        if not cellvars:
            return self.visit_func_children(node)

        # Create closure scope extension type
        cellvar_fields = [(name, var.type) for name, var in cellvars.iteritems()]
        fields = numba.struct(cellvar_fields).fields
        if self.outer_scope:
            fields.insert(0, ('__numba_base_scope',
                              self.parent_scope.__numba_ext_type))

        class py_class(object):
            pass

        func_name = self.func.__name__
        py_class.__name__ = '%s_scope' % func_name
        scope_type = numba_types.ClosureScopeType(py_class, self.outer_scope)
        scope_type.unmangled_symtab = dict(fields)

        mangled_fields = [(mangle(name, scope_type), type)
                              for name, type in fields]
        scope_type.set_attributes(mangled_fields)

        ext_type = extension_types.create_new_extension_type(
                            func_name , (object,), {}, scope_type,
                            vtab=None, vtab_type=numba.struct(),
                            llvm_methods=[], method_pointers=[])

        # Instantiate closure scope
        if self.outer_scope:
            outer_scope = self.outer_scope
        else:
            outer_scope = None

        cellvar_scope = nodes.InstantiateClosureScope(
                    node, ext_type, scope_type, outer_scope)
        node.body.insert(0, cellvar_scope)

        # Path closures to get the closure scope as the first argument
        for closure in node.closures:
            closure.scope_type = scope_type
            closure.ext_type = ext_type

            # patch symtab and function parameters
            var = Variable(scope_type, is_local=True)
            closure.symtab[CLOSURE_SCOPE_ARG_NAME] = var

            param = ast.Name(id=CLOSURE_SCOPE_ARG_NAME, ctx=ast.Param())
            param.variable = var
            param.type = var.type

            closure.func_def.args.args.insert(0, param)
            closure.need_closure_scope = True

            # patch closure signature
            closure.type.signature.args = (scope_type,) + closure.type.signature.args

        # Update closures python functions and visit children
        return self.visit_func_children(node)

    # process_closures performs the recursion
    #def visit_ClosureNode(self, node):
    #    node.func_def = self.visit(node.func_def)
    #    return node


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
        p, result = numba.pipeline.infer_types_from_ast_and_sig(
                    context, closure.py_func, closure.func_def,
                    closure.type.signature,
                    closure_scope=closure_scope)

        _, symtab, ast = result
        closure.symtab = symtab
        closure.type_inferred_ast = ast


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
        ctor = nodes.objconst(object.__new__)
        ext_type_arg = nodes.objconst(node.ext_type)
        create_scope = nodes.ObjectCallNode(
                    signature=node.scope_type(object_), func=ctor,
                    args=[ext_type_arg])

        create_scope = create_scope.cloneable
        scope = create_scope.clone
        stats = [create_scope]

        for arg in self.ast.args.args:
            assert isintance(arg, ast.Name)
            if arg.id in node.scope_type.unmangled_symtab:
                var_name = mangle(arg.id, node.scope_type)
                dst = lookup_scope_attribute(scope, node.scope_type,
                                             var_name, ast.Store())
                src = self._load_name(arg.id)
                assmt = ast.Assign(targets=[dst], value=src)
                stats.append(assmt)

        self.ast.cur_scope = scope
        return self.visit(nodes.ExpressionNode(stmts=stats, expr=scope))

    def visit_ClosureNode(self, node):
        """
        Compile the inner function.
        """
        closure_scope = self.ast.cur_scope

        if closure_scope is None:
            closure_scope = nodes.NULL
        else:
            assert node.func_def.args.args[0].type

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
        scope_type = node.scope_type or void.pointer()
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