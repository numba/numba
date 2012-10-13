"""
This module provides the AST. Subclass :py:class:`Context` and override the
various methods to allow minivect visitors over the AST, to promote and map types,
etc. Subclass and override :py:class:`ASTBuilder`'s methods to provide alternative
AST nodes or different implementations.
"""

import copy
import string
import types

import minitypes
import miniutils
import minivisitor
import specializers
import type_promoter
import minicode
import codegen
import llvm_codegen
import graphviz

try:
    import llvm.core
    import llvm.ee
    import llvm.passes
except ImportError:
    llvm = None

class UndocClassAttribute(object):
    "Use this to document class attributes for Sphinx"
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

def make_cls(cls1, cls2):
    "Fuse two classes together."
    name = "%s_%s" % (cls1.__name__, cls2.__name__)
    return type(name, (cls1, cls2), {})

class Context(object):
    """
    A context that knows how to map ASTs back and forth, how to wrap nodes
    and types, and how to instantiate a code generator for specialization.

    An opaque_node or foreign node is a node that is not from our AST,
    and a normal node is one that has a interface compatible with ours.

    To provide custom functionality, set the following attributes, or
    subclass this class.

    :param astbuilder: the :py:class:`ASTBuilder` or ``None``
    :param typemapper: the :py:class:`minivect.minitypes.Typemapper` or
                       ``None`` for the default.

    .. attribute:: codegen_cls

        The code generator class that is used to generate code.
        The default is :py:class:`minivect.codegen.CodeGen`

    .. attribute:: cleanup_codegen_cls

        The code generator that generates code to dispose of any
        garbage (e.g. intermediate object temporaries).
        The default is :py:class:`minivect.codegen.CodeGenCleanup`

    .. attribute:: codewriter_cls

        The code writer that the code generator writes its generated code
        to. This may be strings or arbitrary objects.
        The default is :py:class:`minivect.minicode.CodeWriter`, which accepts
        arbitrary objects.

    .. attribute:: codeformatter_cls

        A formatter to format the generated code.

        The default is :py:class:`minivect.minicode.CodeFormatter`,
        which returns a list of objects written. Set this to
        :py:class:`minivect.minicode.CodeStringFormatter`
        to have the strings joined together.

    .. attribute:: specializer_mixin_cls

        A specializer mixin class that can override or intercept
        functionality. This class should likely participate
        cooperatively in MI.

    .. attribute:: variable_resolving_mixin_cls

        A specializer mixin class that resolves wrapped miniasts in a foreign
        AST. This is only needed if you are using :py:class:`NodeWrapper`,
        which wraps a miniast somewhere at the leaves.

    .. attribute: graphviz_cls

        Visitor to generate a Graphviz graph. See the :py:module:`graphviz`
        module.

    .. attribute: minifunction

        The current minifunction that is being translated.

    Use subclass :py:class:`CContext` to get the defaults for C code generation.
    """

    debug = False
    debug_elements = False

    use_llvm = False
    optimize_broadcasting = True

    shape_type = minitypes.Py_ssize_t.pointer()
    strides_type = shape_type

    astbuilder_cls = None
    codegen_cls = UndocClassAttribute(codegen.VectorCodegen)
    cleanup_codegen_cls = UndocClassAttribute(codegen.CodeGenCleanup)
    codewriter_cls = UndocClassAttribute(minicode.CodeWriter)
    codeformatter_cls = UndocClassAttribute(minicode.CodeFormatter)
    graphviz_cls = UndocClassAttribute(graphviz.GraphvizGenerator)

    specializer_mixin_cls = None
    variable_resolving_mixin_cls = None

    func_counter = 0

    final_specializer = specializers.FinalSpecializer

    def __init__(self):
        self.init()
        if self.use_llvm:
            if llvm is None:
                import llvm.core as llvm_py_not_available # llvm-py not available

            self.llvm_module = llvm.core.Module.new('default_module')
            # self.llvm_ee = llvm.ee.ExecutionEngine.new(self.llvm_module)
            self.llvm_ee = llvm.ee.EngineBuilder.new(self.llvm_module).force_jit().opt(3).create()
            self.llvm_fpm = llvm.passes.FunctionPassManager.new(self.llvm_module)
            self.llvm_fpm.initialize()
            if not self.debug:
                for llvm_pass in self.llvm_passes():
                    self.llvm_fpm.add(llvm_pass)
        else:
            self.llvm_ee = None
            self.llvm_module = None

    def init(self):
        self.astbuilder = self.astbuilder_cls(self)
        self.typemapper = minitypes.TypeMapper(self)

    def run_opaque(self, astmapper, opaque_ast, specializers):
        return self.run(astmapper.visit(opaque_ast), specializers)

    def run(self, ast, specializer_classes, graphviz_outfile=None,
            print_tree=False):
        """
        Specialize the given AST with all given specializers and return
        an iterable of generated code in the form of
        ``(specializer, new_ast, codewriter, code_obj)``

        The code_obj is the generated code (e.g. a string of C code),
        depending on the code formatter used.
        """
        for specializer_class in specializer_classes:
            self.init()
            pipeline = self.pipeline(specializer_class)

            specialized_ast = specializers.specialize_ast(ast)
            self.astbuilder.minifunction = specialized_ast
            for transform in pipeline:
                specialized_ast = transform.visit(specialized_ast)

            if print_tree:
                specialized_ast.print_tree(self)

            if graphviz_outfile is not None:
                data = self.graphviz(specialized_ast)
                graphviz_outfile.write(data)

            codewriter = self.codewriter_cls(self)
            codegen = self.codegen_cls(self, codewriter)
            codegen.visit(specialized_ast)

            yield (pipeline[0], specialized_ast, codewriter,
                   self.codeformatter_cls().format(codewriter))

    def debug_c(self, ast, specializer, astbuilder_cls=None):
        "Generate C code (for debugging)"
        context = CContext()

        if astbuilder_cls:
            context.astbuilder_cls = astbuilder_cls
        else:
            context.astbuilder_cls = self.astbuilder_cls

        context.shape_type = self.shape_type
        context.strides_type = self.strides_type

        context.debug = self.debug
        result = context.run(ast, [specializer]).next()
        _, specialized_ast, _, (proto, impl) = result
        return impl

    def pipeline(self, specializer_class):
        # add specializer mixin and run specializer
        if self.specializer_mixin_cls:
            specializer_class = make_cls(self.specializer_mixin_cls,
                                         specializer_class)

        specializer = specializer_class(self)
        pipeline = [specializer]

        # Add variable resolving mixin to the final specializer and run
        # transform
        final_specializer_cls = self.final_specializer
        if final_specializer_cls:
            if self.variable_resolving_mixin_cls:
                final_specializer_cls = make_cls(
                    self.variable_resolving_mixin_cls,
                    final_specializer_cls)

            pipeline.append(final_specializer_cls(self, specializer))

        pipeline.append(type_promoter.TypePromoter(self))
        return pipeline

    def generate_disposal_code(self, code, node):
        "Run the disposal code generator on an (sub)AST"
        transform = self.cleanup_codegen_cls(self, code)
        transform.visit(node)

    #
    ### Override in subclasses where needed
    #

    def llvm_passes(self):
        "Returns a list of LLVM optimization passes"
        return []
        return [
            # llvm.passes.PASS_CFG_SIMPLIFICATION
            llvm.passes.PASS_BLOCK_PLACEMENT,
            llvm.passes.PASS_BASIC_ALIAS_ANALYSIS,
            llvm.passes.PASS_NO_AA,
            llvm.passes.PASS_SCALAR_EVOLUTION_ALIAS_ANALYSIS,
#            llvm.passes.PASS_ALIAS_ANALYSIS_COUNTER,
            llvm.passes.PASS_AAEVAL,
            llvm.passes.PASS_LOOP_DEPENDENCE_ANALYSIS,
            llvm.passes.PASS_BREAK_CRITICAL_EDGES,
            llvm.passes.PASS_LOOP_SIMPLIFY,
            llvm.passes.PASS_PROMOTE_MEMORY_TO_REGISTER,
            llvm.passes.PASS_CONSTANT_PROPAGATION,
            llvm.passes.PASS_LICM,
            # llvm.passes.PASS_CONSTANT_MERGE,
            llvm.passes.PASS_LOOP_STRENGTH_REDUCE,
            llvm.passes.PASS_LOOP_UNROLL,
            # llvm.passes.PASS_FUNCTION_ATTRS,
            # llvm.passes.PASS_GLOBAL_OPTIMIZER,
            # llvm.passes.PASS_GLOBAL_DCE,
            llvm.passes.PASS_DEAD_CODE_ELIMINATION,
            llvm.passes.PASS_INSTRUCTION_COMBINING,
            llvm.passes.PASS_CODE_GEN_PREPARE,
        ]

    def mangle_function_name(self, name):
        name = "%s_%d" % (name, self.func_counter)
        self.func_counter += 1
        return name

    def promote_types(self, type1, type2):
        "Promote types in an arithmetic operation"
        if type1 == type2:
            return type1
        return self.typemapper.promote_types(type1, type2)

    def getchildren(self, node):
        "Implement to allow a minivisitor.Visitor over a foreign AST."
        return node.child_attrs

    def getpos(self, opaque_node):
        "Get the position of a foreign node"
        filename, line, col = opaque_node.pos
        return Position(filename, line, col)

    def gettype(self, opaque_node):
        "Get a type of a foreign node"
        return opaque_node.type

    def may_error(self, opaque_node):
        "Return whether this node may result in an exception."
        raise NotImplementedError

    def declare_type(self, type):
        "Return a declaration for a type"
        raise NotImplementedError

    def to_llvm(self, type):
        "Return an LLVM type for the given minitype"
        return self.typemapper.to_llvm(type)

    def graphviz(self, node, graphviz_name="AST"):
        visitor = self.graphviz_cls(self, graphviz_name)
        graphviz_graph = visitor.visit(node)
        return graphviz_graph.to_string()

class CContext(Context):
    "Set defaults for C code generation."

    codegen_cls = codegen.VectorCodegen
    codewriter_cls = minicode.CCodeWriter
    codeformatter_cls = minicode.CCodeStringFormatter

class LLVMContext(Context):
    "Context with default for LLVM code generation"

    use_llvm = True
    codegen_cls = llvm_codegen.LLVMCodeGen

class ASTBuilder(object):
    """
    This class is used to build up a minivect AST. It can be used by a user
    from a transform or otherwise, but the important bit is that we use it
    in our code to build up an AST that can be overridden by the user,
    and which makes it convenient to build up complex ASTs concisely.
    """

    # the 'pos' attribute is set for each visit to each node by
    # the ASTMapper
    pos = None
    temp_reprname_counter = 0

    def __init__(self, context):
        """
        :param context: the :py:class:`Context`
        """
        self.context = context

    def _infer_type(self, value):
        "Used to infer types for self.constant()"
        if isinstance(value, (int, long)):
            return minitypes.IntType()
        elif isinstance(value, float):
            return minitypes.FloatType()
        elif isinstance(value, str):
            return minitypes.CStringType()
        else:
            raise minierror.InferTypeError()

    def create_function_type(self, function, strides_args=True):
        arg_types = []
        for arg in function.arguments + function.scalar_arguments:
            if arg.used:
                if arg.type and arg.type.is_array and not strides_args:
                    arg_types.append(arg.data_pointer.type)
                    arg.variables = [arg.data_pointer]
                else:
                    for variable in arg.variables:
                        arg_types.append(variable.type)

        function.type = minitypes.FunctionType(
                    return_type=function.success_value.type, args=arg_types)

    def function(self, name, body, args, shapevar=None, posinfo=None,
                 omp_size=None):
        """
        Create a new function.

        :type name: str
        :param name: name of the function

        :type args: [:py:class:`FunctionArgument`]
        :param args: all array and scalar arguments to the function, excluding
                     shape or position information.

        :param shapevar: the :py:class:`Variable` for the total broadcast shape
                         If ``None``, a default of ``Py_ssize_t *`` is assumed.

        :type posinfo: :py:class:`FunctionArgument`
        :param posinfo: if given, this will be the second, third and fourth
                        arguments to the function ``(filename, lineno, column)``.
        """
        if shapevar is None:
            shapevar = self.variable(self.context.shape_type, 'shape')

        arguments, scalar_arguments = [], []
        for arg in args:
            if arg.type.is_array:
                arguments.append(arg)
            else:
                scalar_arguments.append(arg)

        arguments.insert(0, self.funcarg(shapevar))
        if posinfo:
            arguments.insert(1, posinfo)

        body = self.stats(self.nditerate(body))

        error_value = self.constant(-1)
        success_value = self.constant(0)

        function = FunctionNode(self.pos, name, body,
                            arguments, scalar_arguments,
                            shapevar, posinfo,
                            error_value=error_value,
                            success_value=success_value,
                            omp_size=omp_size or self.constant(1024))

        # prepending statements, used during specialization
        function.prepending = self.stats()
        function.body = self.stats(function.prepending, function.body)

        self.create_function_type(function)
        return function

    def build_function(self, variables, body, name=None, shapevar=None):
        "Convenience method for building a minivect function"
        args = []
        for var in variables:
            if var.type.is_array:
                args.append(self.array_funcarg(var))
            else:
                args.append(self.funcarg(var))

        name = name or 'function'
        return self.function(name, body, args, shapevar=shapevar)

    def funcarg(self, variable, *variables):
        """
        Create a (compound) function argument consisting of one or multiple
        argument Variables.
        """
        if variable.type is not None and variable.type.is_array:
            assert not variables
            return self.array_funcarg(variable)

        if not variables:
            variables = [variable]
        return FunctionArgument(self.pos, variable, list(variables))

    def array_funcarg(self, variable):
        "Create an array function argument"
        return ArrayFunctionArgument(
                self.pos, variable.type, name=variable.name,
                variable=variable,
                data_pointer=self.data_pointer(variable),
                #shape_pointer=self.shapevar(variable),
                strides_pointer=self.stridesvar(variable))

    def incref(self, var, funcname='Py_INCREF'):
        "Generate a Py_INCREF() statement"
        functype = minitypes.FunctionType(return_type=minitypes.void,
                                          args=[minitypes.object_])
        py_incref = self.funcname(functype, funcname)
        return self.expr_stat(self.funccall(py_incref, [var]))

    def decref(self, var):
        "Generate a Py_DECCREF() statement"
        return self.incref(var, funcname='Py_DECREF')

    def print_(self, *args):
        "Print out all arguments to stdout"
        return PrintNode(self.pos, args=list(args))

    def funccall(self, func_or_pointer, args, inline=False):
        """
        Generate a call to the given function (a :py:class:`FuncNameNode`) of
        :py:class:`minivect.minitypes.FunctionType` or a
        pointer to a function type and the given arguments.
        """
        type = func_or_pointer.type
        if type.is_pointer:
            type = func_or_pointer.type.base_type
        return FuncCallNode(self.pos, type.return_type,
                            func_or_pointer=func_or_pointer, args=args,
                            inline=inline)

    def funcname(self, type, name, is_external=True):
        assert type.is_function
        return FuncNameNode(self.pos, type, name=name, is_external=is_external)

    def nditerate(self, body):
        """
        This node wraps the given AST expression in an :py:class:`NDIterate`
        node, which will be expanded by the specializers to one or several
        loops.
        """
        return NDIterate(self.pos, body)

    def for_(self, body, init, condition, step, index=None):
        """
        Create a for loop node.

        :param body: loop body
        :param init: assignment expression
        :param condition: boolean loop condition
        :param step: step clause (assignment expression)
        """
        return ForNode(self.pos, init, condition, step, body, index=index)

    def for_range_upwards(self, body, upper, lower=None, step=None):
        """
        Create a single upwards for loop, typically used from a specializer to
        replace an :py:class:`NDIterate` node.

        :param body: the loop body
        :param upper: expression specifying an upper bound
        """
        index_type = upper.type.unqualify("const")

        if lower is None:
            lower = self.constant(0, index_type)
        if step is None:
            step = self.constant(1, index_type)

        temp = self.temp(index_type)
        init = self.assign_expr(temp, lower)
        condition = self.binop(minitypes.bool_, '<', temp, upper)
        step = self.assign_expr(temp, self.add(temp, step))

        result = self.for_(body, init, condition, step)
        result.target = temp
        return result

    def omp_for(self, for_node, if_clause):
        """
        Annotate the for loop with an OpenMP parallel for clause.

        :param if_clause: the expression node that determines whether the
                          parallel section is executed or whether it is
                          executed sequentially (to avoid synchronization
                          overhead)
        """
        if isinstance(for_node, PragmaForLoopNode):
            for_node = for_node.for_node
        return OpenMPLoopNode(self.pos, for_node=for_node,
                              if_clause=if_clause,
                              lastprivates=[for_node.init.lhs],
                              privates=[])

    def omp_if(self, if_body, else_body=None):
        return OpenMPConditionalNode(self.pos, if_body=if_body,
                                     else_body=else_body)

    def pragma_for(self, for_node):
        """
        Annotate the for loop with pragmas.
        """
        return PragmaForLoopNode(self.pos, for_node=for_node)

    def stats(self, *statements):
        """
        Wrap a bunch of statements in an AST node.
        """
        return StatListNode(self.pos, list(statements))

    def expr_stat(self, expr):
        "Turn an expression into a statement"
        return ExprStatNode(expr.pos, type=expr.type, expr=expr)

    def expr(self, stats=(), expr=None):
        "Evaluate a bunch of statements before evaluating an expression."
        return ExprNodeWithStatement(self.pos, type=expr.type,
                                     stat=self.stats(*stats), expr=expr)

    def if_(self, cond, body):
        "If statement"
        return self.if_else(cond, body, None)

    def if_else_expr(self, cond, lhs, rhs):
        "If/else expression, resulting in lhs if cond else rhs"
        type = self.context.promote_types(lhs.type, rhs.type)
        return IfElseExprNode(self.pos, type=type, cond=cond, lhs=lhs, rhs=rhs)

    def if_else(self, cond, if_body, else_body):
        return IfNode(self.pos, cond=cond, body=if_body, else_body=else_body)

    def promote(self, dst_type, node):
        "Promote or demote the node to the given dst_type"
        if node.type != dst_type:
            if node.is_constant and node.type.kind == dst_type.kind:
                node.type = dst_type
                return node
            return PromotionNode(self.pos, dst_type, node)
        return node

    def binop(self, type, op, lhs, rhs):
        """
        Binary operation on two nodes.

        :param type: the result type of the expression
        :param op: binary operator
        :type op: str
        """
        return BinopNode(self.pos, type, op, lhs, rhs)

    def add(self, lhs, rhs, result_type=None, op='+'):
        """
        Shorthand for the + binop. Filters out adding 0 constants.
        """
        if lhs.is_constant and lhs.value == 0:
            return rhs
        elif rhs.is_constant and rhs.value == 0:
            return lhs

        if result_type is None:
            result_type = self.context.promote_types(lhs.type, rhs.type)
        return self.binop(result_type, op, lhs, rhs)

    def sub(self, lhs, rhs, result_type=None):
        return self.add(lhs, rhs, result_type, op='-')

    def mul(self, lhs, rhs, result_type=None, op='*'):
        """
        Shorthand for the * binop. Filters out multiplication with 1 constants.
        """
        if op == '*' and lhs.is_constant and lhs.value == 1:
            return rhs
        elif rhs.is_constant and rhs.value == 1:
            return lhs

        if result_type is None:
            result_type = self.context.promote_types(lhs.type, rhs.type)
        return self.binop(result_type, op, lhs, rhs)

    def div(self, lhs, rhs, result_type=None):
        return self.mul(lhs, rhs, result_type=result_type, op='/')

    def min(self, lhs, rhs):
        """
        Returns min(lhs, rhs) expression.

        .. NOTE:: Make lhs and rhs temporaries if they should only be
                  evaluated once.
        """
        type = self.context.promote_types(lhs.type, rhs.type)
        cmp_node = self.binop(type, '<', lhs, rhs)
        return self.if_else_expr(cmp_node, lhs, rhs)

    def index(self, pointer, index, dest_pointer_type=None):
        """
        Index a pointer with the given index node.

        :param dest_pointer_type: if given, cast the result (*after* adding
                                  the index) to the destination type and
                                  dereference.
        """
        if dest_pointer_type:
            return self.index_multiple(pointer, [index], dest_pointer_type)
        return SingleIndexNode(self.pos, pointer.type.base_type,
                               pointer, index)

    def index_multiple(self, pointer, indices, dest_pointer_type=None):
        """
        Same as :py:meth:`index`, but accepts multiple indices. This is
        useful e.g. after multiplication of the indices with the strides.
        """
        for index in indices:
            pointer = self.add(pointer, index)

        if dest_pointer_type is not None:
            pointer = self.cast(pointer, dest_pointer_type)

        return self.dereference(pointer)

    def assign_expr(self, node, value, may_reorder=False):
        "Create an assignment expression assigning ``value`` to ``node``"
        assert node is not None
        if not isinstance(value, Node):
            value = self.constant(value)
        return AssignmentExpr(self.pos, node.type, node, value,
                              may_reorder=may_reorder)

    def assign(self, node, value, may_reorder=False):
        "Assignment statement"
        expr = self.assign_expr(node, value, may_reorder=may_reorder)
        return self.expr_stat(expr)

    def dereference(self, pointer):
        "Dereference a pointer"
        return DereferenceNode(self.pos, pointer.type.base_type, pointer)

    def unop(self, type, operator, operand):
        "Unary operation. ``type`` indicates the result type of the expression."
        return UnopNode(self.pos, type, operator, operand)

    def coerce_to_temp(self, expr):
        "Coerce the given expression to a temporary"
        type = expr.type
        if type.is_array:
            type = type.dtype
        temp = self.temp(type)
        return self.expr(stats=[self.assign(temp, expr)], expr=temp)

    def temp(self, type, name=None):
        "Allocate a temporary of a given type"
        name = name or 'temp'
        repr_name = '%s%d' % (name.rstrip(string.digits),
                              self.temp_reprname_counter)
        self.temp_reprname_counter += 1
        return TempNode(self.pos, type, name=name, repr_name=repr_name)

    def constant(self, value, type=None):
        """
        Create a constant from a Python value. If type is not given, it is
        inferred (or it will raise a
        :py:class:`minivect.minierror.InferTypeError`).
        """
        if type is None:
            type = self._infer_type(value)

        return ConstantNode(self.pos, type, value)

    def variable(self, type, name):
        """
        Create a variable with a name and type. Variables
        may refer to function arguments, functions, etc.
        """
        return Variable(self.pos, type, name)

    def resolved_variable(self, array_type, name, element):
        """
        Creates a node that keeps the array operand information such as the
        original array type, but references an actual element in the array.

        :param type: original array type
        :param name: original array's name
        :param element: arbitrary expression that resolves some element in the
                        array
        """
        return ResolvedVariable(self.pos, element.type, name,
                                element=element, array_type=array_type)

    def cast(self, node, dest_type):
        "Cast node to the given destination type"
        return CastNode(self.pos, dest_type, node)

    def return_(self, result):
        "Return a result"
        return ReturnNode(self.pos, result)

    def data_pointer(self, variable):
        "Return the data pointer of an array variable"
        assert variable.type.is_array
        return DataPointer(self.pos, variable.type.dtype.pointer(),
                           variable)

    def shape_index(self, index, function):
        "Index the shape of the array operands with integer `index`"
        return self.index(function.shape, self.constant(index))

    def extent(self, variable, index, function):
        "Index the shape of a specific variable with integer `index`"
        assert variable.type.is_array
        offset = function.ndim - variable.type.ndim
        return self.index(function.shape, self.constant(index + offset))

    def stridesvar(self, variable):
        "Return the strides variable for the given array operand"
        return StridePointer(self.pos, self.context.strides_type, variable)

    def stride(self, variable, index):
        "Return the stride of array operand `variable` at integer `index`"
        return self.index(self.stridesvar(variable), self.constant(index))

    def sizeof(self, type):
        "Return the expression sizeof(type)"
        return SizeofNode(self.pos, minitypes.size_t, sizeof_type=type)

    def jump(self, label):
        "Jump to a label"
        return JumpNode(self.pos, label)

    def jump_target(self, label):
        """
        Return a target that can be jumped to given a label. The label is
        shared between the jumpers and the target.
        """
        return JumpTargetNode(self.pos, label)

    def label(self, name):
        "Return a label with a name"
        return LabelNode(self.pos, name)

    def raise_exc(self, posinfo, exc_var, msg_val, fmt_args):
        """
        Raise an exception given the positional information (see the `posinfo`
        method), the exception type (PyExc_*), a formatted message string and
        a list of values to be used for the format string.
        """
        return RaiseNode(self.pos, posinfo, exc_var, msg_val, fmt_args)

    def posinfo(self, posvars):
        """
        Return position information given a list of position variables
        (filename, lineno, column). This can be used for raising exceptions.
        """
        return PositionInfoNode(self.pos, posinfo=posvars)

    def error_handler(self, node):
        """
        Wrap the given node, which may raise exceptions, in an error handler.
        An error handler allows the code to clean up before propagating the
        error, and finally returning an error indicator from the function.
        """
        return ErrorHandler(self.pos, body=node,
                            error_label=self.label('error'),
                            cleanup_label=self.label('cleanup'))

    def wrap(self, opaque_node, specialize_node_callback, **kwds):
        """
        Wrap a node and type and return a NodeWrapper node. This node
        will have to be handled by the caller in a code generator. The
        specialize_node_callback is called when the NodeWrapper is
        specialized by a Specializer.
        """
        type = minitypes.TypeWrapper(self.context.gettype(opaque_node),
                                     self.context)
        return NodeWrapper(self.context.getpos(opaque_node), type,
                           opaque_node, specialize_node_callback, **kwds)

    #
    ### Vectorization Functionality
    #

    def _vector_type(self, base_type, size):
        return minitypes.VectorType(element_type=base_type, vector_size=size)

    def vector_variable(self, variable, size):
        "Return a vector variable for a data pointer variable"
        type = self._vector_type(variable.type.dtype, size)

        if size == 4:
            name = 'xmm_%s' % variable.name
        else:
            name = 'ymm_%s' % variable.name

        return VectorVariable(self.pos, type, name, variable=variable)

    def vector_load(self, data_pointer, size):
        "Load a SIMD vector of size `size` given an array operand variable"
        type = self._vector_type(data_pointer.type.base_type, size)
        return VectorLoadNode(self.pos, type, data_pointer, size=size)

    def vector_store(self, data_pointer, vector_expr):
        "Store a SIMD vector of size `size`"
        assert data_pointer.type.base_type == vector_expr.type.element_type
        return VectorStoreNode(self.pos, None, "=", data_pointer, vector_expr)

    def vector_binop(self, operator, lhs, rhs):
        "Perform a binary SIMD operation between two operands of the same type"
        assert lhs.type == rhs.type, (lhs.type, rhs.type)
        type = lhs.type
        return VectorBinopNode(self.pos, type, operator, lhs=lhs, rhs=rhs)

    def vector_unop(self, type, operator, operand):
        return VectorUnopNode(self.pos, type, operator, operand)

    def vector_const(self, type, constant):
        return ConstantVectorNode(self.pos, type, constant=constant)

    def noop_expr(self):
        return NoopExpr(self.pos, type=None)

class DynamicArgumentASTBuilder(ASTBuilder):
    """
    Create a function with a dynamic number of arguments. This means the
    signature looks like

        func(int *shape, float *data[n_ops], int *strides[n_ops])

    To create minivect kernels supporting this signature, set the
    astbuilder_cls attribute of Context to this class.
    """

    def data_pointer(self, variable):
        if not hasattr(variable, 'data_pointer'):
            temp =  self.temp(variable.type.dtype.pointer(),
                              variable.name + "_data_temp")
            variable.data_pointer = temp

        return variable.data_pointer

    def _create_data_pointer(self, function, argument, i):
        variable = argument.variable

        temp = self.data_pointer(variable)
        p = self.index(function.data_pointers, self.constant(i))
        p = self.cast(p, variable.type.dtype.pointer())
        assmt = self.assign(temp, p)

        function.body.stats.insert(0, assmt)
        return temp

    def stridesvar(self, variable):
        "Return the strides variable for the given array operand"
        if not hasattr(variable, 'strides_pointer'):
            temp = self.temp(self.context.strides_type,
                             variable.name + "_stride_temp")
            variable.strides_pointer = temp

        return variable.strides_pointer

    def _create_strides_pointer(self, function, argument, i):
        variable = argument.variable
        temp = self.stridesvar(variable)
        strides = self.index(function.strides_pointers, self.constant(i))
        function.body.stats.insert(0, self.assign(temp, strides))
        return temp

    def function(self, name, body, args, shapevar=None, posinfo=None,
                 omp_size=None):
        function = super(DynamicArgumentASTBuilder, self).function(
                        name, body, args, shapevar, posinfo, omp_size)

        function.data_pointers = self.variable(
                    minitypes.void.pointer().pointer(), 'data_pointers')
        function.strides_pointers = self.variable(
                    function.shape.type.pointer(), 'strides_pointer')

        i = len(function.arrays) - 1
        for argument in function.arrays[::-1]:
            data_p = self._create_data_pointer(function, argument, i)
            strides_p = self._create_strides_pointer(function, argument, i)

            argument.data_pointer = data_p
            argument.strides_pointer = strides_p

            argument.used = False
            i -= 1

        argpos = 1
        if posinfo:
            argpos = 4

        function.arguments.insert(argpos,
                                  self.funcarg(function.strides_pointers))
        function.arguments.insert(argpos,
                                  self.funcarg(function.data_pointers))

        self.create_function_type(function)
        # print function.type
        # print self.context.debug_c(
        #        function, specializers.StridedSpecializer, type(self))
        return function

Context.astbuilder_cls = UndocClassAttribute(ASTBuilder)

class Position(object):
    "Each node has a position which is an instance of this type."

    def __init__(self, filename, line, col):
        self.filename = filename
        self.line = line
        self.col = col

    def __str__(self):
        return "%s:%d:%d" % (self.filename, self.line, self.col)

class Node(miniutils.ComparableObjectMixin):
    """
    Base class for AST nodes.
    """

    is_expression = False

    is_statlist = False
    is_constant = False
    is_assignment = False
    is_unop = False
    is_binop = False

    is_node_wrapper = False
    is_data_pointer = False
    is_jump = False
    is_label = False
    is_temp = False
    is_statement = False
    is_sizeof = False
    is_variable = False

    is_function = False
    is_funcarg = False
    is_array_funcarg = False

    is_specialized = False

    child_attrs = []

    def __init__(self, pos, **kwds):
        self.pos = pos
        vars(self).update(kwds)

    def may_error(self, context):
        """
        Return whether something may go wrong and we need to jump to an
        error handler.
        """
        visitor = minivisitor.MayErrorVisitor(context)
        visitor.visit(self)
        return visitor.may_error

    def print_tree(self, context):
        visitor = minivisitor.PrintTree(context)
        visitor.visit(self)

    @property
    def children(self):
        return [getattr(self, attr) for attr in self.child_attrs
                    if getattr(self, attr) is not None]

    @property
    def comparison_objects(self):
        type = getattr(self, 'type', None)
        if type is None:
            return self.children
        return tuple(self.children) + (type,)

    def __eq__(self, other):
        # Don't use isinstance here, compare on exact type to be consistent
        # with __hash__. Override where sensible
        return (type(self) is type(other) and
                self.comparison_objects == other.comparison_objects)

    def __hash__(self):
        h = hash(type(self))
        for obj in self.comparison_objects:
            h = h ^ hash(obj)

        return h

class ExprNode(Node):
    "Base class for expressions. Each node has a type."

    is_expression = True

    hoistable = False
    need_temp = False

    def __init__(self, pos, type, **kwds):
        super(ExprNode, self).__init__(pos, **kwds)
        self.type = type

class FunctionNode(Node):
    """
    Function node. error_value and success_value are returned in case of
    exceptions and success respectively.

    .. attribute:: shape

        the broadcast shape for all operands

    .. attribute:: ndim

        the ndim of the total broadcast' shape

    .. attribute:: arguments

        all array arguments

    .. attribute:: scalar arguments

        all non-array arguments

    .. attribute:: posinfo

        the position variables we can write to in case of an exception

    .. attribute:: omp_size

        the threshold of minimum data size needed before starting a parallel
        section. May be overridden at any time before specialization time.
    """

    is_function = True

    child_attrs = ['body', 'arguments', 'scalar_arguments']

    def __init__(self, pos, name, body, arguments, scalar_arguments,
                 shape, posinfo, error_value, success_value, omp_size):
        super(FunctionNode, self).__init__(pos)
        self.type = None # see ASTBuilder.create_function_type
        self.name = name
        self.body = body
        self.arrays = [arg for arg in arguments if arg.type and arg.type.is_array]
        self.arguments = arguments
        self.scalar_arguments = scalar_arguments
        self.shape = shape
        self.posinfo = posinfo
        self.error_value = error_value
        self.success_value = success_value
        self.omp_size = omp_size

        self.args = dict((v.name, v) for v in arguments)
        self.ndim = max(arg.type.ndim for arg in arguments
                                          if arg.type and arg.type.is_array)

class FuncCallNode(ExprNode):
    """
    Call a function given a pointer or its name (FuncNameNode)
    """

    inline = False
    child_attrs = ['func_or_pointer', 'args']

class FuncNameNode(ExprNode):
    """
    Load an external function by its name.
    """
    name = None

class ReturnNode(Node):
    "Return an operand"

    child_attrs = ['operand']

    def __init__(self, pos, operand):
        super(ReturnNode, self).__init__(pos)
        self.operand = operand

class RaiseNode(Node):
    "Raise a Python exception. The callee must hold the GIL."

    child_attrs = ['posinfo', 'exc_var', 'msg_val', 'fmt_args']

    def __init__(self, pos, posinfo, exc_var, msg_val, fmt_args):
        super(RaiseNode, self).__init__(pos)
        self.posinfo = posinfo
        self.exc_var, self.msg_val, self.fmt_args = (exc_var, msg_val, fmt_args)

class PositionInfoNode(Node):
    """
    Node that holds a position of where an error occurred. This position
    needs to be returned to the callee if the callee supports it.
    """

class FunctionArgument(ExprNode):
    """
    Argument to the FunctionNode. Array arguments contain multiple
    actual arguments, e.g. the data and stride pointer.

    .. attribute:: variable

        some argument to the function (array or otherwise)

    .. attribute:: variables

        the actual variables this operand should be unpacked into
    """
    child_attrs = ['variables']
    if_funcarg = True

    used = True

    def __init__(self, pos, variable, variables):
        super(FunctionArgument, self).__init__(pos, variable.type)
        self.variables = variables
        self.variable = variable
        self.name = variable.name
        self.args = dict((v.name, v) for v in variables)

class ArrayFunctionArgument(ExprNode):
    "Array operand to the function"

    child_attrs = ['data_pointer', 'strides_pointer']
    is_array_funcarg = True

    used = True

    def __init__(self, pos, type, data_pointer, strides_pointer, **kwargs):
        super(ArrayFunctionArgument, self).__init__(pos, type, **kwargs)
        self.data_pointer = data_pointer
        self.strides_pointer = strides_pointer
        self.variables = [data_pointer, strides_pointer]

class PrintNode(Node):
    "Print node for some arguments"

    child_attrs = ['args']

class NDIterate(Node):
    """
    Iterate in N dimensions. See :py:class:`ASTBuilder.nditerate`
    """

    child_attrs = ['body']

    def __init__(self, pos, body):
        super(NDIterate, self).__init__(pos)
        self.body = body

class ForNode(Node):
    """
    A for loop, see :py:class:`ASTBuilder.for_`
    """

    child_attrs = ['init', 'condition', 'step', 'body']

    is_controlling_loop = False
    is_tiling_loop = False

    should_vectorize = False
    is_fixup = False

    def __init__(self, pos, init, condition, step, body, index=None):
        super(ForNode, self).__init__(pos)
        self.init = init
        self.condition = condition
        self.step = step
        self.body = body

        self.index = index or init.lhs

class IfNode(Node):
    "An 'if' statement, see A for loop, see :py:class:`ASTBuilder.if_`"

    child_attrs = ['cond', 'body', 'else_body']

    should_vectorize = False
    is_fixup = False

class StatListNode(Node):
    """
    A node to wrap multiple statements, see :py:class:`ASTBuilder.stats`
    """
    child_attrs = ['stats']
    is_statlist = True

    def __init__(self, pos, statements):
        super(StatListNode, self).__init__(pos)
        self.stats = statements

class ExprStatNode(Node):
    "Turn an expression into a statement, see :py:class:`ASTBuilder.expr_stat`"
    child_attrs = ['expr']
    is_statement = True

class ExprNodeWithStatement(Node):
    child_attrs = ['stat', 'expr']

class NodeWrapper(ExprNode):
    """
    Adapt an opaque node to provide a consistent interface. This has to be
    handled by the user's specializer. See :py:class:`ASTBuilder.wrap`
    """

    is_node_wrapper = True
    is_constant_scalar = False

    child_attrs = []

    def __init__(self, pos, type, opaque_node, specialize_node_callback,
                 **kwds):
        super(NodeWrapper, self).__init__(pos, type)
        self.opaque_node = opaque_node
        self.specialize_node_callback = specialize_node_callback
        vars(self).update(kwds)

    def __hash__(self):
        return hash(self.opaque_node)

    def __eq__(self, other):
        if getattr(other, 'is_node_wrapper ', False):
            return self.opaque_node == other.opaque_node

        return NotImplemented

    def __deepcopy__(self, memo):
        kwds = dict(vars(self))
        kwds.pop('opaque_node')
        kwds = copy.deepcopy(kwds, memo)
        opaque_node = self.specialize_node_callback(self, memo)
        return type(self)(opaque_node=opaque_node, **kwds)

class BinaryOperationNode(ExprNode):
    "Base class for binary operations"
    child_attrs = ['lhs', 'rhs']
    def __init__(self, pos, type, lhs, rhs, **kwds):
        super(BinaryOperationNode, self).__init__(pos, type, **kwds)
        self.lhs, self.rhs = lhs, rhs

class BinopNode(BinaryOperationNode):
    "Node for binary operations"

    is_binop = True

    def __init__(self, pos, type, operator, lhs, rhs, **kwargs):
        super(BinopNode, self).__init__(pos, type, lhs, rhs, **kwargs)
        self.operator = operator

    @property
    def comparison_objects(self):
        return (self.operator, self.lhs, self.rhs)

class SingleOperandNode(ExprNode):
    "Base class for operations with one operand"
    child_attrs = ['operand']
    def __init__(self, pos, type, operand, **kwargs):
        super(SingleOperandNode, self).__init__(pos, type, **kwargs)
        self.operand = operand

class AssignmentExpr(BinaryOperationNode):
    is_assignment = True

class IfElseExprNode(ExprNode):
    child_attrs = ['cond', 'lhs', 'rhs']

class PromotionNode(SingleOperandNode):
    pass

class UnopNode(SingleOperandNode):

    is_unop = True

    def __init__(self, pos, type, operator, operand, **kwargs):
        super(UnopNode, self).__init__(pos, type, operand, **kwargs)
        self.operator = operator

    @property
    def comparison_objects(self):
        return (self.operator, self.operand)

class CastNode(SingleOperandNode):
    is_cast = True

class DereferenceNode(SingleOperandNode):
    is_dereference = True

class SingleIndexNode(BinaryOperationNode):
    is_index = True

class ConstantNode(ExprNode):
    is_constant = True
    def __init__(self, pos, type, value):
        super(ConstantNode, self).__init__(pos, type)
        self.value = value

class SizeofNode(ExprNode):
    is_sizeof = True

class Variable(ExprNode):
    """
    Represents use of a function argument in the function.
    """

    is_variable = True
    mangled_name = None

    hoisted = False

    def __init__(self, pos, type, name, **kwargs):
        super(Variable, self).__init__(pos, type, **kwargs)
        self.name = name
        self.array_type = None

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

class ResolvedVariable(Variable):
    child_attrs = ['element']

    def __eq__(self, other):
        return (isinstance(other, ResolvedVariable) and
                self.element == other.element)

class ArrayAttribute(Variable):
    "Denotes an attribute of array operands, e.g. the data or stride pointers"
    def __init__(self, pos, type, arrayvar):
        super(ArrayAttribute, self).__init__(pos, type,
                                             arrayvar.name + self._name)
        self.arrayvar = arrayvar

class DataPointer(ArrayAttribute):
    "Reference to the start of an array operand"
    _name = '_data'

class StridePointer(ArrayAttribute):
    "Reference to the stride pointer of an array variable operand"
    _name = '_strides'

#class ShapePointer(ArrayAttribute):
#    "Reference to the shape pointer of an array operand."
#    _name = '_shape'

class TempNode(Variable):
    "A temporary of a certain type"

    is_temp = True

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

class OpenMPLoopNode(Node):
    """
    Execute a loop in parallel.
    """
    child_attrs = ['for_node', 'if_clause', 'lastprivates', 'privates']

class OpenMPConditionalNode(Node):
    """
    Execute if_body if _OPENMP, otherwise execute else_body.
    """
    child_attrs = ['if_body', 'else_body']

class PragmaForLoopNode(Node):
    """
    Generate compiler-specific pragmas to aid things like SIMDization.
    """
    child_attrs = ['for_node']

class ErrorHandler(Node):
    """
    A node to handle errors. If there is an error handler in the outer scope,
    the specializer will first make this error handler generate disposal code
    for the wrapped AST body, and then jump to the error label of the parent
    error handler. At the outermost (function) level, the error handler simply
    returns an error indication.

    .. attribute:: error_label

        point to jump to in case of an error

    .. attribute:: cleanup_label

        point to jump to in the normal case

    It generates the following:

    .. code-block:: c

        error_var = 0;
        ...
        goto cleanup;
      error:
        error_var = 1;
      cleanup:
        ...
        if (error_var)
            goto outer_error_label;
    """
    child_attrs = ['error_var_init', 'body', 'cleanup_jump',
                   'error_target_label', 'error_set', 'cleanup_target_label',
                   'cascade']

    error_var_init = None
    cleanup_jump = None
    error_target_label = None
    error_set = None
    cleanup_target_label = None
    cascade = None

class JumpNode(Node):
    "A jump to a jump target"
    child_attrs = ['label']
    def __init__(self, pos, label):
        Node.__init__(self, pos)
        self.label = label

class JumpTargetNode(JumpNode):
    "A point to jump to"

class LabelNode(ExprNode):
    "A goto label or memory address that we can jump to"

    def __init__(self, pos, name):
        super(LabelNode, self).__init__(pos, None)
        self.name = name
        self.mangled_name = None

class NoopExpr(ExprNode):
    "Do nothing expression"

#
### Vectorization Functionality
#

class VectorVariable(Variable):
    child_attrs = ['variable']

class VectorLoadNode(SingleOperandNode):
    "Load a SIMD vector"

class VectorStoreNode(BinopNode):
    "Store a SIMD vector"

class VectorBinopNode(BinopNode):
    "Binary operation on SIMD vectors"

class VectorUnopNode(SingleOperandNode):
    "Unary operation on SIMD vectors"

class ConstantVectorNode(ExprNode):
    "Load the constant into the vector register"