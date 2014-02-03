# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast

from numba.templating import temp_name
from numba import error, pipeline, nodes, ufunc_builder
from numba.minivect import specializers, miniast, miniutils, minitypes
from numba import utils, functions
from numba import typesystem
from numba import visitors

from numba.support.numpy_support import slicenodes
from numba.vectorize import basic

import llvm.core

print_ufunc = False

# ______________________________________________________________________

def is_elementwise_assignment(assmnt_node):
    target_type = assmnt_node.targets[0].type
    value_type = assmnt_node.value.type

    if target_type.is_array:
        # Allow arrays and scalars
        return value_type.is_array or not value_type.is_object

    return False

# ______________________________________________________________________

def get_py_ufunc_ast(env, lhs, node):
    if lhs is not None:
        lhs.ctx = ast.Load()

    builder = ufunc_builder.UFuncConverter(env)
    tree = builder.visit(node)
    ufunc_ast = builder.build_ufunc_ast(tree)

    if print_ufunc:
        from meta import asttools
        module = ast.Module(body=[ufunc_ast])
        print((asttools.python_source(module)))

    # Vectorize Python function
    if lhs is None:
        restype = node.type
    else:
        restype = lhs.type.dtype

    argtypes = [op.type.dtype if op.type.is_array else op.type
                    for op in builder.operands]
    signature = restype(*argtypes)

    return ufunc_ast, signature, builder

def get_py_ufunc(env, lhs, node):
    ufunc_ast, signature, ufunc_builder = get_py_ufunc_ast(env, lhs, node)
    py_ufunc = ufunc_builder.compile_to_pyfunc(ufunc_ast)
    return py_ufunc, signature, ufunc_builder

# ______________________________________________________________________

class ArrayExpressionRewrite(visitors.NumbaTransformer):
    """
    Find element-wise expressions and run ElementalMapper to turn it into
    a minivect AST or a ufunc.
    """

    nesting_level = 0
    elementwise = False

    is_slice_assign = False

    def register_array_expression(self, node, lhs=None):
        """
        Start the mapping process for the outermost node in the array expression.
        """
        self.elementwise = False

    def visit_elementwise(self, elementwise, node):
        if elementwise and self.nesting_level == 0:
            return self.register_array_expression(node)

        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        self.elementwise = elementwise
        return node

    def visit_Assign(self, node):
        self.is_slice_assign = False
        self.visitlist(node.targets)

        target_node = node.targets[0]
        is_slice_assign = self.is_slice_assign

        self.nesting_level = self.is_slice_assign
        node.value = self.visit(node.value)
        self.nesting_level = 0

        elementwise = self.elementwise
        if (len(node.targets) == 1 and is_slice_assign and
                is_elementwise_assignment(node)): # and elementwise):
            target_node = slicenodes.rewrite_slice(target_node,
                                                      self.nopython)
            return self.register_array_expression(node.value, lhs=target_node)

        return node

    def visit_Subscript(self, node):
        # print ast.dump(node)
        self.generic_visit(node)
        is_store = isinstance(node.ctx, ast.Store)
        self.is_slice_assign = is_store and node.type.is_array

        if is_store:
            if nodes.is_ellipsis(node.slice):
                return node.value
        elif node.value.type.is_array and node.type.is_array:
            node = slicenodes.rewrite_slice(node, self.nopython)

        return node

    def visit_Call(self, node):
        if self.query(node, 'is_math'):
            elementwise = node.type.is_array
            return self.visit_elementwise(elementwise, node)

        self.visitchildren(node)
        return node

    def visit_BinOp(self, node):
        elementwise = node.type.is_array
        return self.visit_elementwise(elementwise, node)

    visit_UnaryOp = visit_BinOp

# ______________________________________________________________________

class ArrayExpressionRewriteUfunc(ArrayExpressionRewrite):
    """
    Compile array expressions to ufuncs. Then call the ufunc with the array
    arguments.

        vectorizer_cls: the ufunc vectorizer to use

    CANNOT be used in a nopython context
    """

    def __init__(self, context, func, ast, vectorizer_cls=None):
        super(ArrayExpressionRewriteUfunc, self).__init__(context, func, ast)
        self.vectorizer_cls = vectorizer_cls or basic.BasicASTVectorize

    def register_array_expression(self, node, lhs=None):
        super(ArrayExpressionRewriteUfunc, self).register_array_expression(node,
                                                                           lhs)
        py_ufunc, signature, ufunc_builder = self.get_py_ufunc(lhs, node)

        # Vectorize Python function
        vectorizer = self.vectorizer_cls(py_ufunc)
        vectorizer.add(restype=signature.return_type, argtypes=signature.args)
        ufunc = vectorizer.build_ufunc()

        # Call ufunc
        args = ufunc_builder.operands
        if lhs is None:
            keywords = None
        else:
            keywords = [ast.keyword('out', lhs)]

        func = nodes.ObjectInjectNode(ufunc)
        call_ufunc = nodes.ObjectCallNode(signature=None, func=func, args=args,
                                          keywords=keywords, py_func=ufunc)
        return nodes.ObjectTempNode(call_ufunc)

# ______________________________________________________________________

class NumbaStaticArgsContext(utils.NumbaContext):
    "Use a static argument list: shape, data1, strides1, data2, strides2, ..."

    astbuilder_cls = miniast.ASTBuilder
    optimize_llvm = False
    optimize_broadcasting = False
    # debug = True
    # debug_elements = True

    def init(self):
        self.astbuilder = self.astbuilder_cls(self)
        self.typemapper = minitypes.TypeMapper(self)

    # def promote_types(self, t1, t2):
    #     return typesystem.promote(t1, t2)
    #
    def to_llvm(self, type):
        if type.is_object:
            return typesystem.object_.to_llvm(self)
        return NotImplementedError("to_llvm", type)

# ______________________________________________________________________

class ArrayExpressionRewriteNative(ArrayExpressionRewrite):
    """
    Compile array expressions to a minivect kernel that calls a Numba
    scalar kernel with scalar inputs:

        a[:, :] = b[:, :] * c[:, :]

    becomes

        tmp_a = slice(a)
        tmp_b = slice(b)
        tmp_c = slice(c)
        shape = broadcast(tmp_a, tmp_b, tmp_c)
        call minikernel(shape, tmp_a.data, tmp_a.strides,
                               tmp_b.data, tmp_b.strides,
                               tmp_c.data, tmp_c.strides)

    with

        def numba_kernel(b, c):
            return b * c

        def minikernel(...):
            for (...)
                for(...)
                    a[i, j] = numba_kernel(b[i, j], c[i, j])

    CAN be used in a nopython context
    """

    expr_count = 0

    def array_attr(self, node, attr):
        # Perform a low-level bitcast from object to an array type
        # array = nodes.CoercionNode(node, float_[:])
        array = node
        return nodes.ArrayAttributeNode(attr, array)

    def register_array_expression(self, node, lhs=None):
        super(ArrayExpressionRewriteNative, self).register_array_expression(
            node, lhs)

        # llvm_module = llvm.core.Module.new(temp_name("array_expression_module"))
        # llvm_module = self.env.llvm_context.module

        lhs_type = lhs.type if lhs else node.type
        is_expr = lhs is None

        if node.type.is_array and lhs_type.ndim < node.type.ndim:
            # TODO: this is valid in NumPy if the leading dimensions of the
            # TODO: RHS have extent 1
            raise error.NumbaError(
                node, "Right hand side must have a "
                      "dimensionality <= %d" % lhs_type.ndim)

        # Create ufunc scalar kernel
        ufunc_ast, signature, ufunc_builder = get_py_ufunc_ast(self.env, lhs, node)

        # Compile ufunc scalar kernel with numba
        ast.fix_missing_locations(ufunc_ast)
        # func_env = self.env.crnt.inherit(
        #     func=None, ast=ufunc_ast, func_signature=signature,
        #     wrap=False, #link=False, #llvm_module=llvm_module,
        # )
        # pipeline.run_env(self.env, func_env) #, pipeline_name='codegen')

        func_env, (_, _, _) = pipeline.run_pipeline2(
            self.env, None, ufunc_ast, signature,
            function_globals=self.env.crnt.function_globals,
            wrap=False, link=False, nopython=True,
            #llvm_module=llvm_module, # pipeline_name='codegen',
        )
        llvm_module = func_env.llvm_module

        operands = ufunc_builder.operands
        operands = [nodes.CloneableNode(operand) for operand in operands]

        if lhs is not None:
            lhs = nodes.CloneableNode(lhs)
            broadcast_operands = [lhs] + operands
            lhs = lhs.clone
        else:
            broadcast_operands = operands[:]

        shape = slicenodes.BroadcastNode(lhs_type, broadcast_operands)
        operands = [op.clone for op in operands]

        if lhs is None and self.nopython:
            raise error.NumbaError(
                node, "Cannot allocate new memory in nopython context")
        elif lhs is None:
            # TODO: determine best output order at runtime
            shape = shape.cloneable
            lhs = nodes.ArrayNewEmptyNode(lhs_type, shape.clone,
                                          lhs_type.is_f_contig).cloneable

        # Build minivect wrapper kernel
        context = NumbaStaticArgsContext()
        context.llvm_module = llvm_module
        # context.llvm_ee = self.env.llvm_context.execution_engine

        b = context.astbuilder
        variables = [b.variable(name_node.type, "op%d" % i)
                     for i, name_node in enumerate([lhs] + operands)]
        miniargs = [b.funcarg(variable) for variable in variables]
        body = miniutils.build_kernel_call(func_env.lfunc.name, signature,
                                           miniargs, b)

        minikernel = b.function_from_numpy(
            temp_name("array_expression"), body, miniargs)
        lminikernel, = context.run_simple(minikernel,
                                          specializers.StridedSpecializer)
        # lminikernel.linkage = llvm.core.LINKAGE_LINKONCE_ODR

        # pipeline.run_env(self.env, func_env, pipeline_name='post_codegen')
        # llvm_module.verify()
        del func_env

        assert lminikernel.module is llvm_module
        # print("---------")
        # print(llvm_module)
        # print("~~~~~~~~~~~~")
        lminikernel = self.env.llvm_context.link(lminikernel)

        # Build call to minivect kernel
        operands.insert(0, lhs)
        args = [shape]
        scalar_args = []
        for operand in operands:
            if operand.type.is_array:
                data_p = self.array_attr(operand, 'data')
                data_p = nodes.CoercionNode(data_p,
                                            operand.type.dtype.pointer())
                if not isinstance(operand, nodes.CloneNode):
                    operand = nodes.CloneNode(operand)
                strides_p = self.array_attr(operand, 'strides')
                args.extend((data_p, strides_p))
            else:
                scalar_args.append(operand)

        args.extend(scalar_args)
        result = nodes.NativeCallNode(minikernel.type, args, lminikernel)

        # Use native slicing in array expressions
        slicenodes.mark_nopython(ast.Suite(body=result.args))

        if not is_expr:
            # a[:] = b[:] * c[:]
            return result

        # b[:] * c[:], return new array as expression
        return nodes.ExpressionNode(stmts=[result], expr=lhs.clone)
