"""
Specializers for various sorts of data layouts and memory alignments.

These specializers operate on a copy of the simplified array expression
representation (i.e., one with an NDIterate node). This node is replaced
with one or several ForNode nodes in a specialized order.

For auto-tuning code for tile size and OpenMP size, see
https://github.com/markflorisson88/cython/blob/_array_expressions/Cython/Utility/Vector.pyx
"""

import sys
import copy
import functools

import minivisitor
import miniutils
import minitypes
import minierror
import codegen

strength_reduction = True

def debug(*args):
    sys.stderr.write(" ".join(str(arg) for arg in args) + '\n')

def specialize_ast(ast):
    return copy.deepcopy(ast)

class ASTMapper(minivisitor.VisitorTransform):
    """
    Base class to map foreign ASTs onto a minivect AST, or vice-versa.
    This sets the current node's position in the astbuilder for each
    node that is being visited, to make it easy to build new AST nodes
    without passing in source position information everywhere.
    """

    def __init__(self, context):
        super(ASTMapper, self).__init__(context)
        self.astbuilder = context.astbuilder

    def getpos(self, opaque_node):
        return self.context.getpos(opaque_node)

    def map_type(self, opaque_node, **kwds):
        "Return a mapped type for the foreign node."
        return self.context.typemapper.map_type(
                        self.context.gettype(opaque_node), **kwds)

    def visit(self, node, *args):
        prev = self.astbuilder.pos
        self.astbuilder.pos = node.pos
        result = super(ASTMapper, self).visit(node)
        self.astbuilder.pos = prev
        return result

class BaseSpecializer(ASTMapper):
    """
    Base class for specialization. Does not perform any specialization itself.
    """

    def getpos(self, node):
        return node.pos

    def get_type(self, type):
        "Resolve the type to the dtype of the array if an array type"
        if type.is_array:
            return type.dtype
        return type

    def visit(self, node, *args):
        result = super(BaseSpecializer, self).visit(node)
        if result is not None:
            result.is_specialized = True
        return result

    def visit_Node(self, node):
        # node = copy.copy(node)
        self.visitchildren(node)
        return node

    def init_pending_stats(self, node):
        """
        Allow modifications while visiting some descendant of this node
        This happens especially while variables are resolved, which
        calls compute_inner_dim_pointer()
        """
        b = self.astbuilder

        if not node.is_function:
            node.prepending = b.stats()
        node.appending = b.stats()

    def handle_pending_stats(self, node):
        """
        Handle any pending statements that need to be inserted further
        up in the AST.
        """
        b = self.astbuilder

        # self.visitchildren(node.prepending)
        # self.visitchildren(node.appending)
        if node.is_function:
            # prepending is a StatListNode already part of the function body
            # assert node.prepending in list(self.treepath(node, '//StatListNode'))
            node.body = b.stats(node.body, node.appending)
        else:
            node.body = b.stats(node.prepending, node.body, node.appending)

        if not self.context.use_llvm:
            node.body = self.fuse_omp_stats(node.body)

    def get_loop(self, loop_level):
        if loop_level:
            return self.function.for_loops[self.loop_level - 1]
        return self.function

    def fuse_omp_stats(self, node):
        """
        Fuse consecutive OpenMPConditionalNodes.
        """
        import miniast

        if not node.stats:
            return node

        b = self.astbuilder
        stats = [node.stats[0]]
        for next_stat in node.stats[1:]:
            stat = stats[-1]
            c1 = isinstance(stat, miniast.OpenMPConditionalNode)
            c2 = isinstance(next_stat, miniast.OpenMPConditionalNode)

            if c1 and c2:
                if_body = None
                else_body = None

                if stat.if_body or next_stat.if_body:
                    if_body = b.stats(stat.if_body, next_stat.if_body)
                if stat.else_body or next_stat.else_body:
                    else_body = b.stats(stat.else_body, next_stat.else_body)

                stats[-1] = b.omp_if(if_body, else_body)
            else:
                stats.append(next_stat)

        node.stats[:] = stats
        return node

    #
    ### Stubs for cooperative multiple inheritance
    #

    def visit_NDIterate(self, node):
        # Do not visit children
        return node

    visit_AssignmentExpr = visit_Node
    visit_ErrorHandler = visit_Node
    visit_BinopNode = visit_Node
    visit_UnopNode = visit_Node
    visit_IfNode = visit_Node

class Specializer(BaseSpecializer):
    """
    Base class for most specializers, provides some basic functionality
    for subclasses. Implement visit_* methods to specialize nodes
    to some pattern.

    Implements implementations to handle errors and cleanups, adds a return
    statement to the function and can insert debug print statements if
    context.debug is set to a true value.
    """

    is_contig_specializer = False
    is_tiled_specializer = False
    is_vectorizing_specializer = False
    is_inner_contig_specializer = False
    is_strided_specializer = False

    vectorized_equivalents = None

    def __init__(self, context, specialization_name=None):
        super(Specializer, self).__init__(context)
        if specialization_name is not None:
            self.specialization_name = specialization_name
        self.variables = {}

    def _index_list(self, pointer, ndim):
        "Return a list of indexed pointers"
        return [self.astbuilder.index(pointer, self.astbuilder.constant(i))
                    for i in range(ndim)]

    def _debug_function_call(self, b, node):
        """
        Generate debug print statements when the specialized function is
        called.
        """
        stats = [
            b.print_(b.constant(
                "Calling function %s (%s specializer)" % (
                                node.mangled_name, self.specialization_name)))
        ]
        if self.is_vectorizing_specializer:
            stats.append(
                b.print_(b.constant("Vectorized version size=%d" %
                                                        self.vector_size)))

        stats.append(
            b.print_(b.constant("shape:"), *self._index_list(node.shape,
                                                             node.ndim)))
        if self.is_tiled_specializer:
            stats.append(b.print_(b.constant("blocksize:"), self.get_blocksize()))

        if not self.is_contig_specializer:
            for idx, arg in enumerate(node.arguments):
                if arg.is_array_funcarg:
                    stats.append(b.print_(b.constant("strides operand%d:" % idx),
                                          *self._index_list(arg.strides_pointer,
                                                            arg.type.ndim)))
                    stats.append(b.print_(b.constant("data pointer %d:" % idx),
                                          arg.data_pointer))

        node.prepending.stats.append(b.stats(*stats))

    def visit_FunctionNode(self, node):
        """
        Handle a FunctionNode. Sets node.total_shape to the product of the
        shape, wraps the function's body in a
        :py:class:`minivect.miniast.ErrorHandler` if needed and adds a
        return statement.
        """
        b = self.astbuilder
        self.compute_total_shape(node)

        node.mangled_name = self.context.mangle_function_name(node.name)

        # set this so bad people can specialize during code generation time
        node.specializer = self
        node.specialization_name = self.specialization_name
        self.function = node

        if self.context.debug:
            self._debug_function_call(b, node)

        if node.body.may_error(self.context):
            node.body = b.error_handler(node.body)

        node.body = b.stats(node.body, b.return_(node.success_value))

        self.visitchildren(node)

#        if not self.is_contig_specializer:
#            self.compute_temp_strides(b, node)

        return node

    def visit_ForNode(self, node):
        if node.body.may_error(self.context):
            node.body = self.astbuilder.error_handler(node.body)
        self.visitchildren(node)
        return node

    def visit_Variable(self, node):
        if node.name not in self.variables:
            self.variables[node.name] = node
        return self.visit_Node(node)

    def get_data_pointer(self, variable, loop_level):
        return self.function.args[variable.name].data_pointer

    def omp_for(self, node):
        """
        Insert an OpenMP for loop with an 'if' clause that checks to see
        whether the total data size exceeds the given OpenMP auto-tuned size.
        The caller needs to adjust the size, set in the FunctionNode's
        'omp_size' attribute, depending on the number of computations.
        """
        if_clause = self.astbuilder.binop(minitypes.bool_, '>',
                                          self.function.total_shape,
                                          self.function.omp_size)
        return self.astbuilder.omp_for(node, if_clause)

class FinalSpecializer(BaseSpecializer):
    """
    Perform any final specialization and optimizations. The initial specializer
    is concerned with specializing for the given data layouts, whereas this
    specializer is concerned with any rewriting of the AST to support
    fundamental operations.
    """

    vectorized_equivalents = None
    in_lhs_expr = False
    should_vectorize = False

    def __init__(self, context, previous_specializer):
        super(FinalSpecializer, self).__init__(context)
        self.previous_specializer = previous_specializer
        self.sp = previous_specializer

        self.error_handlers = []
        self.loop_level = 0

        self.variables = {}
        self.strides = {}
        self.outer_pointers = {}
        self.vector_temps = {}

    def run_optimizations(self, node):
        """
        Run any optimizations on the AST. Currently only loop-invariant code
        motion is implemented when broadcasting information is present.
        """
        import optimize

        # TODO: support vectorized specializations
        if (self.context.optimize_broadcasting and not
                self.sp.is_contig_specializer or
                self.sp.is_vectorizing_specializer):
            optimizer = optimize.HoistBroadcastingExpressions(self.context)
            node = optimizer.visit(node)

        return node

    def visit_Variable(self, node):
        """
        Process variables, which includes arrays and scalars. For arrays,
        this means retrieving the element from the array. Performs strength
        reduction for index calculation of array variables.
        """
        if node.type.is_array:
            tiled = self.sp.is_tiled_specializer
            last_loop_level = (self.loop_level == self.function.ndim or
                               (self.sp.is_vectorizing_specializer and not
                                self.should_vectorize))
            inner_contig = (
                self.sp.is_inner_contig_specializer and
                (last_loop_level or node.hoisted) and
                (not self.sp.is_strided_specializer or
                 self.sp.matching_contiguity(node.type)))

            contig = self.sp.is_contig_specializer

            # Get the array data pointer
            arg_data_pointer = self.function.args[node.name].data_pointer
            if self.sp.is_contig_specializer:
                # Contiguous, no strength reduction needed
                data_pointer = arg_data_pointer
            else:
                # Compute strength reduction pointers for all dimensions leading
                # up the the dimension this variable occurs in.
                self.compute_temp_strides(node, inner_contig, tiled=tiled)
                data_pointer = self.compute_data_pointer(
                            node, arg_data_pointer, inner_contig, tiled)

            # Get the loop level corresponding to the occurrence of the variable
            for_node = self.function.for_loops[self.loop_level - 1]

            if self.should_vectorize:
                return self.handle_vector_variable(node, data_pointer, for_node,
                                                   inner_contig, contig)
            else:
                element = self.element_location(data_pointer, for_node,
                                                inner_contig, contig,
                                                tiled=tiled, variable=node)
                return self.astbuilder.resolved_variable(
                                node.name, node.type, element)
        else:
            return node

    def visit_VectorVariable(self, vector_variable):
        # use visit_Variable, since is does the strength reduction and such
        return self.visit_Variable(vector_variable.variable)

    def element_location(self, data_pointer, for_node,
                         inner_contig, is_contig, tiled, variable):
        "Return the element in the array for the current index set"
        b = self.astbuilder

        def debug(item):
            if self.context.debug_elements:
                string = b.constant("Referenced element from %s:" %
                                                        variable.name)
                print_ = self.visit(b.print_(string, item))
                for_node = self.function.for_loops[self.loop_level - 1]
                for_node.prepending.stats.append(print_)

                if not is_contig:
                    stats = []
                    for i, stride in enumerate(self.strides[variable]):
                        if stride is not None:
                            string = b.constant("%s step[%d]:" % (variable.name, i))
                            stats.append(b.print_(string, stride))

                    print_steps = b.stats(*stats)
                    self.function.prepending.stats.append(self.visit(print_steps))

            return item

        if inner_contig or is_contig:
            # contiguous access, index the data pointer in the inner dimension
            return debug(b.index(data_pointer, for_node.index))
        else:
            # strided access, this dimension is performing strength reduction,
            # so we just need to dereference the data pointer
            return debug(b.dereference(data_pointer))

    def handle_vector_variable(self, variable, data_pointer, for_node,
                               inner_contig, is_contig):
        "Same as `element_location`, except for Vector variables"
        b = self.astbuilder

        # For array operands, load reads into registers, and store
        # writes back into the data pointer. For assignment to a register
        # we use a vector type, for assignment to a data pointer, the
        # data pointer type

        if inner_contig or is_contig:
            data_pointer = b.add(data_pointer, for_node.index)

        if self.in_lhs_expr:
            return data_pointer
        else:
            variable = b.vector_variable(variable, self.sp.vector_size)
            if variable in self.vector_temps:
                return self.vector_temps[variable]

            rhs = b.vector_load(data_pointer, self.sp.vector_size)
            temp = b.temp(variable.type, 'xmm')
            self.vector_temps[variable] = temp

            for_node.prepending.stats.append(b.assign(temp, rhs))

            return self.visit(temp)

    def compute_temp_strides(self, variable, handle_inner_dim, tiled=False):
        """
        Compute the temporary strides needed for the strength reduction. These
        should be small constants, so division should be fast. We could use
        char * instead of element_type *, but it's nicer to avoid the casts.
        """
        b = self.astbuilder

        if variable in self.strides:
            return self.strides[variable]

        start = 0
        stop = variable.type.ndim
        if handle_inner_dim:
            if self.sp.order == "F":
                start = 1
            else:
                stop = stop - 1

        self.strides[variable] = strides = [None] * len(self.function.for_loops)

        for dim in range(start, stop):
            stride = b.stride(variable, dim)
            temp_stride = b.temp(stride.type.unqualify("const"),
                                 name="%s_stride%d" % (variable.name, dim))

            stat = b.assign(temp_stride,
                            b.div(stride, b.sizeof(variable.type.dtype)))
            self.function.prepending.stats.append(stat)
            strides[dim] = temp_stride

        return strides

    def compute_data_pointer(self, variable, argument_data_pointer,
                             handle_inner_dim, tiled):
        """
        Compute the data pointer for the dimension the variable is located in
        (the loop level). This involves generating a strength reduction in
        each outer dimension.

        Variables referring to the same array may be found on different
        loop levels.
        """
        b = self.astbuilder

        assert variable.type.is_array
        pointer_type = argument_data_pointer.type.unqualify("const")
        loop_level = self.loop_level

        offset = self.function.ndim - variable.type.ndim
        stop = loop_level - handle_inner_dim
        if self.outer_pointers.get(variable):
            start = len(self.outer_pointers[variable])
            if stop <= start:
                return self.outer_pointers[variable][stop - 1]
        else:
            self.outer_pointers[variable] = []
            start = offset

        outer_pointers = self.outer_pointers[variable]
        temp = argument_data_pointer
        for_loops = self.function.for_loops[start:stop]

        # Loop over all outer loop levels
        for i, for_node in zip(range(start, stop), for_loops):
            if for_node.dim < offset:
                continue

            # Allocate a temp_data_pointer on each outer loop level
            temp = b.temp(pointer_type)
            dim = for_node.dim - offset

            if not outer_pointers: #i == offset:
                outer_node = self.function
                outer_pointer = self.function.args[variable.name].data_pointer
            else:
                outer_node = self.function.for_loops[i - 1]
                outer_pointer = outer_pointers[-1]

            # Generate: temp_data_pointer = outer_data_pointer
            assmt = b.assign(temp, outer_pointer)
            outer_node.prepending.stats.append(assmt)

            stride = original_stride = self.strides[variable][dim]
            assert stride is not None, ('strides', self.strides[variable],
                                        'dim', dim, 'start', start,
                                        'stop', stop, 'offset', offset,
                                        'specializer', self.sp)

            if for_node.is_controlling_loop:
                # controlling loop for tiled specializations, multiply by the
                # tiling blocksize for this dimension
                stride = b.mul(stride, for_node.blocksize)

            # Generate: temp_data_pointer += stride
            stat = b.assign(temp, b.add(temp, stride))
            if not outer_pointers:
                # Outermost loop level, generate some additional OpenMP
                # parallel-loop-compatible code
                # Generate: temp_data_pointer = data_pointer + i * stride0
                omp_body = b.assign(temp, b.add(outer_pointer,
                                                b.mul(original_stride, for_node.index)))
                for_node.prepending.stats.append(b.omp_if(omp_body))
                for_node.appending.stats.append(b.omp_if(None, stat))
                omp_for = self.treepath_first(self.function, '//OpenMPLoopNode')
                if omp_for is not None:
                    omp_for.privates.append(temp)
            else:
                for_node.appending.stats.append(stat)

            self.outer_pointers[variable].append(temp)

        return temp

    def visit_FunctionNode(self, node):
        self.function = node
        self.indices = self.sp.indices
        node = self.run_optimizations(node)

        self.init_pending_stats(node)
        self.visitchildren(node)
        self.handle_pending_stats(node)

        return node

    def _visit_set_vectorizing_flag(self, node):
        was_vectorizing = self.should_vectorize
        self.should_vectorize = node.should_vectorize
        self.visitchildren(node)
        self.should_vectorize = was_vectorizing
        return node

    def visit_ForNode(self, node):
        is_nd_fornode = node in self.function.for_loops or node.is_fixup
        self.loop_level += is_nd_fornode

        self.init_pending_stats(node)
        self._visit_set_vectorizing_flag(node)
        self.handle_pending_stats(node)

        self.loop_level -= is_nd_fornode
        return node

    def visit_IfNode(self, node):
        self.loop_level += node.is_fixup
        result = self._visit_set_vectorizing_flag(node)
        self.loop_level -= node.is_fixup
        return result

    def visit_AssignmentExpr(self, node):
        # assignment expressions should not be nested
        self.in_lhs_expr = True
        node.lhs = self.visit(node.lhs)
        self.in_lhs_expr = False
        node.rhs = self.visit(node.rhs)

        if node.lhs.type.is_pointer and node.rhs.type.is_vector:
            # This expression must be a statement
            return self.astbuilder.vector_store(node.lhs, node.rhs)

        return node

    def visit_TempNode(self, node):
        self.visitchildren(node)
        return node

    def visit_BinopNode(self, node):
        type = self.get_type(node.type)
        if node.operator == '%' and type.is_float and not self.context.use_llvm:
            # rewrite modulo for floats to fmod()
            b = self.astbuilder
            functype = minitypes.FunctionType(return_type=type,
                                              args=[type, type])
            if type.itemsize == 4:
                modifier = "f"
            elif type.itemsize == 8:
                modifier = ""
            else:
                modifier = "l"

            fmod = b.variable(functype, "fmod%s" % modifier)
            return self.visit(b.funccall(fmod, [node.lhs, node.rhs]))

        self.visitchildren(node)
        return node

    def visit_UnopNode(self, node):
        if node.type.is_vector and node.operator == '-':
            # rewrite unary subtract
            type = node.operand.type
            if type.is_float:
                constant = 0.0
            else:
                constant = 0
            lhs = self.astbuilder.vector_const(type, constant)
            node = self.astbuilder.binop(type, '-', lhs, node.operand)
            return self.visit(node)

        self.visitchildren(node)
        return node

    def visit_DereferenceNode(self, node):
        node.operand = self.visit(node.operand)
        if self.context.llvm:
            node = self.astbuilder.index(node, self.astbuilder.constant(0))
        return node

    def visit_IfElseExprNode(self, node):
        self.visitchildren(node)

        if self.context.use_llvm:
            # Rewrite 'cond ? x : y' expressions to if/else statements
            b = self.astbuilder
            temp = b.temp(node.lhs.type, name='if_temp')
            stat = b.if_else(node.cond, b.assign(temp, node.lhs),
                                        b.assign(temp, node.rhs))

            for_node = self.get_loop(self.loop_level)
            for_node.prepending.stats.append(stat)

            node = temp

        return node

    def visit_PrintNode(self, node):
        b = self.astbuilder

        printf_type = minitypes.FunctionType(
                return_type=minitypes.int_,
                args=[minitypes.CStringType()],
                is_vararg=True)

        printf = b.funcname(printf_type, 'printf')

        args = []
        specifiers = []
        for i, arg in enumerate(node.args):
            specifier, arg = codegen.format_specifier(arg, b)
            args.append(arg)
            specifiers.append(specifier)

        args.insert(0, b.constant(" ".join(specifiers) + "\n"))
        return b.expr_stat(b.funccall(printf, args))

    def visit_PositionInfoNode(self, node):
        """
        Replace with the setting of positional source information in case
        of an error.
        """
        b = self.astbuidler

        posinfo = self.function.posinfo
        if posinfo:
            pos = node.posinfo
            return b.stats(
                b.assign(b.deref(posinfo.filename), b.constant(pos.filename)),
                b.assign(b.deref(posinfo.lineno), b.constant(pos.lineno)),
                b.assign(b.deref(posinfo.column), b.constant(pos.column)))

    def visit_RaiseNode(self, node):
        """
        Generate a call to PyErr_Format() to set an exception.
        """
        from minitypes import FunctionType, object_
        b = self.astbuilder

        args = [object_] * (2 + len(node.fmt_args))
        functype = FunctionType(return_type=object_, args=args)
        return b.expr_stat(
            b.funccall(b.funcname(functype, "PyErr_Format"),
                       [node.exc_var, node.msg_val] + node.fmt_args))

    def visit_ErrorHandler(self, node):
        """
        See miniast.ErrorHandler for an explanation of what this needs to do.
        """
        b = self.astbuilder

        node.error_variable = b.temp(minitypes.bool_)
        node.error_var_init = b.assign(node.error_variable, 0)
        node.cleanup_jump = b.jump(node.cleanup_label)
        node.error_target_label = b.jump_target(node.error_label)
        node.cleanup_target_label = b.jump_target(node.cleanup_label)
        node.error_set = b.assign(node.error_variable, 1)

        if self.error_handlers:
            cascade_code = b.jump(self.error_handlers[-1].error_label)
        else:
            cascade_code = b.return_(self.function.error_value)

        node.cascade = b.if_(node.error_variable, cascade_code)

        self.error_handlers.append(node)
        self.visitchildren(node)
        self.error_handlers.pop()
        return node

    def visit_PragmaForLoopNode(self, node):
        if self.previous_specializer.is_vectorizing_specializer:
            return self.visit(node.for_node)
        else:
            self.visitchildren(node)
            return node

    def visit_StatListNode(self, node):
        self.visitchildren(node)
        return self.fuse_omp_stats(node)

class OrderedSpecializer(Specializer):
    """
    Specializer that understands C and Fortran data layout orders.
    """

    vectorized_equivalents = None

    def compute_total_shape(self, node):
        """
        Compute the product of the shape (entire length of array output).
        Sets the total shape as attribute of the function (total_shape).
        """
        b = self.astbuilder
        # compute the product of the shape and insert it into the function body
        extents = [b.index(node.shape, b.constant(i))
                       for i in range(node.ndim)]
        node.total_shape = b.temp(node.shape.type.base_type)
        init_shape = b.assign(node.total_shape, reduce(b.mul, extents),
                              may_reorder=True)
        node.body = b.stats(init_shape, node.body)
        return node.total_shape

    def loop_order(self, order, ndim=None):
        """
        Returns arguments to (x)range() to process something in C or Fortran
        order.
        """
        if ndim is None:
            ndim = self.function.ndim

        if order == "C":
            return self.c_loop_order(ndim)
        else:
            return self.f_loop_order(ndim)

    def c_loop_order(self, ndim):
        return ndim - 1, -1, -1

    def f_loop_order(self, ndim):
        return 0, ndim, 1

    def order_indices(self, indices):
        """
        Put the indices of the for loops in the right iteration order. The
        loops were build backwards (Fortran order), so for C we need to
        reverse them.

        Note: the indices are always ordered on the dimension they index
        """
        if self.order == "C":
            indices.reverse()

    def ordered_loop(self, node, result_indices, lower=None, upper=None,
                     step=None, loop_order=None):
        """
        Return a ForNode ordered in C or Fortran order.
        """
        b = self.astbuilder

        if lower is None:
            lower = lambda i: None
        if upper is None:
            upper = lambda i: b.shape_index(i, self.function)
        if loop_order is None:
            loop_order = self.loop_order(self.order)

        indices = []
        for_loops = []
        for i in range(*loop_order):
            node = b.for_range_upwards(node, lower=lower(i), upper=upper(i),
                                       step=step)
            node.dim = i
            for_loops.append(node)
            indices.append(node.target)

        self.order_indices(indices)
        result_indices.extend(indices)
        return for_loops[::-1], node

    def _index_pointer(self, pointer, indices, strides):
        """
        Return an element for an N-dimensional index into a strided array.
        """
        b = self.astbuilder
        return b.index_multiple(
            b.cast(pointer, minitypes.char.pointer()),
            [b.mul(index, stride) for index, stride in zip(indices, strides)],
            dest_pointer_type=pointer.type)

    def _strided_element_location(self, node, indices=None, strides_index_offset=0,
                                  ndim=None, pointer=None):
        """
        Like _index_pointer, but given only an array operand indices. It first
        needs to get the data pointer and stride nodes.
        """
        indices = indices or self.indices
        b = self.astbuilder
        if ndim is None:
            ndim = node.type.ndim
        if pointer is None:
            pointer = b.data_pointer(node)

        indices = [index for index in indices[len(indices) - ndim:]]
        strides = [b.stride(node, i + strides_index_offset)
                   for i, idx in enumerate(indices)]
        node = self._index_pointer(pointer, indices, strides)
        self.visitchildren(node)
        return node


def get_any_array_argument(arguments):
    for arg in arguments:
        if arg.type is not None and arg.type.is_array:
            return arg

class CanVectorizeVisitor(minivisitor.TreeVisitor):
    """
    Determines whether we can vectorize a given expression. Currently only
    support arithmetic on floats and doubles.
    """

    can_vectorize = True

    def _valid_type(self, type):
        if type.is_array:
            type = type.dtype
        return type.is_float and type.itemsize in (4, 8)

    def visit_FunctionNode(self, node):
        array_dtypes = [
            arg.type.dtype for arg in node.arguments[1:]
                               if arg.type is not None and arg.type.is_array]

        all_the_same = miniutils.all(
                dtype == array_dtypes[0] for dtype in array_dtypes)
        self.can_vectorize = all_the_same and self._valid_type(array_dtypes[0])

        if self.can_vectorize:
            self.visitchildren(node)

    def visit_BinopNode(self, node):
        if node.lhs.type != node.rhs.type or not self._valid_type(node.lhs.type):
            self.can_vectorize = False
        else:
            self.visitchildren(node)

    def visit_UnopNode(self, node):
        if self._valid_type(node.type):
            self.visitchildren(node)
        else:
            self.can_vectorize = False

    def visit_FuncCallNode(self, node):
        self.can_vectorize = False

    def visit_NodeWrapper(self, node):
        # TODO: dispatch to self.context.can_vectorize
        self.can_vectorize = False

    def visit_Node(self, node):
        self.visitchildren(node)

def visit_if_should_vectorize(func):
    """
    Visits the given method if we are vectorizing, otherwise visit the
    superclass' method of :py:class:`VectorizingSpecialization`
    """
    @functools.wraps(func)
    def wrapper(self, node):
        if self.should_vectorize:
            return func(self, node)
        else:
            method = getattr(super(VectorizingSpecializer, self), func.__name__)
            return method(node)
    return wrapper

class VectorizingSpecializer(Specializer):
    """
    Generate explicitly vectorized code if supported.

    :param vector_size: number of 32-bit operands in the vector
    """

    is_vectorizing_specializer = True

    can_vectorize_visitor = CanVectorizeVisitor
    vectorized_equivalents = None

    # set in subclasses
    vector_size = None

    def __init__(self, context, specialization_name=None):
        super(VectorizingSpecializer, self).__init__(context,
                                                     specialization_name)
        # temporary registers
        self.temps = {}

        # Flag to vectorize expressions in a vectorized loop
        self.should_vectorize = True

    @classmethod
    def can_vectorize(cls, context, ast):
        visitor = cls.can_vectorize_visitor(context)
        visitor.visit(ast)
        # print visitor.can_vectorize, ast.pos
        return visitor.can_vectorize

    @visit_if_should_vectorize
    def visit_FunctionNode(self, node):
        self.dtype = get_any_array_argument(node.arguments).type.dtype
        return super(VectorizingSpecializer, self).visit_FunctionNode(node)

    @visit_if_should_vectorize
    def visit_Variable(self, variable):
        if variable.type.is_array:
            variable = self.astbuilder.vector_variable(variable, self.vector_size)

        return variable

    @visit_if_should_vectorize
    def visit_BinopNode(self, node):
        self.visitchildren(node)
        if node.lhs.type.is_vector:
            # TODO: promotion
            node = self.astbuilder.vector_binop(node.operator,
                                                node.lhs, node.rhs)

        return node

    @visit_if_should_vectorize
    def visit_UnopNode(self, node):
        self.visitchildren(node)
        if node.operand.type.is_vector:
            if node.operator == '+':
                node = node.operand
            else:
                assert node.operator == '~'
                raise NotImplementedError
                node = self.astbuilder.vector_unop(node.type, node.operator,
                                                   self.visit(node.operand))

        return node

    @visit_if_should_vectorize
    def visit_ForNode(self, node):
        node.should_vectorize = True
        self.visitchildren(node)
        return node

    @visit_if_should_vectorize
    def visit_IfNode(self, node):
        node.should_vectorize = True
        self.visitchildren(node)
        return node

    def _modify_inner_loop(self, b, elements_per_vector, node, step):
        """
        Turn 'for (i = 0; i < N; i++)' into 'for (i = 0; i < N - 3; i += 4)'
        for a vector size of 4. In case the data size is not a multiple of
        4, we can only SIMDize that part, and need a fixup loop for any
        remaining elements. Returns the upper limit and the counter (N and i).
        """
        i = node.step.lhs
        N = node.condition.rhs

        # Adjust step
        step = b.mul(step, b.constant(elements_per_vector))
        node.step = b.assign_expr(i, b.add(i, step))

        # Adjust condition
        vsize_minus_one = b.constant(elements_per_vector - 1)
        node.condition.rhs = b.sub(N, vsize_minus_one)

        return N, i

    def fixup_loop(self, i, N, body, elements_per_vector):
        """
        Generate a loop to fix up any remaining elements that didn't fit into
        our SIMD vectors.
        """
        b = self.astbuilder

        cond = b.binop(minitypes.bool_, '<', i, N)
        if elements_per_vector - 1 == 1:
            fixup_loop = b.if_(cond, body)
        else:
            # fixup_loop = b.for_range_upwards(body, lower=i, upper=N)
            init = b.noop_expr()
            step = b.assign_expr(i, b.add(i, b.constant(1)))
            fixup_loop = b.for_(body, init, cond, step, index=i)

        fixup_loop.is_fixup = True

        self.should_vectorize = False
        fixup_loop = self.visit(fixup_loop)
        self.should_vectorize = True

        return fixup_loop

    def process_inner_forloop(self, node, original_expression, step=None):
        """
        Process an inner loop, adjusting the step accordingly and injecting
        any temporary assignments where necessary. Returns the fixup loop,
        needed when the data size is not a multiple of the vector size.

        :param original_expression: original, unmodified, array expression (
                                    the body of the NDIterate node)
        """
        b = self.astbuilder

        if step is None:
            step = b.constant(1)

        elements_per_vector = self.vector_size * 4 / self.dtype.itemsize

        N, i = self._modify_inner_loop(b, elements_per_vector, node, step)
        return self.fixup_loop(i, N, original_expression, elements_per_vector)

class StridedCInnerContigSpecializer(OrderedSpecializer):
    """
    Specialize on the first or last dimension being contiguous (depending
    on the 'order' attribute).
    """

    specialization_name = "inner_contig"
    order = "C"

    is_inner_contig_specializer = True
    vectorized_equivalents = None

    def __init__(self, context, specialization_name=None):
        super(StridedCInnerContigSpecializer, self).__init__(
                                    context, specialization_name)
        self.indices = []

    def _generate_inner_loop(self, b, node):
        """
        Generate innermost loop, injecting the pointer assignments in the
        right place
        """
        loop = node
        if len(self.indices) > 1:
            for index in self.indices[:-2]:
                loop = node.body

            self.inner_loop = loop.body
            loop.body = b.pragma_for(self.inner_loop)
            node = self.omp_for(node)
        else:
            self.inner_loop = loop
            node = self.omp_for(b.pragma_for(self.inner_loop))

        return loop, node

    def _vectorize_inner_loop(self, b, loop, node, original_expr):
        "Vectorize the inner loop and insert the fixup loop"
        if self.is_vectorizing_specializer:
            fixup_loop = self.process_inner_forloop(self.inner_loop,
                                                    original_expr)
            if len(self.indices) > 1:
                loop.body = b.stats(loop.body, fixup_loop)
            else:
                node = b.stats(node, fixup_loop)

        return node

    def visit_NDIterate(self, node):
        """
        Replace this node with ordered loops and a direct index into a
        temporary data pointer in the contiguous dimension.
        """
        b = self.astbuilder

        assert not list(self.treepath(node, '//NDIterate'))

        original_expr = specialize_ast(node.body)

        # start by generating a C or Fortran ordered loop
        self.function.for_loops, node = self.ordered_loop(node.body,
                                                          self.indices)
        loop, node = self._generate_inner_loop(b, node)
        result = self.visit(node)
        node = self._vectorize_inner_loop(b, loop, node, original_expr)

        return result

    def index(self, loop_level):
        if self.order == 'C':
            return self.indices[loop_level]
        else:
            return self.indices[-loop_level]

    def strided_indices(self):
        "Return the list of strided indices for this order"
        return self.indices[:-1]

    def contig_index(self):
        "The contiguous index"
        return self.indices[-1]

    def get_data_pointer(self, variable, loop_level):
        return self.compute_inner_dim_pointer(variable, loop_level)


class StridedFortranInnerContigSpecializer(StridedCInnerContigSpecializer):
    """
    Specialize on the first dimension being contiguous.
    """

    order = "F"
    specialization_name = "inner_contig_fortran"

    vectorized_equivalents = None

    def strided_indices(self):
        return self.indices[1:]

    def contig_index(self):
        return self.indices[0]


class StrengthReducingStridedSpecializer(StridedCInnerContigSpecializer):
    """
    Specialize on strided operands. If some operands are contiguous in the
    dimension compatible with the order we are specializing for (the first
    if Fortran, the last if C), then perform a direct index into a temporary
    date pointer. For strided operands, perform strength reduction in the
    inner dimension by adding the stride to the data pointer in each iteration.
    """

    specialization_name = "strided"
    order = "C"

    is_strided_specializer = True
    vectorized_equivalents = None

    def matching_contiguity(self, type):
        """
        Check whether the array operand for the given type can be directly
        indexed.
        """
        return ((type.is_c_contig and self.order == "C") or
                (type.is_f_contig and self.order == "F"))

    def visit_NDIterate(self, node):
        b = self.astbuilder
        outer_loop = super(StridedSpecializer, self).visit_NDIterate(node)
        # outer_loop = self.strength_reduce_inner_dimension(outer_loop,
        #                                                   self.inner_loop)
        return outer_loop

    def strength_reduce_inner_dimension(self, outer_loop, inner_loop):
        """
        Reduce the strength of strided array operands in the inner dimension,
        by adding the stride to the temporary pointer.
        """
        b = self.astbuilder

        outer_stats = []
        stats = []
        for arg in self.function.arguments:
            type = arg.variable.type
            if type is None:
                continue

            contig = self.matching_contiguity(type)
            if arg.variable in self.pointers and not contig:
                p = self.pointers[arg.variable]

                if self.order == "C":
                    inner_dim = type.ndim - 1
                else:
                    inner_dim = 0

                # Implement: temp_stride = strides[inner_dim] / sizeof(dtype)
                stride = b.stride(arg.variable, inner_dim)
                temp_stride = b.temp(stride.type.qualify("const"),
                                     name="temp_stride")
                outer_stats.append(
                    b.assign(temp_stride, b.div(stride, b.sizeof(type.dtype))))

                # Implement: temp_pointer += temp_stride
                stats.append(b.assign(p, b.add(p, temp_stride)))

        inner_loop.body = b.stats(inner_loop.body, *stats)
        outer_stats.append(outer_loop)
        return b.stats(*outer_stats)

class StrengthReducingStridedFortranSpecializer(
    StridedFortranInnerContigSpecializer, StrengthReducingStridedSpecializer):
    """
    Specialize on Fortran order for strided operands and apply strength
    reduction in the inner dimension.
    """

    specialization_name = "strided_fortran"
    order = "F"

    vectorized_equivalents = None

class StridedSpecializer(StridedCInnerContigSpecializer):
    """
    Specialize on strided operands. If some operands are contiguous in the
    dimension compatible with the order we are specializing for (the first
    if Fortran, the last if C), then perform a direct index into a temporary
    date pointer.
    """

    specialization_name = "strided"
    order = "C"

    vectorized_equivalents = None
    is_strided_specializer = True

    def matching_contiguity(self, type):
        """
        Check whether the array operand for the given type can be directly
        indexed.
        """
        return ((type.is_c_contig and self.order == "C") or
                (type.is_f_contig and self.order == "F"))

    def _element_location(self, variable, loop_level):
        """
        Generate a strided or directly indexed load of a single element.
        """
        #if variable in self.pointers:
        if self.matching_contiguity(variable.type):
            return super(StridedSpecializer, self)._element_location(variable,
                                                                     loop_level)

        b = self.astbuilder
        pointer = self.get_data_pointer(variable, loop_level)
        indices = [self.contig_index()]

        if self.order == "C":
            inner_dim = variable.type.ndim - 1
        else:
            inner_dim = 0

        strides = [b.stride(variable, inner_dim)]
        return self._index_pointer(pointer, indices, strides)


class StridedFortranSpecializer(StridedFortranInnerContigSpecializer,
                                StridedSpecializer):
    """
    Specialize on Fortran order for strided operands.
    """

    specialization_name = "strided_fortran"
    order = "F"

    vectorized_equivalents = None

if strength_reduction:
    StridedSpecializer = StrengthReducingStridedSpecializer
    StridedFortranSpecializer = StrengthReducingStridedFortranSpecializer

class ContigSpecializer(OrderedSpecializer):
    """
    Specialize on all specializations being contiguous (all F or all C).
    """

    specialization_name = "contig"
    is_contig_specializer = True

    def visit_FunctionNode(self, node):
        node = super(ContigSpecializer, self).visit_FunctionNode(node)
        self.astbuilder.create_function_type(node, strides_args=False)
        return node

    def visit_NDIterate(self, node):
        """
        Generate a single ForNode over the total data size.
        """
        b = self.astbuilder
        original_expr = specialize_ast(node.body)
        node = super(ContigSpecializer, self).visit_NDIterate(node)

        for_node = b.for_range_upwards(node.body,
                                       upper=self.function.total_shape)
        self.function.for_loops = [for_node]
        self.indices = [for_node.index]

        node = self.omp_for(b.pragma_for(for_node))
        self.target = for_node.target
        node = self.visit(node)

        if self.is_vectorizing_specializer:
            fixup_loop = self.process_inner_forloop(for_node, original_expr)
            node = b.stats(node, fixup_loop)

        return node

    def visit_StridePointer(self, node):
        return None

    def _element_location(self, node, loop_level):
        "Directly index the data pointer"
        data_pointer = self.astbuilder.data_pointer(node)
        return self.astbuilder.index(data_pointer, self.target)

    def index(self, loop_level):
        return self.target

    def contig_index(self):
        return self.target


class CTiledStridedSpecializer(StridedSpecializer):
    """
    Generate tiled code for the last two (C) or first two (F) dimensions.
    The blocksize may be overridden through the get_blocksize method, in
    a specializer subclass or mixin (see miniast.Context.specializer_mixin_cls).
    """
    specialization_name = "tiled"
    order = "C"
    is_tiled_specializer = True

    vectorized_equivalents = None

    def get_blocksize(self):
        """
        Get the tile size. Override in subclasses to provide e.g. parametric
        tiling.
        """
        return self.astbuilder.constant(64)

    def tiled_order(self):
        "Tile in the last two dimensions"
        return self.function.ndim - 1, self.function.ndim - 1 - 2, -1

    def untiled_order(self):
        return self.function.ndim - 1 - 2, -1, -1

    def visit_NDIterate(self, node):
        assert self.function.ndim >= 2
        return self._tile_in_two_dimensions(node)

    def _tile_in_two_dimensions(self, node):
        """
        This version generates tiling loops in the first or last two dimensions
        (depending on C or Fortran order).
        """
        b = self.astbuilder

        self.tiled_indices = []
        self.indices = []
        self.blocksize = self.get_blocksize()

        # Generate the two outer tiling loops
        tiled_loop_body = b.stats(b.constant(0)) # fake empty loop body
        controlling_loops, body = self.ordered_loop(
                tiled_loop_body, self.tiled_indices, step=self.blocksize,
                loop_order=self.tiled_order())
        del tiled_loop_body.stats[:]

        # Generate some temporaries to store the upper limit of the inner
        # tiled loops
        upper_limits = {}
        stats = []
        # sort the indices in forward order, to match up with the ordered
        # indices
        tiled_order = sorted(range(*self.tiled_order()))
        for i, index in zip(tiled_order, self.tiled_indices):
            upper_limit = b.temp(index.type)
            tiled_loop_body.stats.append(
                b.assign(upper_limit, b.min(b.add(index, self.blocksize),
                                            b.shape_index(i, self.function))))
            upper_limits[i] = upper_limit

        tiled_indices = dict(zip(tiled_order, self.tiled_indices))
        def lower(i):
            if i in tiled_indices:
                return tiled_indices[i]
            return None

        def upper(i):
            if i in upper_limits:
                return upper_limits[i]
            return b.shape_index(i, self.function)

        # Generate the inner tiled loops
        outer_for_node = node.body
        inner_body = node.body

        tiling_loops, inner_loops = self.ordered_loop(
            node.body, self.indices,
            lower=lower, upper=upper,
            loop_order=self.tiled_order())

        tiled_loop_body.stats.append(inner_loops)
        innermost_loop = inner_loops.body

        # Generate the outer loops (in case the array operands have more than
        # two dimensions)
        indices = []
        outer_loops, body = self.ordered_loop(body, indices,
                                              loop_order=self.untiled_order())

        body = self.omp_for(body)
        # At this point, 'self.indices' are the indices of the tiled loop
        # (the indices in the first two dimensions for Fortran,
        #  the indices in the last two # dimensions for C)
        # 'indices' are the indices of the outer loops
        if self.order == "C":
            self.indices = indices + self.indices
        else:
            self.indices = self.indices + indices

        # if strength_reduction:
        #     body = self.strength_reduce_inner_dimension(body, innermost_loop)

        for dim, for_node in enumerate(controlling_loops):
            for_node.is_controlling_loop = True
            for_node.blocksize = self.blocksize

        for dim, for_node in enumerate(tiling_loops):
            for_node.is_tiling_loop = True

        self.set_dims(controlling_loops)
        self.set_dims(tiling_loops)

        self.function.controlling_loops = controlling_loops
        self.function.tiling_loops = tiling_loops
        self.function.outer_loops = outer_loops
        self.function.for_loops = outer_loops + controlling_loops + tiling_loops

        self.function.lower_tiling_limits = tiled_indices
        self.function.upper_tiling_limits = upper_limits

        return self.visit(body)

    def set_dims(self, tiled_loops):
        "Set the 'dim' attributes of the tiling and controlling loops"
        # We need to reverse our tiled order, since this order is used to
        # build up the for nodes in reverse. We have an ordered list of for
        # nodes.
        tiled_order = reversed(range(*self.tiled_order()))
        for dim, for_node in zip(tiled_order, tiled_loops):
            for_node.dim = dim

    def _tile_in_all_dimensions(self, node):
        """
        This version generates tiling loops in all dimensions.
        """
        b = self.astbuilder

        self.tiled_indices = []
        self.indices = []
        self.blocksize = self.get_blocksize()

        tiled_loop_body = b.stats(b.constant(0)) # fake empty loop body
        controlling_loops, body = self.ordered_loop(tiled_loop_body,
                                                     self.tiled_indices,
                                                     step=self.blocksize)
        body = self.omp_for(body)
        del tiled_loop_body.stats[:]

        upper_limits = []
        stats = []
        for i, index in enumerate(self.tiled_indices):
            upper_limit = b.temp(index.type)
            tiled_loop_body.stats.append(
                b.assign(upper_limit, b.min(b.add(index, self.blocksize),
                                            b.shape_index(i, self.function))))
            upper_limits.append(upper_limit)

        tiling_loops, inner_body = self.ordered_loop(
            node.body, self.indices,
            lower=lambda i: self.tiled_indices[i],
            upper=lambda i: upper_limits[i])
        tiled_loop_body.stats.append(inner_body)

        self.function.controlling_loops = controlling_loops
        self.function.tiling_loops = tiling_loops
        self.function.outer_loops = []
        self.function.for_loops = tiling_loops

        return self.visit(body)

    def strided_indices(self):
        return self.indices[:-1] + [self.tiled_indices[1]]

    def _element_location(self, variable, loop_level):
        """
        Return data + i * strides[0] + j * strides[1] when we are not using
        strength reduction. Otherwise generate temp_data += strides[1]. For
        this to work, temp_data must be set to
        data + i * strides[0] + outer_j * strides[1]. This happens through
        _compute_inner_dim_pointers with tiled=True.
        """
        if strength_reduction:
            return super(CTiledStridedSpecializer, self)._element_location(
                                                        variable, loop_level)
        else:
            return self._strided_element_location(variable)

    def get_data_pointer(self, variable, loop_level):
        return self.compute_inner_dim_pointer(variable, loop_level, tiled=True)

class FTiledStridedSpecializer(StridedFortranSpecializer,
                               #StrengthReducingStridedFortranSpecializer,
                               CTiledStridedSpecializer):
    "Tile in Fortran order"

    specialization_name = "tiled_fortran"
    order = "F"

    def tiled_order(self):
        "Tile in the first two dimensions"
        return 0, 2, 1

    def untiled_order(self):
        return 2, self.function.ndim, 1

    def strided_indices(self):
        return [self.tiled_indices[0]] + self.indices[1:]

#
### Vectorized specializer equivalents
#
def create_vectorized_specializers(specializer_cls):
    """
    Creates Vectorizing specializer classes from the given specializer for
    SSE and AVX.
    """
    bases = (VectorizingSpecializer, specializer_cls)
    d = dict(vectorized_equivalents=None)
    name = 'Vectorized%%d%s' % specializer_cls.__name__
    cls1 = type(name % 4, bases, dict(d, vector_size=4))
    cls2 = type(name % 8, bases, dict(d, vector_size=8))
    return cls1, cls2

ContigSpecializer.vectorized_equivalents = (
                create_vectorized_specializers(ContigSpecializer))
StridedCInnerContigSpecializer.vectorized_equivalents = (
                create_vectorized_specializers(StridedCInnerContigSpecializer))
StridedFortranInnerContigSpecializer.vectorized_equivalents = (
                create_vectorized_specializers(StridedFortranInnerContigSpecializer))

#
### Create cict of all specializers
#
_specializer_list = [
    ContigSpecializer,
    StridedCInnerContigSpecializer, StridedFortranInnerContigSpecializer,
    StridedSpecializer, StridedFortranSpecializer,
    CTiledStridedSpecializer, FTiledStridedSpecializer,
]

specializers = {}

for sp in _specializer_list:
    specializers[sp.specialization_name] = sp
    vectorizers = getattr(sp, 'vectorized_equivalents', None)
    if vectorizers:
        specializers[sp.specialization_name + '_sse'] = vectorizers[0]
        specializers[sp.specialization_name + '_avx'] = vectorizers[1]
