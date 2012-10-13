"""
Code generator module. Subclass CodeGen to implement a code generator
as a visitor.
"""

import sys
import string

import minierror
import minitypes
import minivisitor

class CodeGen(minivisitor.TreeVisitor):
    """
    Base class for code generators written as visitors.
    """

    def __init__(self, context, codewriter):
        super(CodeGen, self).__init__(context)
        self.code = codewriter

    def clone(self, context, codewriter):
        cls = type(self)
        kwds = dict(self.__dict__)
        kwds.update(context=context, codewriter=codewriter)
        result = cls(context, codewriter)
        vars(result).update(kwds)
        return result

    def results(self, *nodes):
        results = []
        for childlist in nodes:
            result = self.visit_childlist(childlist)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

        return tuple(results)

    def visitchild(self, node):
        if node is None:
            return
        return self.visit(node)

class CodeGenCleanup(CodeGen):
    """
    Perform cleanup for all nodes. This is invoked from an appropriate clean-
    up point from an :py:class:`minivect.miniast.ErrorHandler`. Recursion
    should hence stop at ErrorHandler nodes, since an ErrorHandler descendant
    should handle its own descendants.

    Users of minivect should subclass this to DECREF object temporaries, etc.
    """

    def visit_Node(self, node):
        self.visitchildren(node)

    def visit_ErrorHandler(self, node):
        # stop recursion here
        pass

def format_specifier(node, astbuilder):
    "Return a printf() format specifier for the type of the given AST node"
    type = node.type

    format = None
    dst_type = None

    if type.is_pointer:
        format = "%p"
    elif type.is_numeric:
        if type.is_int_like:
            format = "%i"
            dst_type = minitypes.int_
        elif type.is_float:
            format = "%f"
        elif type.is_double:
            format = "%lf"
    elif type.is_c_string:
        format = "%s"

    if format is not None:
        if dst_type:
            node = astbuilder.cast(node, dst_type)
        return format, node
    else:
        raise minierror.UnmappableFormatSpecifierError(type)

class CCodeGen(CodeGen):
    """
    Generates C code from an AST, needs a
    :py:class:`minivect.minicode.CCodeWriter`. To use the vectorized
    specializations, use the :py:class:`VectorCodeGen` below.
    """

    label_counter = 0
    disposal_point = None

    def __init__(self, context, codewriter):
        super(CCodeGen, self).__init__(context, codewriter)
        self.declared_temps = set()
        self.temp_names = set()

    def strip(self, expr_string):
        # strip parentheses from C string expressions where unneeded
        if expr_string and expr_string[0] == '(' and expr_string[-1] == ')':
            return expr_string[1:-1]
        return expr_string

    def visit_FunctionNode(self, node):
        code = self.code

        self.specializer = node.specializer
        self.function = node

        name = code.mangle(node.mangled_name + node.specialization_name)
        node.mangled_name = name

        args = self.results(node.arguments + node.scalar_arguments)
        args = (arg for arg in args if arg is not None)
        proto = "static int %s(%s)" % (name, ", ".join(args))
        code.proto_code.putln(proto + ';')
        code.putln("%s {" % proto)
        code.declaration_levels.append(code.insertion_point())
        code.function_declarations = code.insertion_point()
        code.before_loop = code.insertion_point()
        self.visitchildren(node)
        code.declaration_levels.pop()
        code.putln("}")

    def _argument_variables(self, variables):
        return ", ".join("%s %s" % (v.type, self.visit(v))
                             for v in variables if v is not None)

    def visit_FunctionArgument(self, node):
        if node.used:
            return self._argument_variables(node.variables)

    def visit_ArrayFunctionArgument(self, node):
        if node.used:
            return self._argument_variables([node.data_pointer,
                                             node.strides_pointer])

    def visit_StatListNode(self, node):
        self.visitchildren(node)
        return node

    def visit_ExprStatNode(self, node):
        node.expr.is_statement = True
        result = self.visit(node.expr)
        if result:
            self.code.putln(self.strip(result) + ';')

    def visit_ExprNodeWithStatement(self, node):
        self.visit(node.stat)
        return self.visit(node.expr)

    def visit_OpenMPLoopNode(self, node):
        self.code.putln("#ifdef _OPENMP")

        if_clause = self.visit(node.if_clause)
        lastprivates = ", ".join(self.results(node.lastprivates))
        privates = ""
        if node.privates:
            privates = " private(%s)" % ", ".join(self.results(node.privates))

        pragma = "#pragma omp parallel for if(%s) lastprivate(%s)%s"
        self.code.putln(pragma % (if_clause, lastprivates, privates))
        self.code.putln("#endif")
        self.visit(node.for_node)

    def visit_OpenMPConditionalNode(self, node):
        if node.if_body:
            self.code.putln("#ifdef _OPENMP")
            self.visit(node.if_body)

        if node.else_body:
            if not node.if_body:
                self.code.putln("#ifndef _OPENMP")
            else:
                self.code.putln("#else")

            self.visit(node.else_body)
        self.code.putln("#endif")

    def put_intel_pragmas(self, code):
        """
        Insert Intel compiler specific pragmas. See "A Guide to Vectorization
        with Intel(R) C++ Compilers".
        """
        code.putln("#ifdef __INTEL_COMPILER")
        # force auto-vectorization
        code.putln("#pragma simd")
        # ignore potential data dependencies
        # code.putln("#pragma ivdep")
        # vectorize even if the compiler doesn't think this will be beneficial
        # code.putln("#pragma vector always")
        code.putln("#endif")

    def visit_PragmaForLoopNode(self, node):
        self.put_intel_pragmas(self.code)
        self.visit(node.for_node)

    def visit_ForNode(self, node):
        code = self.code

        exprs = self.results(node.init, node.condition, node.step)
        code.putln("for (%s; %s; %s) {" % tuple(self.strip(e) for e in exprs))

        self.code.declaration_levels.append(code.insertion_point())
        self.code.loop_levels.append(code.insertion_point())

        self.visit(node.init)
        self.visit(node.body)

        self.code.declaration_levels.pop()
        self.code.loop_levels.pop()

        code.putln("}")

    def visit_IfNode(self, node):
        self.code.putln("if (%s) {" % self.results(node.cond))
        self.visit(node.body)
        if node.else_body:
            self.code.putln("} else {")
            self.visit(node.else_body)
        self.code.putln("}")

    def visit_PromotionNode(self, node):
        # Use C rules for promotion
        return self.visit(node.operand)

    def visit_FuncCallNode(self, node):
        return "%s(%s)" % (self.visit(node.func_or_pointer),
                           ", ".join(self.results(node.args)))

    def visit_FuncNameNode(self, node):
        return node.name

    def visit_FuncRefNode(self, node):
        return node.function.mangled_name

    def visit_ReturnNode(self, node):
        self.code.putln("return %s;" % self.results(node.operand))

    def visit_BinopNode(self, node):
        op = node.operator
        return "(%s %s %s)" % (self.visit(node.lhs),
                               op,
                               self.visit(node.rhs))

    def visit_UnopNode(self, node):
        return "(%s%s)" % (node.operator, self.visit(node.operand))

    def _mangle_temp(self, node):
        name = self.code.mangle(node.repr_name or node.name)
        if name in self.temp_names:
            name = "%s_%d" % (name.rstrip(string.digits),
                              len(self.declared_temps))
        node.name = name
        self.temp_names.add(name)
        self.declared_temps.add(node)

    def _declare_temp(self, node, rhs_result=None):
        if node not in self.declared_temps:
            self._mangle_temp(node)
        code = self.code.declaration_levels[-1]
        if rhs_result:
            assignment = " = %s" % (rhs_result,)
        else:
            assignment = ""

        code.putln("%s %s%s;" % (node.type, node.name, assignment))

    def visit_TempNode(self, node):
        if node not in self.declared_temps:
            self._declare_temp(node)

        return node.name

    def visit_AssignmentExpr(self, node):
        if (node.rhs.is_binop and node.rhs.operator == '+' and
                node.rhs.rhs.is_constant and node.rhs.rhs.value == 1):
            return "%s++" % self.visit(node.rhs.lhs)
        elif node.rhs.is_binop and node.lhs == node.rhs.lhs:
            return "(%s %s= %s)" % (self.visit(node.lhs),
                                    node.rhs.operator,
                                    self.visit(node.rhs.rhs))
        elif (node.is_statement and node.lhs.is_temp and
                  node.lhs not in self.declared_temps and node.may_reorder):
            self._mangle_temp(node.lhs)
            self._declare_temp(node.lhs, self.visit(node.rhs))
        else:
            return "(%s = %s)" % self.results(node.lhs, node.rhs)

    def visit_IfElseExprNode(self, node):
        return "(%s ? %s : %s)" % (self.results(node.cond, node.lhs, node.rhs))

    def visit_CastNode(self, node):
        return "((%s) %s)" % (node.type, self.visit(node.operand))

    def visit_DereferenceNode(self, node):
        return "(*%s)" % self.visit(node.operand)

    def visit_SingleIndexNode(self, node):
        return "%s[%s]" % self.results(node.lhs, node.rhs)

    def visit_SizeofNode(self, node):
        return "sizeof(%s)" % node.sizeof_type

    def visit_ArrayAttribute(self, node):
        return node.name

    def visit_Variable(self, node):
        if node.type.is_function:
            return node.name

        if not node.mangled_name:
            node.mangled_name = self.code.mangle(node.name)
        return node.mangled_name

    def visit_NoopExpr(self, node):
        return ""

    def visit_ResolvedVariable(self, node):
        return self.visit(node.element)

    def visit_JumpNode(self, node):
        self.code.putln("goto %s;" % self.results(node.label))

    def visit_JumpTargetNode(self, node):
        self.code.putln("%s:" % self.results(node.label))

    def visit_LabelNode(self, node):
        if node.mangled_name is None:
            node.mangled_name = self.code.mangle("%s%d" % (node.name,
                                                           self.label_counter))
            self.label_counter += 1
        return node.mangled_name

    def visit_ConstantNode(self, node):
        if node.type.is_c_string:
            return '"%s"' % node.value.encode('string-escape')
        return str(node.value)

    def visit_ErrorHandler(self, node):
        # initialize the mangled names before generating code for the body
        self.visit(node.error_label)
        self.visit(node.cleanup_label)

        self.visit(node.error_var_init)
        self.visit(node.body)
        self.visit(node.cleanup_jump)
        self.visit(node.error_target_label)
        self.visit(node.error_set)
        self.visit(node.cleanup_target_label)

        disposal_codewriter = self.code.insertion_point()
        self.context.generate_disposal_code(disposal_codewriter, node.body)
        #have_disposal_code = disposal_codewriter.getvalue()

        self.visit(node.cascade)
        return node


class VectorCodegen(CCodeGen):
    """
    Generate C code for vectorized ASTs. As a subclass of :py:class:`CCodeGen`,
    can write C code for any minivect AST.
    """

    types = {
        minitypes.VectorType(minitypes.float_, 4) : '_mm_%s_ps',
        minitypes.VectorType(minitypes.float_, 8) : '_mm256_%s_ps',
        minitypes.VectorType(minitypes.double, 4) : '_mm_%s_pd',
        minitypes.VectorType(minitypes.double, 8) : '_mm256_%s_pd',
    }

    binops = {
        '+': 'add',
        '*': 'mul',
        '-': 'sub',
        '/': 'div',
        # pow not supported
        # floordiv not supported
        # mod not supported

        '<': 'cmplt',
        '<=': 'cmple',
        '==': 'cmpeq',
        '!=': 'cmpne',
        '>=': 'cmpge',
        '>': 'cmpgt',
    }

    def visit_VectorVariable(self, node):
        return self.visit(node.variable)

    def visit_VectorLoadNode(self, node):
        load = self.types[node.type] % 'loadu'
        return '%s(%s)' % (load, self.visit(node.operand))

    def visit_VectorStoreNode(self, node):
        # Assignment to data pointer
        store = self.types[node.rhs.type] % 'storeu'
        return '%s(%s, %s)' % (store, self.visit(node.lhs),
                               self.visit(node.rhs))

    def visit_VectorBinopNode(self, node):
        binop_name = self.binops[node.operator]
        func_name =  self.types[node.lhs.type] % binop_name
        return '%s(%s, %s)' % (func_name, self.visit(node.lhs),
                                          self.visit(node.rhs))

    def visit_ConstantVectorNode(self, node):
        func_template = self.types[node.type]
        if node.constant == 0:
            return func_template % 'setzero'
        else:
            func = func_template % 'set'
            c = node.constant
            return '%s(%s, %s, %s, %s)' % (func, c, c, c, c)