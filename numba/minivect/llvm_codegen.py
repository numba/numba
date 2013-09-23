# -*- coding: utf-8 -*-
"""
Generate LLVM code for a minivect AST.
"""
from __future__ import print_function, division, absolute_import

import sys

try:
    import llvm.core
    import llvm.ee
    import llvm.passes
except ImportError:
    llvm = None

from . import codegen
from . import minierror
from . import minitypes
from . import minivisitor
from . import ctypes_conversion

def handle_struct_passing(builder, alloca_func, largs, signature):
    """
    Handle signatures with structs. If signature.struct_by_reference
    is set, we need to pass in structs by reference, and retrieve
    struct return values by refererence through an additional argument.

    Structs are always loaded as pointers.
    Complex numbers are always immediate struct values.
    """
    for i, (arg_type, larg) in enumerate(zip(signature.args, largs)):
        if minitypes.pass_by_ref(arg_type):
            if signature.struct_by_reference:
                if arg_type.is_complex or \
                        arg_type.is_datetime or \
                        arg_type.is_timedelta:
                    new_arg = alloca_func(arg_type)
                    builder.store(larg, new_arg)
                    larg = new_arg

            largs[i] = larg

    if (signature.struct_by_reference and
            minitypes.pass_by_ref(signature.return_type)):
        return_value = alloca_func(signature.return_type)
        largs.append(return_value)
        return return_value

    return None

class LLVMCodeGen(codegen.CodeGen):
    """
    Generate LLVM code for a minivect AST.

    Takes a regular :py:class:`minivect.minicode.CodeWriter` to which it
    writes the LLVM function and a ctypes function.
    """

    in_lhs_expr = 0

    def __init__(self, context, codewriter):
        super(LLVMCodeGen, self).__init__(context, codewriter)
        self.declared_temps = set()
        self.temp_names = set()

        self.astbuilder = context.astbuilder
        self.blocks = []
        self.symtab = {}
        self.llvm_temps = {}

        import llvm # raise an error at this point if llvm-py is not installed

        self.init_binops()
        self.init_comparisons()

        # List of LLVM call instructions to inline
        self.inline_calls = []

    def append_basic_block(self, name='unamed'):
        "append a basic block and keep track of it"
        idx = len(self.blocks)
        bb = self.lfunc.append_basic_block('%s_%d' % (name, idx))
        self.blocks.append(bb)
        return bb

    def inline_funcs(self):
        for call_instr in self.inline_calls:
            # print 'inlining...', call_instr
            llvm.core.inline_function(call_instr)

    def optimize(self):
        "Run llvm optimizations on the generated LLVM code"
        llvm_fpm = llvm.passes.FunctionPassManager.new(self.llvm_module)
        # target_data = llvm.ee.TargetData(self.context.llvm_ee)
        #llvm_fpm.add(self.context.llvm_ee.target_data.clone())
        pmb = llvm.passes.PassManagerBuilder.new()
        pmb.opt_level = 3
        pmb.vectorize = True

        pmb.populate(llvm_fpm)
        llvm_fpm.run(self.lfunc)

    def optimize2(self, opt=3, cg=3, inline=1000):
        features = '-avx'
        tm = self.__machine = llvm.ee.TargetMachine.new(
            opt=cg, cm=llvm.ee.CM_JITDEFAULT, features=features)
        has_loop_vectorizer = llvm.version >= (3, 2)
        passmanagers = llvm.passes.build_pass_managers(
            tm, opt=opt, inline_threshold=inline,
            loop_vectorize=has_loop_vectorizer, fpm=False)
        passmanagers.pm.run(self.llvm_module)

    def visit_FunctionNode(self, node):
        self.specializer = node.specializer
        self.function = node
        self.llvm_module = self.context.llvm_module

        name = node.name + node.specialization_name
        node.mangled_name = name

        lfunc_type = node.type.to_llvm(self.context)
        self.lfunc = self.llvm_module.add_function(lfunc_type, node.mangled_name)
        # self.lfunc.linkage = llvm.core.LINKAGE_LINKONCE_ODR

        self.entry_bb = self.append_basic_block('entry')
        self.builder = llvm.core.Builder.new(self.entry_bb)

        self.add_arguments(node)
        self.visit(node.body)

        # print self.lfunc
        self.llvm_module.verify()
        self.inline_funcs()
        if self.context.optimize_llvm:
            self.optimize2()

        self.code.write(self.lfunc)

        # from numba.codegen.llvmcontext import LLVMContextManager
        # ctypes_func = ctypes_conversion.get_ctypes_func(
        #             node, self.lfunc, LLVMContextManager().execution_engine,
        #                                                 self.context)
        # self.code.write(ctypes_func)

    def add_arguments(self, function):
        "Insert function arguments into the symtab"
        i = 0
        for arg in function.arguments + function.scalar_arguments:
            if arg.used:
                for var in arg.variables:
                    llvm_arg = self.lfunc.args[i]
                    self.symtab[var.name] = llvm_arg
                    if var.type.is_pointer:
                        llvm_arg.add_attribute(llvm.core.ATTR_NO_ALIAS)
                        llvm_arg.add_attribute(llvm.core.ATTR_NO_CAPTURE)
                    i += 1
            else:
                for var in arg.variables:
                    self.symtab[var.name] = self.visit(var)

    def visit_PrintNode(self, node):
        pass

    def visit_Node(self, node):
        self.visitchildren(node)
        return node

    def visit_OpenMPConditionalNode(self, node):
        "OpenMP is not yet implemented, only process the 'else' directives."
        if node.else_body:
            self.visit(node.else_body)
        return node

    def visit_ForNode(self, node):
        '''
        Implements simple for loops with iternode as range, xrange
        '''
        bb_cond = self.append_basic_block('for.cond')
        bb_incr = self.append_basic_block('for.incr')
        bb_body = self.append_basic_block('for.body')
        bb_exit = self.append_basic_block('for.exit')

        # generate initializer
        self.visit(node.init)
        self.builder.branch(bb_cond)

        # generate condition
        self.builder.position_at_end(bb_cond)
        cond = self.visit(node.condition)
        self.builder.cbranch(cond, bb_body, bb_exit)

        # generate increment
        self.builder.position_at_end(bb_incr)
        self.visit(node.step)
        self.builder.branch(bb_cond)

        # generate body
        self.builder.position_at_end(bb_body)
        self.visit(node.body)
        self.builder.branch(bb_incr)

        # move to exit block
        self.builder.position_at_end(bb_exit)

    def visit_IfNode(self, node):
        cond = self.visit(node.cond)

        bb_true = self.append_basic_block('if.true')
        bb_endif = self.append_basic_block('if.end')

        if node.else_body:
            bb_false = self.append_basic_block('if.false')
        else:
            bb_false = bb_endif

        test = self.visit(node.cond)
        self.builder.cbranch(test, bb_true, bb_false)

        # if cond
        self.builder.position_at_end(bb_true)
        self.visit(node.body)
        self.builder.branch(bb_endif)

        if node.else_body:
            # else
            self.builder.position_at_end(bb_false)
            self.visit(node.else_body)
            self.builder.branch(bb_endif)

        # endif
        self.builder.position_at_end(bb_endif)

    def visit_ReturnNode(self, node):
        self.builder.ret(self.visit(node.operand))

    def visit_CastNode(self, node):
        if node.type.is_pointer:
            result = self.visit(node.operand)
            dest_type = node.type.to_llvm(self.context)
            # print result, dest_type
            # node.print_tree(self.context)
            return self.builder.bitcast(result, dest_type)
            # return result.bitcast(node.type)

        return self.visit_PromotionNode(node)

    def visit_PromotionNode(self, node):
        """
        Handle promotions as inserted by
        :py:class:`minivect.type_promoter.TypePromoter`
        """
        result = self.visit(node.operand)
        type = node.type
        op_type = node.operand.type

        smaller = type.itemsize < op_type.itemsize
        if type.is_int and op_type.is_int:
            op = (('zext', 'sext'), ('trunc', 'trunc'))[smaller][type.signed]
        elif type.is_float and op_type.is_float:
            op =  ('fpext', 'fptrunc')[smaller]
        elif type.is_int and op_type.is_float:
            op = ('fptoui', 'fptosi')[type.signed]
        elif type.is_float and op_type.is_int:
            op = ('fptoui', 'fptosi')[type.signed]
        else:
            raise NotImplementedError((type, op_type))

        ltype = type.to_llvm(self.context)
        return getattr(self.builder, op)(result, ltype)

    def init_binops(self):
        # (float_op, unsigned_int_op, signed_int_op)
        self._binops = {
            '+': ('fadd', 'add', 'add'),
            '-': ('fsub', 'sub', 'sub'),
            '*': ('fmul', 'mul', 'mul'),
            '/': ('fdiv', 'udiv', 'sdiv'),
            '%': ('frem', 'urem', 'srem'),

            '&': (None, 'and_', 'and_'),
            '|': (None, 'or_', 'or_'),
            '^': (None, 'xor', 'xor'),

            # TODO: other ops
        }

    def init_comparisons(self):
        """
        Define binary operation LLVM instructions. Do this in a function in
        case llvm-py is not installed.
        """
        self._compare_mapping_float = {
            '>':  llvm.core.FCMP_OGT,
            '<':  llvm.core.FCMP_OLT,
            '==': llvm.core.FCMP_OEQ,
            '>=': llvm.core.FCMP_OGE,
            '<=': llvm.core.FCMP_OLE,
            '!=': llvm.core.FCMP_ONE,
        }

        self._compare_mapping_sint = {
            '>':  llvm.core.ICMP_SGT,
            '<':  llvm.core.ICMP_SLT,
            '==': llvm.core.ICMP_EQ,
            '>=': llvm.core.ICMP_SGE,
            '<=': llvm.core.ICMP_SLE,
            '!=': llvm.core.ICMP_NE,
        }

        self._compare_mapping_uint = {
            '>':  llvm.core.ICMP_UGT,
            '<':  llvm.core.ICMP_ULT,
            '==': llvm.core.ICMP_EQ,
            '>=': llvm.core.ICMP_UGE,
            '<=': llvm.core.ICMP_ULE,
            '!=': llvm.core.ICMP_NE,
        }

    def visit_BinopNode(self, node):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)

        op = node.operator
        if (node.type.is_int or node.type.is_float) and node.operator in self._binops:
            idx = 0 if node.type.is_float else node.type.is_int + node.type.signed
            llvm_method_name = self._binops[op][idx]
            meth = getattr(self.builder, llvm_method_name)
            if not lhs.type == rhs.type:
                node.print_tree(self.context)
                assert False, (node.lhs.type, node.rhs.type, lhs.type, rhs.type)
            return meth(lhs, rhs)
        elif node.operator in self._compare_mapping_float:
            return self.generate_compare(node, op, lhs, rhs)
        elif node.type.is_pointer:
            if node.rhs.type.is_pointer:
                lhs, rhs = rhs, lhs
            return self.builder.gep(lhs, [rhs])
        else:
            raise minierror.CompileError(
                node, "Binop %s (type=%s) not implemented for types (%s, %s)" % (
                                                op, node.type, lhs.type, rhs.type))

    def generate_compare(self, node, op, lhs_value, rhs_value):
        op = node.operator
        lop = None

        if node.lhs.type.is_float and node.rhs.type.is_float:
            lfunc = self.builder.fcmp
            lop = self._compare_mapping_float[op]
        elif node.lhs.type.is_int and node.rhs.type.is_int:
            lfunc = self.builder.icmp
            if node.lhs.type.signed and node.rhs.type.signed:
                lop = self._compare_mapping_sint[op]
            elif not (node.lhs.type.signed or node.rhs.type.signed):
                lop = self._compare_mapping_uint[op]

        if lop is None:
            raise minierror.CompileError(
                node, "%s for types (%s, %s)" % (node.operator,
                                                 node.lhs.type, node.rhs.type))

        return lfunc(lop, lhs_value, rhs_value)

    def visit_UnopNode(self, node):
        result = self.visit(node.operand)
        if node.operator == '-':
            return self.builder.neg(result)
        elif node.operator == '+':
            return result
        else:
            raise NotImplementedError(node.operator)

    def visit_TempNode(self, node):
        if node not in self.declared_temps:
            llvm_temp = self._declare_temp(node)
        else:
            llvm_temp = self.llvm_temps[node]

        if self.in_lhs_expr:
            return llvm_temp
        else:
            return self.builder.load(llvm_temp)

    def _mangle_temp(self, node):
        name = node.repr_name or node.name
        if name in self.temp_names:
            name = "%s%d" % (name, len(self.declared_temps))
        node.name = name
        self.temp_names.add(name)
        self.declared_temps.add(node)

    def _declare_temp(self, node, rhs_result=None):
        if node not in self.declared_temps:
            self._mangle_temp(node)

        llvm_temp = self.alloca(node.type)
        self.llvm_temps[node] = llvm_temp
        return llvm_temp

    def alloca(self, type, name=''):
        bb = self.builder.basic_block
        self.builder.position_at_beginning(self.entry_bb)
        llvm_temp = self.builder.alloca(type.to_llvm(self.context), name)
        self.builder.position_at_end(bb)
        return llvm_temp

    def visit_AssignmentExpr(self, node):
        self.in_lhs_expr += 1
        lhs = self.visit(node.lhs)
        self.in_lhs_expr -= 1
        rhs = self.visit(node.rhs)
        return self.builder.store(rhs, lhs)

    def visit_SingleIndexNode(self, node):
        in_lhs_expr = self.in_lhs_expr
        if in_lhs_expr:
            self.in_lhs_expr -= 1
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        if in_lhs_expr:
            self.in_lhs_expr += 1

        result = self.builder.gep(lhs, [rhs])

        if self.in_lhs_expr:
            return result
        else:
            return self.builder.load(result)

    def visit_DereferenceNode(self, node):
        node = self.astbuilder.index(node.operand, self.astbuilder.constant(0))
        return self.visit_SingleIndexNode(node)

    def visit_SizeofNode(self, node):
        return self.visit(self.astbuilder.constant(node.sizeof_type.itemsize,
                                                   node.type))

    def visit_Variable(self, node):
        value = self.symtab[node.name]
        return value

    def visit_ArrayAttribute(self, node):
        return self.symtab[node.name]

    def visit_NoopExpr(self, node):
        pass

    def visit_ResolvedVariable(self, node):
        return self.visit(node.element)

    def visit_JumpNode(self, node):
        return self.builder.branch(self.visit(node.label))

    def visit_JumpTargetNode(self, node):
        basic_block = self.visit(node.label)
        self.builder.branch(basic_block)
        self.builder.position_at_end(basic_block)

    def visit_LabelNode(self, node):
        if node not in self.labels:
            self.labels[node] = self.append_basic_block(node.label)

        return self.labels[node]

    def handle_string_constant(self, b, constant):
        # Create a global string constant, if it doesn't already
        # Based on code in Numba. Seems easier than creating a stack variable
        string_constants = self.context.string_constants = getattr(
                                    self.context, 'string_constants', {})
        if constant in string_constants:
            lvalue = string_constants[constant]
        else:
            lstring = llvm.core.Constant.stringz(constant)
            lvalue = self.context.llvm_module.add_global_variable(
                        lstring.type, "__string_%d" % len(string_constants))
            lvalue.initializer = lstring
            lvalue.linkage = llvm.core.LINKAGE_INTERNAL

            lzero = self.visit(b.constant(0))
            lvalue = self.builder.gep(lvalue, [lzero, lzero])
            string_constants[constant] = lvalue

        return lvalue

    def visit_ConstantNode(self, node):
        b = self.astbuilder

        ltype = node.type.to_llvm(self.context)
        constant = node.value

        if node.type.is_float:
            lvalue = llvm.core.Constant.real(ltype, constant)
        elif node.type.is_int:
            lvalue = llvm.core.Constant.int(ltype, constant)
        elif node.type.is_pointer and self.pyval == 0:
            lvalue = llvm.core.ConstantPointerNull
        elif node.type.is_c_string:
            lvalue = self.handle_string_constant(b, constant)
        else:
            raise NotImplementedError("Constant %s of type %s" % (constant,
                                                                  node.type))

        return lvalue

    def visit_FuncCallNode(self, node):
        llvm_args = list(self.results(node.args))
        llvm_func = self.visit(node.func_or_pointer)

        signature = node.func_or_pointer.type
        if signature.struct_by_reference:
            result = handle_struct_passing(
                        self.builder, self.alloca, llvm_args, signature)

        llvm_call = self.builder.call(llvm_func, llvm_args)

        if node.inline:
            self.inline_calls.append(llvm_call)

        if (signature.struct_by_reference and
                minitypes.pass_by_ref(signature.return_type)):
            return self.builder.load(result)
        else:
            return llvm_call

    def visit_FuncNameNode(self, node):
        func_type = node.type.to_llvm(self.context)
        func = self.context.llvm_module.get_or_insert_function(func_type,
                                                               node.name)
        return func

    def visit_FuncRefNode(self, node):
        raise NotImplementedError
