#! /usr/bin/env python
# ______________________________________________________________________
'''Defines a bytecode based LLVM translator for llnumba code.
'''
# ______________________________________________________________________
# Module imports

import opcode

import llvm.core as lc

import opcode_util

import bytetype
from bytecode_visitor import BytecodeFlowVisitor
from byte_flow import BytecodeFlowBuilder
from byte_control import ControlFlowBuilder
from phi_injector import PhiInjector, synthetic_opname

# ______________________________________________________________________
# Module data

# XXX Stolen from numba.translate:

_compare_mapping_float = {'>':lc.FCMP_OGT,
                           '<':lc.FCMP_OLT,
                           '==':lc.FCMP_OEQ,
                           '>=':lc.FCMP_OGE,
                           '<=':lc.FCMP_OLE,
                           '!=':lc.FCMP_ONE}

_compare_mapping_sint = {'>':lc.ICMP_SGT,
                          '<':lc.ICMP_SLT,
                          '==':lc.ICMP_EQ,
                          '>=':lc.ICMP_SGE,
                          '<=':lc.ICMP_SLE,
                          '!=':lc.ICMP_NE}

# XXX Stolen from numba.llvm_types:

class LLVMCaster (object):
    def build_pointer_cast(_, builder, lval1, lty2):
        return builder.bitcast(lval1, lty2)

    def build_int_cast(_, builder, lval1, lty2, unsigned = False):
        width1 = lval1.type.width
        width2 = lty2.width
        ret_val = lval1
        if width2 > width1:
            if unsigned:
                ret_val = builder.zext(lval1, lty2)
            else:
                ret_val = builder.sext(lval1, lty2)
        elif width2 < width1:
            ret_val = builder.trunc(lval1, lty2)
        return ret_val

    def build_float_ext(_, builder, lval1, lty2):
        return builder.fpext(lval1, lty2)

    def build_float_trunc(_, builder, lval1, lty2):
        return builder.fptrunc(lval1, lty2)

    def build_int_to_float_cast(_, builder, lval1, lty2, unsigned = False):
        ret_val = None
        if unsigned:
            ret_val = builder.uitofp(lval1, lty2)
        else:
            ret_val = builder.sitofp(lval1, lty2)
        return ret_val

    def build_float_to_int_cast(_, builder, lval1, lty2, unsigned = False):
        ret_val = None
        if unsigned:
            ret_val = builder.fptoui(lval1, lty2)
        else:
            ret_val = builder.fptosi(lval1, lty2)
        return ret_val

    CAST_MAP = {
        lc.TYPE_POINTER : build_pointer_cast,
        lc.TYPE_INTEGER: build_int_cast,
        (lc.TYPE_FLOAT, lc.TYPE_DOUBLE) : build_float_ext,
        (lc.TYPE_DOUBLE, lc.TYPE_FLOAT) : build_float_trunc,
        (lc.TYPE_INTEGER, lc.TYPE_FLOAT) : build_int_to_float_cast,
        (lc.TYPE_INTEGER, lc.TYPE_DOUBLE) : build_int_to_float_cast,
        (lc.TYPE_FLOAT, lc.TYPE_INTEGER) : build_float_to_int_cast,
        (lc.TYPE_DOUBLE, lc.TYPE_INTEGER) : build_float_to_int_cast,

    }

    @classmethod
    def build_cast(cls, builder, lval1, lty2, *args, **kws):
        ret_val = lval1
        lty1 = lval1.type
        lkind1 = lty1.kind
        lkind2 = lty2.kind

        if lkind1 == lkind2:

            if lkind1 in cls.CAST_MAP:
                ret_val = cls.CAST_MAP[lkind1](cls, builder, lval1, lty2,
                                               *args, **kws)
            else:
                raise NotImplementedError(lkind1)
        else:
            map_index = (lkind1, lkind2)
            if map_index in cls.CAST_MAP:
                ret_val = cls.CAST_MAP[map_index](cls, builder, lval1, lty2,
                                                  *args, **kws)
            else:
                raise NotImplementedError(lkind1, lkind2)
        return ret_val

# ______________________________________________________________________
# Class definitions

class LLVMTranslator (BytecodeFlowVisitor):
    '''Transformer responsible for visiting a set of bytecode flow
    trees, emitting LLVM code.

    Unlike other translators in :py:mod:`numba.llnumba`, this
    incorporates the full transformation chain, starting with
    :py:class:`numba.llnumba.byte_flow.BytecodeFlowBuilder`, then
    :py:class:`numba.llnumba.byte_control.ControlFlowBuilder`, and
    then :py:class:`numba.llnumba.phi_injector.PhiInjector`.'''

    def __init__ (self, llvm_module = None, *args, **kws):
        '''Constructor for LLVMTranslator.'''
        super(LLVMTranslator, self).__init__(*args, **kws)
        if llvm_module is None:
            llvm_module = lc.Module.new('Translated_Module_%d' % (id(self),))
        self.llvm_module = llvm_module
        self.bytecode_flow_builder = BytecodeFlowBuilder()
        self.control_flow_builder = ControlFlowBuilder()
        self.phi_injector = PhiInjector()

    def translate (self, function, llvm_type = None, env = None):
        '''Translate a function to the given LLVM function type.

        If no type is given, then assume the function is of LLVM type
        "void ()".

        The optional env parameter allows extension of the global
        environment.'''
        if llvm_type is None:
            llvm_type = lc.Type.function(lvoid, ())
        if env is None:
            env = {}
        else:
            env = env.copy()
        env.update((name, method)
                   for name, method in lc.Builder.__dict__.items()
                   if not name.startswith('_'))
        env.update((name, value)
                   for name, value in bytetype.__dict__.items()
                   if not name.startswith('_'))
        self.loop_stack = []
        self.llvm_type = llvm_type
        self.function = function
        self.code_obj = opcode_util.get_code_object(function)
        func_globals = getattr(function, 'func_globals',
                               getattr(function, '__globals__', {})).copy()
        func_globals.update(env)
        self.globals = func_globals
        nargs = self.code_obj.co_argcount
        self.cfg = self.control_flow_builder.visit(
            self.bytecode_flow_builder.visit(self.code_obj), nargs)
        flow = self.phi_injector.visit_cfg(self.cfg, nargs)
        ret_val = self.visit(flow)
        del self.cfg
        del self.globals
        del self.code_obj
        del self.function
        del self.llvm_type
        del self.loop_stack
        return ret_val

    def enter_flow_object (self, flow):
        super(LLVMTranslator, self).enter_flow_object(flow)
        self.llvm_function = self.llvm_module.add_function(
            self.llvm_type, self.function.__name__)
        self.llvm_blocks = {}
        self.llvm_definitions = {}
        self.pending_phis = {}
        for block in self.block_list:
            if 0 in self.cfg.blocks_reaching[block]:
                bb = self.llvm_function.append_basic_block(
                    'BLOCK_%d' % (block,))
                self.llvm_blocks[block] = bb

    def exit_flow_object (self, flow):
        super(LLVMTranslator, self).exit_flow_object(flow)
        ret_val = self.llvm_function
        del self.pending_phis
        del self.llvm_definitions
        del self.llvm_blocks
        del self.llvm_function
        return ret_val

    def enter_block (self, block):
        ret_val = False
        if block in self.llvm_blocks:
            self.llvm_block = self.llvm_blocks[block]
            self.builder = lc.Builder.new(self.llvm_block)
            ret_val = True
        return ret_val

    def exit_block (self, block):
        del self.llvm_block
        del self.builder

    def visit_synthetic_op (self, i, op, arg, *args, **kws):
        method = getattr(self, 'op_%s' % (synthetic_opname[op],))
        return method(i, op, arg, *args, **kws)

    def op_REF_ARG (self, i, op, arg, *args, **kws):
        return [self.llvm_function.args[arg]]

    def op_BUILD_PHI (self, i, op, arg, *args, **kws):
        phi_type = None
        incoming = []
        pending = []
        for child_arg in arg:
            child_block, _, child_opname, child_arg, _ = child_arg
            assert child_opname == 'REF_DEF'
            if child_arg in self.llvm_definitions:
                child_def = self.llvm_definitions[child_arg]
                if phi_type is None:
                    phi_type = child_def.type
                incoming.append((child_block, child_def))
            else:
                pending.append((child_arg, child_block))
        phi = self.builder.phi(phi_type)
        for block_index, defn in incoming:
            phi.add_incoming(defn, self.llvm_blocks[block_index])
        for defn_index, block_index in pending:
            if defn_index not in self.pending_phis:
                self.pending_phis[defn_index] = []
            self.pending_phis[defn_index].append((phi, block_index))
        return [phi]

    def op_DEFINITION (self, i, op, def_index, *args, **kws):
        assert len(args) == 1
        arg = args[0]
        if def_index in self.pending_phis:
            for phi, block_index in self.pending_phis[def_index]:
                phi.add_incoming(arg, self.llvm_blocks[block_index])
        self.llvm_definitions[def_index] = arg
        return args

    def op_REF_DEF (self, i, op, arg, *args, **kws):
        return [self.llvm_definitions[arg]]

    def op_BINARY_ADD (self, i, op, arg, *args, **kws):
        arg1, arg2 = args
        if arg1.type.kind == lc.TYPE_INTEGER:
            ret_val = [self.builder.add(arg1, arg2)]
        elif arg1.type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
            ret_val = [self.builder.fadd(arg1, arg2)]
        elif arg1.type.kind == lc.TYPE_POINTER:
            ret_val = [self.builder.gep(arg1, [arg2])]
        else:
            raise NotImplementedError("LLVMTranslator.op_BINARY_ADD for %r" %
                                      (args,))
        return ret_val

    def op_BINARY_AND (self, i, op, arg, *args, **kws):
        return [self.builder.and_(args[0], args[1])]

    def op_BINARY_DIVIDE (self, i, op, arg, *args, **kws):
        arg1, arg2 = args
        if arg1.type.kind == lc.TYPE_INTEGER:
            ret_val = [self.builder.sdiv(arg1, arg2)]
        elif arg1.type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
            ret_val = [self.builder.fdiv(arg1, arg2)]
        else:
            raise NotImplementedError("LLVMTranslator.op_BINARY_DIVIDE for %r"
                                      % (args,))
        return ret_val

    def op_BINARY_FLOOR_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_BINARY_FLOOR_DIVIDE")

    def op_BINARY_LSHIFT (self, i, op, arg, *args, **kws):
        return [self.builder.shl(args[0], args[1])]

    def op_BINARY_MODULO (self, i, op, arg, *args, **kws):
        arg1, arg2 = args
        if arg1.type.kind == lc.TYPE_INTEGER:
            ret_val = [self.builder.srem(arg1, arg2)]
        elif arg1.type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
            ret_val = [self.builder.frem(arg1, arg2)]
        else:
            raise NotImplementedError("LLVMTranslator.op_BINARY_MODULO for %r"
                                      % (args,))
        return ret_val

    def op_BINARY_MULTIPLY (self, i, op, arg, *args, **kws):
        arg1, arg2 = args
        if arg1.type.kind == lc.TYPE_INTEGER:
            ret_val = [self.builder.mul(arg1, arg2)]
        elif arg1.type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
            ret_val = [self.builder.fmul(arg1, arg2)]
        else:
            raise NotImplementedError("LLVMTranslator.op_BINARY_MULTIPLY for "
                                      "%r" % (args,))
        return ret_val

    def op_BINARY_OR (self, i, op, arg, *args, **kws):
        return [self.builder.or_(args[0], args[1])]

    def op_BINARY_POWER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_BINARY_POWER")

    def op_BINARY_RSHIFT (self, i, op, arg, *args, **kws):
        return [self.builder.lshr(args[0], args[1])]

    def op_BINARY_SUBSCR (self, i, op, arg, *args, **kws):
        arr_val = args[0]
        index_vals = args[1:]
        ret_val = gep_result = self.builder.gep(arr_val, index_vals)
        if (gep_result.type.kind == lc.TYPE_POINTER and
            gep_result.type.pointee.kind != lc.TYPE_POINTER):
            ret_val = self.builder.load(gep_result)
        return [ret_val]

    def op_BINARY_SUBTRACT (self, i, op, arg, *args, **kws):
        arg1, arg2 = args
        if arg1.type.kind == lc.TYPE_INTEGER:
            ret_val = [self.builder.sub(arg1, arg2)]
        elif arg1.type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
            ret_val = [self.builder.fsub(arg1, arg2)]
        else:
            raise NotImplementedError("LLVMTranslator.op_BINARY_SUBTRACT for "
                                      "%r" % (args,))
        return ret_val

    op_BINARY_TRUE_DIVIDE = op_BINARY_DIVIDE

    def op_BINARY_XOR (self, i, op, arg, *args, **kws):
        return [self.builder.xor(args[0], args[1])]

    def op_BREAK_LOOP (self, i, op, arg, *args, **kws):
        return [self.builder.branch(self.llvm_blocks[arg])]

    def op_BUILD_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_BUILD_SLICE")

    def op_BUILD_TUPLE (self, i, op, arg, *args, **kws):
        return args

    def op_CALL_FUNCTION (self, i, op, arg, *args, **kws):
        fn = args[0]
        args = args[1:]
        if isinstance(fn, lc.Type):
            if isinstance(fn, lc.FunctionType):
                ret_val = [self.builder.call(
                    self.llvm_module.get_or_insert_function(fn, fn.__name__),
                    args)]
            else:
                assert len(args) == 1
                ret_val = [LLVMCaster.build_cast(self.builder, args[0], fn)]
        elif fn.__name__ in lc.Builder.__dict__:
            ret_val = [fn(self.builder, *args)]
        else:
            raise NotImplementedError("Don't know how to call %r!" % (fn,))
        return ret_val

    def op_CALL_FUNCTION_KW (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_CALL_FUNCTION_KW")

    def op_CALL_FUNCTION_VAR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_CALL_FUNCTION_VAR")

    def op_CALL_FUNCTION_VAR_KW (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_CALL_FUNCTION_VAR_KW")

    def op_COMPARE_OP (self, i, op, arg, *args, **kws):
        arg1, arg2 = args
        cmp_kind = opcode.cmp_op[arg]
        if isinstance(arg1.type, lc.IntegerType):
            ret_val = [self.builder.icmp(_compare_mapping_sint[cmp_kind],
                                         arg1, arg2)]
        elif arg1.type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
            ret_val = [self.builder.fcmp(_compare_mapping_float[cmp_kind],
                                         arg1, arg2)]
        else:
            raise NotImplementedError('Comparison of type %r' % (arg1.type,))
        return ret_val

    def op_CONTINUE_LOOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_CONTINUE_LOOP")

    def op_DELETE_ATTR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_DELETE_ATTR")

    def op_DELETE_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_DELETE_SLICE")

    def op_FOR_ITER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_FOR_ITER")

    def op_GET_ITER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_GET_ITER")

    op_INPLACE_ADD = op_BINARY_ADD
    op_INPLACE_AND = op_BINARY_AND
    op_INPLACE_DIVIDE = op_BINARY_DIVIDE
    op_INPLACE_FLOOR_DIVIDE = op_BINARY_FLOOR_DIVIDE
    op_INPLACE_LSHIFT = op_BINARY_LSHIFT
    op_INPLACE_MODULO = op_BINARY_MODULO
    op_INPLACE_MULTIPLY = op_BINARY_MULTIPLY
    op_INPLACE_OR = op_BINARY_OR
    op_INPLACE_POWER = op_BINARY_POWER
    op_INPLACE_RSHIFT = op_BINARY_RSHIFT
    op_INPLACE_SUBTRACT = op_BINARY_SUBTRACT
    op_INPLACE_TRUE_DIVIDE = op_BINARY_TRUE_DIVIDE
    op_INPLACE_XOR = op_BINARY_XOR

    def op_JUMP_ABSOLUTE (self, i, op, arg, *args, **kws):
        return [self.builder.branch(self.llvm_blocks[arg])]

    def op_JUMP_FORWARD (self, i, op, arg, *args, **kws):
        return [self.builder.branch(self.llvm_blocks[i + arg + 3])]

    def op_JUMP_IF_FALSE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_JUMP_IF_FALSE")

    def op_JUMP_IF_FALSE_OR_POP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_JUMP_IF_FALSE_OR_POP")

    def op_JUMP_IF_TRUE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_JUMP_IF_TRUE")

    def op_JUMP_IF_TRUE_OR_POP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_JUMP_IF_TRUE_OR_POP")

    def op_LOAD_ATTR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_LOAD_ATTR")

    def op_LOAD_CONST (self, i, op, arg, *args, **kws):
        py_val = self.code_obj.co_consts[arg]
        if isinstance(py_val, int):
            ret_val = [lc.Constant.int(bytetype.lc_int, py_val)]
        elif isinstance(py_val, float):
            ret_val = [lc.Constant.double(py_val)]
        elif py_val == None:
            ret_val = [None]
        else:
            raise NotImplementedError('Constant converstion for %r' %
                                      (py_val,))
        return ret_val

    def op_LOAD_GLOBAL (self, i, op, arg, *args, **kws):
        ret_val = self.globals[self.code_obj.co_names[arg]]
        if not hasattr(ret_val, '__name__'):
            ret_val.__name__ = self.code_obj.co_names[arg]
        return [ret_val]

    def op_POP_BLOCK (self, i, op, arg, *args, **kws):
        self.loop_stack.pop()
        return [self.builder.branch(self.llvm_blocks[i + 1])]

    def op_POP_JUMP_IF_FALSE (self, i, op, arg, *args, **kws):
        return [self.builder.cbranch(args[0], self.llvm_blocks[i + 3],
                                     self.llvm_blocks[arg])]

    def op_POP_JUMP_IF_TRUE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_POP_JUMP_IF_TRUE")

    def op_POP_TOP (self, i, op, arg, *args, **kws):
        return args

    def op_RETURN_VALUE (self, i, op, arg, *args, **kws):
        if args[0] is None:
            ret_val = [self.builder.ret_void()]
        else:
            ret_val = [self.builder.ret(args[0])]
        return ret_val

    def op_SETUP_LOOP (self, i, op, arg, *args, **kws):
        self.loop_stack.append((i, arg))
        return [self.builder.branch(self.llvm_blocks[i + 3])]

    def op_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_SLICE")

    def op_STORE_ATTR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_STORE_ATTR")

    def op_STORE_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_STORE_SLICE")

    def op_STORE_SUBSCR (self, i, op, arg, *args, **kws):
        store_val, arr_val, index_val = args
        dest_addr = self.builder.gep(arr_val, [index_val])
        return [self.builder.store(store_val, dest_addr)]

    def op_UNARY_CONVERT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_UNARY_CONVERT")

    def op_UNARY_INVERT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_UNARY_INVERT")

    def op_UNARY_NEGATIVE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_UNARY_NEGATIVE")

    def op_UNARY_NOT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_UNARY_NOT")

    def op_UNARY_POSITIVE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("LLVMTranslator.op_UNARY_POSITIVE")

# ______________________________________________________________________

def translate_function (func, lltype, llvm_module = None, **kws):
    '''Given a function and an LLVM function type, emit LLVM code for
    that function using a new LLVMTranslator instance.'''
    translator = LLVMTranslator(llvm_module)
    translator.translate(func, lltype, kws)
    return translator

# ______________________________________________________________________
# Main (self-test) routine

def main (*args):
    from tests import llfuncs, llfunctys
    if not args:
        args = ('doslice',)
    elif 'all' in args:
        args = [llfunc
                for llfunc in dir(llfuncs) if not llfunc.startswith('_')]
    llvm_module = lc.Module.new('test_module')
    for arg in args:
        translate_function(getattr(llfuncs, arg), getattr(llfunctys, arg),
                           llvm_module)
    print(llvm_module)

# ______________________________________________________________________

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of byte_translator.py
