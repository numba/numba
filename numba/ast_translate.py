import opcode
import sys
import types
import __builtin__
import functools
from contextlib import contextmanager
import ast

import numpy as np

import llvm
from llvm import _ObjectCache, WeakValueDictionary
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le

from utils import itercode, debugout
#from ._ext import make_ufunc
from .cfg import ControlFlowGraph
from .llvm_types import _plat_bits, _int1, _int8, _int32, _intp, _intp_star, \
    _void_star, _float, _double, _complex64, _complex128, _pyobject_head, \
    _trace_refs_, _head_len, _numpy_struct,  _numpy_array, \
    _numpy_array_field_ofs, _LLVMCaster
#from .multiarray_api import MultiarrayAPI # not used
from .symtab import Variable
from . import _numba_types as _types
from ._numba_types import Complex64, Complex128, BuiltinType

import numba
from numba import *
from . import visitors, nodes, llvm_types
from .minivect import minitypes
from numba.pymothoa import compiler_errors
from numba import ndarray_helpers, translate, error

import logging
logger = logging.getLogger(__name__)

if __debug__:
    import pprint

_int32_zero = lc.Constant.int(_int32, 0)

def map_to_function(func, typs, mod):
    typs = [str_to_llvmtype(x) if isinstance(x, str) else x for x in typs]
    INTR = getattr(lc, 'INTR_%s' % func.__name__.upper())
    return lc.Function.intrinsic(mod, INTR, typs)

class DelayedObj(object):
    def __init__(self, base, args):
        self.base = base
        self.args = args

    def get_start(self):
        if len(self.args) > 1:
            ret_val = self.args[0]
        else:
            # FIXME: Need to infer case where this might be over floats.
            ret_val = Variable(_types.int32, lvalue=lc.Constant.int(_int32, 0))
        return ret_val

    def get_inc(self):
        if len(self.args) > 2:
            ret_val = self.args[2]
        else:
            # FIXME: Need to infer case where this might be over floats.
            ret_val = Variable(type=_types.int32, lvalue=lc.Constant.int(_int32, 1))
        return ret_val

    def get_stop(self):
        return self.args[0 if (len(self.args) == 1) else 1]

class MethodReference(object):
    def __init__(self, object_var, py_method):
        self.object_var = object_var
        self.py_method = py_method


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

_compare_mapping_uint = {'>':lc.ICMP_UGT,
                          '<':lc.ICMP_ULT,
                          '==':lc.ICMP_EQ,
                          '>=':lc.ICMP_UGE,
                          '<=':lc.ICMP_ULE,
                          '!=':lc.ICMP_NE}


class LLVMContextManager(object):
    # TODO: make these instance attributes and create a singleton global object
    _ee = None          # execution engine
    _mods = {}          # module's name => module instance
    _fpass = {}         # module => function passes
    _DEFAULT_MODULE = 'default'

    def __init__(self):
        self._initialize()

    @classmethod
    def _initialize(cls):
        if not cls._mods: # no modules yet
            # Create default module
            default_mod = cls._init_module(cls._DEFAULT_MODULE)

            # Create execution engine
            # NOTE: EE owns all registered modules
            cls._ee = le.ExecutionEngine.new(default_mod)

    @classmethod
    def _init_module(cls, name):
        '''
        Initialize a module with the given name;
        Prepare pass managers for it.
        '''
        mod = lc.Module.new(name)
        cls._mods[name] = mod

        # TODO: We should use a PassManagerBuilder so that we can use O1, O2, O3

        fpm = lp.FunctionPassManager.new(mod)
        cls._fpass[mod] = fpm

        # NOTE: initialize() link all passes into LLVM.
        fpm.initialize()

        #fpm.add(lp.PASS_PROMOTE_MEMORY_TO_REGISTER)
        #fpm.add(lp.PASS_DEAD_CODE_ELIMINATION)

        # NOTE: finalize() unlink all passes from LLVM. I don't see any reason
        #       for a program to do so.
        # fpm.finalize()

        return mod

    def create_module(self, name):
        '''
        Create a llvm Module and add it to the execution engine.

        NOTE: Will we ever need this?
        '''
        mod = self._init_module(name)
        self._ee.add_module(mod)
        return mod

    def get_default_module(self):
        return self.get_module(self._DEFAULT_MODULE)

    def get_function_pass_manager(self, name_or_mod):
        if isinstance(name_or_mod, basestring):
            mod = name_or_mod
        else:
            mod = name_or_mod

        if mod in self._fpass:
            fpm = self._fpass[mod]
        else:
            fpm = lp.FunctionPassManager.new(mod)
            self._fpass[mod] = fpm
            fpm.initialize()

        return fpm

    def get_module(self, name):
        return self._mods[name]

    def get_execution_engine(self):
        return self._ee

class LLVMCodeGenerator(visitors.NumbaVisitor):
    """
    Translate a Python AST to LLVM. Each visit_* method should directly
    return an LLVM value.
    """

    def __init__(self, context, func, ast, func_signature, symtab,
                 optimize=True, func_name=None,
                 llvm_module=None, llvm_ee=None, **kwds):
        super(LLVMCodeGenerator, self).__init__(context, func, ast)

        self.func_name = func_name or func.__name__
        self.func_signature = func_signature

        self.symtab = symtab

        self.blocks = {} # stores id => basic-block

        # code generation attributes
        self.mod = llvm_module or LLVMContextManager().get_default_module()
        self.ee = llvm_ee or LLVMContextManager().get_execution_engine()
        self.module_utils = translate._LLVMModuleUtils()

        # self.ma_obj = None # What is this?
        self.optimize = optimize
        self.flags = kwds

        # internal states
        self._nodes = []  # for tracking parent nodes

    # ________________________ visitors __________________________

    @property
    def current_node(self):
        return self._nodes[-1]

    def visit(self, node):
        # logger.debug('visiting %s', ast.dump(node))
        try:
            fn = getattr(self, 'visit_%s' % type(node).__name__)
        except AttributeError as e:
            # logger.exception(e)
            logger.error('Unhandled visit to %s', ast.dump(node))
            raise
            #raise compiler_errors.InternalError(node, 'Not yet implemented.')
        else:
            try:
                self._nodes.append(node) # push current node
                return fn(node)
            except Exception as e:
                # logger.exception(e)
                raise
            finally:
                self._nodes.pop() # pop current node

    # __________________________________________________________________________

    def setup_func(self):
        self.lfunc_type = self.to_llvm(self.func_signature)
        self.lfunc = self.mod.add_function(self.lfunc_type, self.func_name)
        self.nlocals = len(self.fco.co_varnames)
        # Local variables with LLVM types
        self._locals = [None] * self.nlocals

        # Add entry block for alloca.
        entry = self.append_basic_block('entry')
        self.builder = lc.Builder.new(entry)
        self.caster = _LLVMCaster(self.builder)
        self.object_coercer = ObjectCoercer(self)

        for i, (ltype, argname) in enumerate(zip(self.lfunc.args, self.argnames)):
            ltype.name = argname
            # Store away arguments in locals

            variable = self.symtab[argname]

            stackspace = self.builder.alloca(ltype.type)    # allocate on stack
            stackspace.name = 'arg_%s' % argname
            variable.lvalue = stackspace

            self.builder.store(ltype, stackspace) # store arg value

            self._locals[i] = variable

        for name, var in self.symtab.items():
            # FIXME: 'None' should be handled as a special case (probably).
            if (name not in self.argnames and
                not isinstance(var.type, BuiltinType) and
                var.is_local):
                # Not argument and not builtin type.
                # Allocate storage for all variables.
                stackspace = self.builder.alloca(var.ltype)
                stackspace.name = 'var_%s' % var.name
                var.lvalue = stackspace

        # TODO: Put current function into symbol table for recursive call
        self.setup_return()

        self.loop_beginnings = []
        self.loop_exits = []

        # Control FLow
        # Not needed for now.
        #    self.cfg = None
        #    self.blocks_locals = {}
        #    self.pending_phis = {}
        #    self.pending_blocks = {}
        #    self.stack = []
        #    self.loop_stack = []

    def to_llvm(self, type):
        return type.to_llvm(self.context)

    def translate(self):
        self.setup_func()

        assert isinstance(self.ast, ast.FunctionDef)

        # Handle the doc string for the function
        # FIXME: Ignoring it for now
        if (isinstance(self.ast.body[0], ast.Expr) and
            isinstance(self.ast.body[0].value, ast.Str)):
            # Python doc string
            logger.info('Ignoring python doc string.')
            statements = self.ast.body[1:]
        else:
            statements = self.ast.body

        for node in statements: # do codegen for each statement
            self.visit(node)
        else:
            if not self.is_block_terminated():
                # self.builder.ret_void()
                self.builder.branch(self.cleanup_label)

        self.terminate_cleanup_blocks()

        # Done code generation
        del self.builder  # release the builder to make GC happy

        logger.debug("ast translated function: %s", self.lfunc)
        # Verify code generation
        self.lfunc.verify()
        if self.optimize:
            fp = LLVMContextManager().get_function_pass_manager(self.mod)
            fp.run(self.lfunc)

    def get_ctypes_func(self, llvm=True):
        ee = self.ee
        import ctypes
        sig = self.func_signature
        restype = _types.convert_to_ctypes(sig.return_type)

        # FIXME: Switch to PYFUNCTYPE so it does not release the GIL.
        #
        #    prototype = ctypes.CFUNCTYPE(restype,
        #                                 *[_types.convert_to_ctypes(x)
        #                                       for x in sig.args])
        prototype = ctypes.PYFUNCTYPE(restype,
                                     *[_types.convert_to_ctypes(x)
                                           for x in sig.args])


        if hasattr(restype, 'make_ctypes_prototype_wrapper'):
            # See numba.utils.ComplexMixin for an example of
            # make_ctypes_prototype_wrapper().
            prototype = restype.make_ctypes_prototype_wrapper(prototype)
        if llvm:
            # July 10, 2012: PY_CALL_TO_LLVM_CALL_MAP is removed recent commit.
            #
            #    PY_CALL_TO_LLVM_CALL_MAP[self.func] = \
            #        self.build_call_to_translated_function
            return prototype(ee.get_pointer_to_function(self.lfunc))
        else:
            return prototype(self.func)

    def build_call_to_translated_function(self, target_translator, args):
        # FIXME: At some point, I assume we'll actually want to index
        # by argument types, so this will grab the best suited
        # translation of a function based on the caller circumstance.
        if len(args) != len(self.arg_types):
            raise TypeError("Mismatched argument count in call to translated "
                            "function (%r)." % (self.func))
        ee = LLVMContextManager().get_execution_engine()
        func_name = '%s_%d' % (self.func.__name__, id(self))
        try:
            lfunc_ptr_ptr = target_translator.mod.get_global_variable_named(
                func_name)
        except:
            # FIXME: This is linkage at its grossest (and least
            # dynamic - what if the called function is recompiled at
            # some point?).  See if there is some way to link this up
            # using symbols and module linkage settings.
            lfunc_ptr_ty = self.lfunc.type
            lfunc_ptr_ptr = target_translator.mod.add_global_variable(
                lfunc_ptr_ty, func_name)
            lfunc_ptr_ptr.initializer = lc.Constant.inttoptr(
                lc.Constant.int(_intp, ee.get_pointer_to_function(self.lfunc)),
                lfunc_ptr_ty)
            lfunc_ptr_ptr.linkage = lc.LINKAGE_INTERNAL
        lfunc = target_translator.builder.load(lfunc_ptr_ptr)
        if __debug__:
            print "build_call_to_translated_function():", str(lfunc)

        largs = [arg.llvm(convert_to_strtype(param_typ),
                          builder = target_translator.builder)
                 for arg, param_typ in zip(args, self.arg_types)]
        res = target_translator.builder.call(lfunc, largs)
        res_typ = convert_to_strtype(self.ret_type)
        return res, res_typ

    def setup_return(self):
        # Assign to this value which will be returned
        self.is_void_return = self.func_signature.return_type.is_void
        if not self.is_void_return:
            llvm_ret_type = self.func_signature.return_type.to_llvm(self.context)
            self.return_value = self.builder.alloca(llvm_ret_type,
                                                    "return_value")

        # All non-NULL object emporaries are DECREFed here
        self.cleanup_label = self.append_basic_block('cleanup_label')
        self.current_cleanup_bb = self.cleanup_label

        bb = self.builder.basic_block
        # Jump here in case of an error
        self.error_label = self.append_basic_block("error_label")
        self.builder.position_at_end(self.error_label)
        # Set error return value and jump to cleanup
        self.visit(self.ast.error_return)
        self.builder.position_at_end(bb)

    def terminate_cleanup_blocks(self):
        self.builder.position_at_end(self.current_cleanup_bb)
        if self.is_void_return:
            self.builder.ret_void()
        else:
            ret_type = self.func_signature.return_type
            self.builder.ret(self.builder.load(self.return_value))

    # __________________________________________________________________________

    def visit_ConstNode(self, node):
        return node.value(self)

    def generate_load_symbol(self, name):
        var = self.symtab[name]
        if var.is_local:
            return self.builder.load(var.lvalue, name='load_' + name)
        else:
            raise NotImplementedError("global variables:", var)

    def generate_store_symbol(self, name):
        return self.symtab[name].lvalue

    def visit_Attribute(self, node):
        result = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            return self.generate_load_attribute(node, result)
        else:
            self.generate_store_attribute(node, result)

    def visit_Assign(self, node):
        ast.dump(node)
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        return self.generate_assign(value, target)

    def visit_Num(self, node):
        if node.type.is_int:
            return self.generate_constant_int(node.n)
        elif node.type.is_float:
            return self.generate_constant_real(node.n)
        else:
            assert node.type.is_complex
            return self.generate_constant_complex(node.n)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load): # load
            return self.generate_load_symbol(node.id)
        elif isinstance(node.ctx, ast.Store): # store
            return self.generate_store_symbol(node.id)
            # unreachable
        raise AssertionError('unreachable')

    def visit_For(self, node):
        if node.orelse:
            # FIXME
            raise NotImplementedError('Else in for-loop is not implemented.')

        if node.iter.type.is_range:
            self.generate_for_range(node, node.target, node.iter, node.body)
        else:
            raise NotImplementedError(ast.dump(node))

    def visit_BoolOp(self, node):
        assert len(node.values) == 2
        raise NotImplementedError

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return self.generate_not(operand)
        raise NotImplementedError(ast.dump(node))

    def visit_Compare(self, node):
        lhs, rhs = node.left, node.right
        lhs_lvalue, rhs_lvalue = self.visitlist([lhs, rhs])

        op_map = {
            ast.Gt    : '>',
            ast.Lt    : '<',
            ast.GtE   : '>=',
            ast.LtE   : '<=',
            ast.Eq    : '==',
            ast.NotEq : '!=',
        }

        op = op_map[type(node.ops[0])]
        if lhs.type.is_float and rhs.type.is_float:
            lfunc = self.builder.fcmp
            lop = _compare_mapping_float[op]
            return self.builder.fcmp(lop, lhs_lvalue, rhs_lvalue)
        elif lhs.type.is_int_like and rhs.type.is_int_like:
            lfunc = self.builder.icmp
            # FIXME
            assert lhs.type.signed, 'Unsigned comparator has not been implemented'
            lop = _compare_mapping_sint[op]
        else:
            raise TypeError(lhs.type)

        return lfunc(lop, lhs_lvalue, rhs_lvalue)

    def visit_If(self, node):
        test = self.visit(node.test)
        iftrue_body = node.body
        orelse_body = node.orelse

        bb_true = self.append_basic_block('if.true')
        bb_false = self.append_basic_block('if.false')
        bb_endif = self.append_basic_block('if.end')
        self.builder.cbranch(test, bb_true, bb_false)

        # if true then
        self.builder.position_at_end(bb_true)
        for stmt in iftrue_body:
            self.visit(stmt)
        else: # close the block
            if not self.is_block_terminated():
                self.builder.branch(bb_endif)

        # else
        self.builder.position_at_end(bb_false)
        for stmt in orelse_body:
            self.visit(stmt)
        else: # close the block
            if not self.is_block_terminated():
                self.builder.branch(bb_endif)

        # endif
        self.builder.position_at_end(bb_endif)

    def append_basic_block(self, name='unamed'):
        idx = len(self.blocks)
        bb = self.lfunc.append_basic_block('%s_%d'%(name, idx))
        self.blocks[idx]=bb
        return bb

    def generate_assign(self, lvalue, ltarget):
        '''
        Generate assignment operation and automatically cast value to
        match the target type.
        '''
        if lvalue.type != ltarget.type.pointee:
            lvalue = self.caster.cast(lvalue, ltarget.type.pointee)

        self.builder.store(lvalue, ltarget)

    def visit_Return(self, node):
        if node.value is not None:
            assert not self.is_void_return
            retval = self.visit(node.value)
            rettype = self.func_signature.return_type
            if rettype.is_object or rettype.is_array or rettype.is_pointer:
                retval = self.builder.bitcast(retval,
                                              self.return_value.type.pointee)

            if not retval.type == self.return_value.type.pointee:
                print retval.type
                print self.return_value.type
                print ast.dump(node)
                assert False

            self.builder.store(retval, self.return_value)

            ret_type = self.func_signature.return_type
            if ret_type.is_object or ret_type.is_array:
                sig, lfunc = self.function_cache.function_by_name('Py_IncRef')
                ltype_obj = object_.to_llvm(self.context)
                self.builder.call(lfunc, [self.builder.bitcast(retval,
                                                               ltype_obj)])

        self.builder.branch(self.cleanup_label)

        # if node.value is not None:
        #     self.builder.ret(self.visit(node.value))
        # else:
        #     self.builder.ret_void()

    def is_block_terminated(self):
        '''
        Check if the current basicblock is properly terminated.
        That means the basicblock is ended with a branch or return
        '''
        instructions = self.builder.basic_block.instructions
        return instructions and instructions[-1].is_terminator

    def setup_loop(self, bb_cond, bb_exit):
        self.loop_beginnings.append(bb_cond)
        self.loop_exits.append(bb_exit)

    def teardown_loop(self):
        self.loop_beginnings.pop()
        self.loop_exits.pop()

    def generate_for_range(self, for_node, target, iternode, body):
        '''
        Implements simple for loops with iternode as range, xrange
        '''
        # assert isinstance(target.ctx, ast.Store)
        bb_cond = self.append_basic_block('for.cond')
        bb_incr = self.append_basic_block('for.incr')
        bb_body = self.append_basic_block('for.body')
        bb_exit = self.append_basic_block('for.exit')
        self.setup_loop(bb_cond, bb_exit)

        target = self.visit(target)
        start, stop, step = self.visitlist(iternode.args)

        # generate initializer
        self.generate_assign(start, target)
        self.builder.branch(bb_cond)

        # generate condition
        self.builder.position_at_end(bb_cond)
        op = _compare_mapping_sint['<']
        # Please visit the children properly. Assuming ast.Name breaks the entire approach...
        #ctvalue = self.generate_load_symbol(target.id)
        index = self.visit(for_node.index)
        cond = self.builder.icmp(op, index, stop)
        self.builder.cbranch(cond, bb_body, bb_exit)

        # generate increment
        self.builder.position_at_end(bb_incr)
        self.builder.store(self.builder.add(index, step), target)
        self.builder.branch(bb_cond)

        # generate body
        self.builder.position_at_end(bb_body)
        for stmt in body:
            self.visit(stmt)

        if not self.is_block_terminated():
            self.builder.branch(bb_incr)

        # move to exit block
        self.builder.position_at_end(bb_exit)
        self.teardown_loop()

    def visit_While(self, node):
        bb_cond = self.append_basic_block('while.cond')
        bb_body = self.append_basic_block('while.body')
        bb_exit = self.append_basic_block('while.exit')
        self.setup_loop(bb_cond, bb_exit)

        self.builder.branch(bb_cond)

        # condition
        self.builder.position_at_end(bb_cond)
        cond = self.visit(node.test)
        self.builder.cbranch(cond, bb_body, bb_exit)

        # body
        self.builder.position_at_end(bb_body)
        self.visitlist(node.body)

        # loop or exit
        self.builder.branch(bb_cond)
        self.builder.position_at_end(bb_exit)
        self.teardown_loop()

    def visit_Continue(self, node):
        assert self.loop_beginnings # Python syntax should ensure this
        self.builder.branch(self.loop_beginnings[-1])

    def visit_Break(self, node):
        assert self.loop_exits # Python syntax should ensure this
        self.builder.branch(self.loop_exits[-1])

    def visit_Suite(self, node):
        self.visitlist(node.body)
        return None

    def generate_constant_int(self, val, ty=_types.int_):
        lconstant = lc.Constant.int(ty.to_llvm(self.context), val)
        return lconstant

    _binops = {
        ast.Add: ('fadd', ('add', 'add')),
        ast.Sub: ('fsub', ('sub', 'sub')),
        ast.Mult: ('fmul', ('mul', 'mul')),
        ast.Div: ('fdiv', ('udiv', 'sdiv')),
        # TODO: reuse previously implemented modulo
    }

    def _handle_pow(self, node, lhs, rhs):
        assert node.right.type.is_int
        ltype = node.type.to_llvm(self.context)
        restype = translate.llvmtype_to_strtype(ltype)
        func = self.module_utils.get_py_int_pow(self.mod, restype)
        return self.builder.call(func, (lhs, rhs))

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = type(node.op)

        valid_type = node.type.is_int or node.type.is_float
        if valid_type and op in self._binops:
            llvm_method_name = self._binops[op][node.type.is_int]
            if node.type.is_int:
                llvm_method_name = llvm_method_name[node.type.signed]
            meth = getattr(self.builder, llvm_method_name)
            if not lhs.type == rhs.type:
                print ast.dump(node)
                assert False
            result = meth(lhs, rhs)
        elif valid_type and op == ast.Pow:
            return self._handle_pow(node, lhs, rhs)
        else:
            logging.debug(ast.dump(node))
            raise Exception(op, node.type, lhs, rhs)

        return result

    def visit_CoercionNode(self, node):
        val = self.visit(node.node)

        if node.type == node.node.type:
            return val

        # logger.debug('Coerce %s --> %s', node.node.type, node.dst_type)
        node_type = node.node.type
        dst_type = node.dst_type
        ldst_type = dst_type.to_llvm(self.context)
        if node_type.is_pointer and dst_type.is_int:
            val = self.builder.ptrtoint(val, ldst_type)
        elif node_type.is_int and dst_type.is_pointer:
            val = self.builder.inttoptr(val, ldst_type)
        else:
            val = self.caster.cast(val, node.dst_type.to_llvm(self.context))

        return val

    def visit_CoerceToObject(self, node):
        result = self.visit(node.node)
        if not node.node.type == minitypes.object_:
            result = self.object_coercer.convert_single(node.node.type, result,
                                                        name=node.name)
        return result

    def visit_CoerceToNative(self, node):
        assert node.node.type.is_tuple
        val = self.visit(node.node)
        return self.object_coercer.to_native(node.dst_type, val,
                                             name=node.name)

    def visit_Subscript(self, node):
#        if node.value.type.is_array or node.value.type.is_object:
#            raise NotImplementedError("This node should have been replaced")

        assert node.value.type.is_carray or node.value.type.is_c_string, \
                                                        node.value.type
        value = self.visit(node.value)
        lptr = self.builder.gep(value, [self.visit(node.slice)])
        if node.slice.type.is_int:
            lptr = self._handle_ctx(node, lptr)

        return lptr

    def _handle_ctx(self, node, lptr):
        if isinstance(node.ctx, ast.Load):
            return self.builder.load(lptr)
        else:
            return lptr

    def visit_DataPointerNode(self, node):
        assert node.type.is_array
        lvalue = self.visit(node.node)
        lindices = self.visit(node.slice)
        lptr = node.subscript(self, lvalue, lindices)
        return self._handle_ctx(node, lptr)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_ExtSlice(self, node):
        return self.visitlist(node.dims)

    def visit_Call(self, node):
        raise Exception("This node should have been replaced")

    def visit_Tuple(self, node):
        raise NotImplementedError("This node should have been replaced")
        assert isinstance(node.ctx, ast.Load)
        types = [n.type for n in node.elts]
        largs = self.visitlist(node.elts)
        return self.object_coercer.build_tuple(types, largs)

    def visit_Dict(self, node):
        key_types = [k.type for k in node.keys]
        value_types = [v.type for v in node.values]
        llvm_keys = self.visitlist(node.keys)
        llvm_values = self.visitlist(node.values)
        result = self.object_coercer.build_dict(key_types, value_types,
                                                llvm_keys, llvm_values)
        return result

    def visit_ObjectInjectNode(self, node):
        # FIXME: Currently uses the runtime address of the python function.
        #        Sounds like a hack.
        self.func.live_objects.append(node.object)
        addr = id(node.object)
        obj_addr_int = self.generate_constant_int(addr, _types.Py_ssize_t)
        obj = self.builder.inttoptr(obj_addr_int,
                                    _types.object_.to_llvm(self.context))
        return obj

    def visit_ObjectCallNode(self, node):
        args_tuple = self.visit(node.args_tuple)
        kwargs_dict = self.visit(node.kwargs_dict)

        if node.function is None:
            node.function = nodes.ObjectInjectNode(node.py_func)
        lfunc_addr = self.visit(node.function)

        # call PyObject_Call
        largs = [lfunc_addr, args_tuple, kwargs_dict]
        _, pyobject_call = self.function_cache.function_by_name('PyObject_Call')

        res = self.builder.call(pyobject_call, largs, name=node.name)
        return self.caster.cast(res, node.variable.type.to_llvm(self.context))

    def visit_NativeCallNode(self, node, largs=None):
        if largs is None:
            largs = self.visitlist(node.args)
        return self.builder.call(node.llvm_func, largs, name=node.name)

    def visit_LLVMIntrinsicNode(self, node):
        intr = getattr(llvm.core, 'INTR_' + node.py_func.__name__.upper())
        largs = self.visitlist(node.args)
        node.llvm_func = llvm.core.Function.intrinsic(
                self.mod, intr, [larg.type for larg in largs])
        return self.visit_NativeCallNode(node, largs=largs)

    def visit_MathCallNode(self, node):
        try:
            lfunc = self.mod.get_function_named(node.name)
        except llvm.LLVMException:
            lfunc_type = node.signature.to_llvm(self.context)
            lfunc = self.mod.add_function(lfunc_type, node.name)

        node.llvm_func = lfunc
        return self.visit_NativeCallNode(node)

    def alloca(self, type, name='', change_bb=True):
        return self.llvm_alloca(self.to_llvm(type), name, change_bb)

    def llvm_alloca(self, ltype, name='', change_bb=True):
        return llvm_alloca(self.lfunc, self.builder, ltype, name, change_bb)

    def visit_TempNode(self, node):
        if node.llvm_temp is None:
            value = self.alloca(node.type)
            node.llvm_temp = value

        return node.llvm_temp

    def visit_TempLoadNode(self, node):
        return self.builder.load(self.visit(node.temp))

    def visit_TempStoreNode(self, node):
        return self.visit(node.temp)

    def visit_ObjectTempNode(self, node):
        if isinstance(node.node, nodes.ObjectTempNode):
            return self.visit(node.node)

        bb = self.builder.basic_block
        # Initialize temp to NULL at beginning of function
        self.builder.position_at_beginning(self.lfunc.get_entry_basic_block())
        name = getattr(node.node, 'name', 'object') + '_temp'
        lhs = self.llvm_alloca(llvm_types._pyobject_head_struct_p,
                               name=name, change_bb=False)
        self.generate_assign(self.visit(nodes.NULL_obj), lhs)
        node.llvm_temp = lhs

        # Assign value
        self.builder.position_at_end(bb)
        rhs = self.visit(node.node)
        self.generate_assign(rhs, lhs)

        # goto error if NULL
        self.object_coercer.check_err(rhs)

        # Generate Py_XDECREF(temp) (call Py_DecRef only if not NULL)
        self.decref_temp(lhs)
        return self.builder.load(lhs, name=name + '_load')

    def decref_temp(self, temp, func='Py_DecRef'):
        bb = self.builder.basic_block
        self.builder.position_at_end(self.current_cleanup_bb)
        sig, py_decref = self.function_cache.function_by_name(func)

        def cleanup(b, bb_true, bb_endif):
            object_ltype = object_.to_llvm(self.context)
            b.call(py_decref, [b.bitcast(b.load(temp), object_ltype)])
            b.branch(bb_endif)

        self.object_coercer.check_err(self.builder.load(temp),
                                      callback=cleanup,
                                      cmp=llvm.core.ICMP_NE)
        self.current_cleanup_bb = self.builder.basic_block

        self.builder.position_at_end(bb)

    def incref_temp(self, temp):
        return self.decref_temp(temp, func='Py_IncRef')

    def visit_ArrayAttributeNode(self, node):
        array = self.visit(node.array)
        acc = ndarray_helpers.PyArrayAccessor(self.builder, array)

        attr_name = node.attr_name
        if attr_name == 'shape':
            attr_name = 'dimensions'

        return getattr(acc, attr_name)

    def visit_ShapeAttributeNode(self, node):
        # hurgh, no dispatch on superclasses?
        return self.visit_ArrayAttributeNode(node)

class DisposalVisitor(visitors.NumbaVisitor):
    # TODO: handle errors, check for NULL before calling DECREF

    def __init__(self, context, func, ast, builder):
        super(DisposalVisitor, self).__init__(context, func, ast)
        self.builder = builder

    def visit_TempNode(self, node):
        self.visit(node.node)
        lfunc = self.function_cache.function_by_name('Py_DecRef')
        self.builder.call(lfunc, node.llvm_temp)

def llvm_alloca(lfunc, builder, ltype, name='', change_bb=True):
    if change_bb:
        bb = builder.basic_block
    builder.position_at_beginning(lfunc.get_entry_basic_block())
    lstackvar = builder.alloca(ltype, name)
    if change_bb:
        builder.position_at_end(bb)
    return lstackvar

def if_badval(translator, llvm_result, badval, callback, cmp=llvm.core.ICMP_EQ):
    # Use llvm_cbuilder :(
    b = translator.builder

    bb_true = translator.append_basic_block('if.true')
    bb_endif = translator.append_basic_block('if.end')

    test = b.icmp(cmp, llvm_result, badval)
    b.cbranch(test, bb_true, bb_endif)

    b.position_at_end(bb_true)
    callback(b, bb_true, bb_endif)
    # b.branch(bb_endif)
    b.position_at_end(bb_endif)

    return llvm_result

class ObjectCoercer(object):
    type_to_buildvalue_str = {
        char: "b",
        short: "h",
        int_: "i",
        long_: "l",
        longlong: "L",
        Py_ssize_t: "n",
        npy_intp: "n", # ?
        size_t: "n", # ?
        uchar: "B",
        ushort: "H",
        uint: "I",
        ulong: "k",
        ulonglong: "K",

        float_: "f",
        double: "d",
        complex128: "D",

        object_: "O",
        bool_: "p",
        c_string_type: "s",
    }

    def __init__(self, translator):
        self.context = translator.context
        self.translator = translator
        self.builder = translator.builder
        sig, self.py_buildvalue = translator.function_cache.function_by_name(
                                                              'Py_BuildValue')
        sig, self.pyarg_parsetuple = translator.function_cache.function_by_name(
                                                              'PyArg_ParseTuple')
        self.function_cache = translator.function_cache
        self.NULL = self.translator.visit(nodes.NULL_obj)

    def check_err(self, llvm_result, callback=None, cmp=llvm.core.ICMP_EQ):
        """
        Check for errors. If the result is NULL, and error should have been set
        Jumps to translator.error_label if an exception occurred.
        """
        int_result = self.translator.builder.ptrtoint(llvm_result,
                                                       llvm_types._intp)
        NULL = llvm.core.Constant.int(int_result.type, 0)

        if callback:
            if_badval(self.translator, int_result, NULL,
                      callback=callback or default_callback, cmp=cmp)
        else:
            test = self.builder.icmp(cmp, int_result, NULL)
            bb = self.translator.append_basic_block('no_error')
            self.builder.cbranch(test, self.translator.error_label, bb)
            self.builder.position_at_end(bb)

        return llvm_result

    def check_err_int(self, llvm_result, badval):
        llvm_badval = llvm.core.Constant.int(llvm_result.type, badval)
        if_badval(self.translator, llvm_result, llvm_badval,
                  callback=lambda b, *args: b.branch(self.translator.error_label))

    def lstr(self, types, fmt=None):
        typestrs = []
        for type in types:
            if type.is_array:
                type = object_
            typestrs.append(self.type_to_buildvalue_str[type])

        str = "".join(typestrs)
        if fmt is not None:
            str = fmt % str

        return self.translator.visit(nodes.ConstNode(str, c_string_type))

    def buildvalue(self, *largs, **kwds):
        # The caller should check for errors using check_err or by wrapping
        # its node in an ObjectTempNode
        name = kwds.get('name', '')
        result = self.builder.call(self.py_buildvalue, largs, name=name)
        return result

    def npy_intp_to_py_ssize_t(self, llvm_result, type):
        if type == minitypes.npy_intp:
            lpy_ssize_t = minitypes.Py_ssize_t.to_llvm(self.context)
            llvm_result = self.translator.caster.cast(llvm_result, lpy_ssize_t)
            type = minitypes.Py_ssize_t

        return llvm_result, type

    def py_ssize_t_to_npy_intp(self, llvm_result, type):
        if type == minitypes.npy_intp:
            lnpy_intp = minitypes.npy_intp.to_llvm(self.context)
            llvm_result = self.translator.caster.cast(llvm_result, lnpy_intp)
            type = minitypes.Py_ssize_t

        return llvm_result, type

    def convert_single(self, type, llvm_result, name=''):
        "Generate code to convert an LLVM value to a Python object"
        llvm_result, type = self.npy_intp_to_py_ssize_t(llvm_result, type)
        lstr = self.lstr([type])
        return self.buildvalue(lstr, llvm_result, name=name)

    def build_tuple(self, types, llvm_values):
        "Build a tuple from a bunch of LLVM values"
        assert len(types) == len(llvm_values)
        lstr = self.lstr(types, fmt="(%s)")
        return self.buildvalue(lstr, *llvm_values, name='tuple')

    def build_dict(self, key_types, value_types, llvm_keys, llvm_values):
        "Build a dict from a bunch of LLVM values"
        types = []
        for k, v in zip(key_types, value_types):
            types.append(k)
            types.append(v)

        lstr = self.lstr(types, fmt="{%s}")
        largs = llvm_keys + llvm_values
        return self.buildvalue(lstr, *largs, name='dict')

    def parse_tuple(self, lstr, llvm_tuple, types, name=''):
        lresults = []
        for i, type in enumerate(types):
            var = llvm_alloca(self.translator.lfunc, self.builder,
                              type.to_llvm(self.context),
                              name=name and "%s%d" % (name, i))
            lresults.append(var)

        largs = [llvm_tuple, lstr] + lresults
        parse_result = self.builder.call(self.pyarg_parsetuple, largs)
        self.check_err_int(parse_result, 0)
        return map(self.builder.load, lresults)

    def to_native(self, type, llvm_tuple, name=''):
        "Generate code to convert a Python object to an LLVM value"
        lstr = self.lstr([type])
        lresult, = self.parse_tuple(lstr, llvm_tuple, [type], name=name)
        return lresult



