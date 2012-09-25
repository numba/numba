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
from numba import ndarray_helpers, translate

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
        return self._fpass[mod]

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
                 optimize=True, func_name=None, **kwds):
        super(LLVMCodeGenerator, self).__init__(context, func, ast)

        self.func_name = func_name or func.__name__
        self.func_signature = func_signature

        self.symtab = symtab

        self.blocks = {} # stores id => basic-block

        # code generation attributes
        self.mod = LLVMContextManager().get_default_module()
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
            logger.exception(e)
            logger.error('Unhandled visit to %s', ast.dump(node))
            raise
            #raise compiler_errors.InternalError(node, 'Not yet implemented.')
        else:
            try:
                self._nodes.append(node) # push current node
                return fn(node)
            except Exception as e:
                logger.exception(e)
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
                not var.type.is_object and
                var.is_local):
                # Not argument and not builtin type.
                # Allocate storage for all variables.
                stackspace = self.builder.alloca(var.ltype)
                stackspace.name = 'var_%s' % var.name
                var.lvalue = stackspace

        # TODO: Put current function into symbol table for recursive call

        self.setup_return()

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

        logger.debug(self.lfunc)
        # Verify code generation
        self.lfunc.verify()
        if self.optimize:
            fp = LLVMContextManager().get_function_pass_manager(self.mod)
            fp.run(self.lfunc)

    def get_ctypes_func(self, llvm=True):
        ee = LLVMContextManager().get_execution_engine()
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
            retval = self.builder.load(self.return_value)
            ret_type = self.func_signature.return_type
            if ret_type.is_object or ret_type.is_array:
                ret_ltype = object_.to_llvm(self.context)
                obj = self.builder.bitcast(retval, ret_ltype)
                sig, lfunc = self.function_cache.function_by_name('Py_IncRef')
                self.builder.call(lfunc, [obj])

            self.builder.ret(retval)

    # __________________________________________________________________________

    def visit_ConstNode(self, node):
        return node.value(self)

    def generate_load_symbol(self, name):
        var = self.symtab[name]
        if var.is_local:
            return self.builder.load(var.lvalue)
        else:
            raise NotImplementedError(var)

    def generate_store_symbol(self, name):
        return self.symtab[name].lvalue

    def visit_Attribute(self, node):
        result = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            return self.generate_load_attribute(node, result)
        else:
            self.generate_store_attribute(node, result)

    def visit_Assign(self, node):
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
            retval = self.builder.bitcast(retval,
                                          self.return_value.type.pointee)
            self.builder.store(retval, self.return_value)

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

    def generate_for_range(self, for_node, target, iternode, body):
        '''
        Implements simple for loops with iternode as range, xrange
        '''
        # assert isinstance(target.ctx, ast.Store)
        target = self.visit(target)

        start, stop, step = self.visitlist(iternode.args)

        bb_cond = self.append_basic_block('for.cond')
        bb_incr = self.append_basic_block('for.incr')
        bb_body = self.append_basic_block('for.body')
        bb_exit = self.append_basic_block('for.exit')

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

    def visit_While(self, node):
        bb_cond = self.append_basic_block('while.cond')
        bb_body = self.append_basic_block('while.body')
        bb_exit = self.append_basic_block('while.exit')

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

    def generate_constant_int(self, val, ty=_types.int_):
        lconstant = lc.Constant.int(ty.to_llvm(self.context), val)
        return lconstant

    _binops = {
        ast.Add: ('fadd', 'add'),
        ast.Sub: ('fsub', 'sub'),
        ast.Mult: ('fmul', 'mul'),
        ast.Div: ('fdiv', 'div'),
        # TODO: reuse previously implemented modulo
    }

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = type(node.op)

        if (node.type.is_int or node.type.is_float) and op in self._binops:
            llvm_method_name = self._binops[op][node.type.is_int]
            meth = getattr(self.builder, llvm_method_name)
            if not lhs.type == rhs.type:
                print ast.dump(node)
                assert False
            result = meth(lhs, rhs)
        else:
            raise Exception(op, node.type, lhs, rhs)

        return result

    def visit_CoercionNode(self, node):
        val = self.visit(node.node)
        node_type = node.node.type
        if node.dst_type.is_object and not node_type.is_object:
            return self.object_coercer.convert_single(node_type, val,
                                                      name=node.name)
        elif node.node.type != node.dst_type:
            # logger.debug('Coerce %s --> %s', node.node.type, node.dst_type)
            val = self.caster.cast(val, node.dst_type.to_llvm(self.context))

        return val

    def visit_Subscript(self, node):
        if (node.value.type.is_array and
                isinstance(node.value, nodes.DataPointerNode)):
            # Indexing
            assert not node.type.is_array
            value = self.visit(node.value.node)
            lptr = node.value.subscript(self, value, self.visit(node.slice))
        elif node.type.is_array:
            # Slicing
            if isinstance(node.ctx, ast.Load):
                getitem = self.function_cache.call('PyObject_GetItem',
                                                   node.value, node.slice)
                return self.visit(getitem)
            else:
                raise NotImplementedError
        elif node.value.type.is_carray:
            value = self.visit(node.value)
            lptr = self.builder.gep(value, [self.visit(node.slice)])

        if isinstance(node.ctx, ast.Load): # load the value
            return self.builder.load(lptr)
        elif isinstance(node.ctx, ast.Store): # return a pointer for storing
            return lptr
        else:
            # unreachable
            raise AssertionError("Unknown subscript context: %s" % node.ctx)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_ExtSlice(self, node):
        return self.visitlist(node.dims)

    def visit_Call(self, node):
        raise Exception("This node should have been replaced")

    def visit_Tuple(self, node):
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

    def visit_NativeCallNode(self, node):
        # TODO: Refcounts + error check
        largs = self.visitlist(node.args)
        return self.builder.call(node.llvm_func, largs, name=node.name)

    def visit_TempNode(self, node):
        if node.llvm_temp is None:
            value = self.builder.alloca(self.to_llvm(node.type))
            node.llvm_temp = value

        return node.llvm_temp

    def visit_TempLoadNode(self, node):
        return self.builder.load(self.visit(node.temp))

    def visit_TempStoreNode(self, node):
        return self.visit(node.temp)

    def visit_ObjectTempNode(self, node):
        assert not isinstance(node.node, nodes.ObjectTempNode)

        bb = self.builder.basic_block
        # Initialize temp to NULL at beginning of function
        self.builder.position_at_beginning(self.lfunc.get_entry_basic_block())
        name = getattr(node.node, 'name', 'object') + '_temp'
        lhs = self.builder.alloca(llvm_types._pyobject_head_struct_p,
                                  name=name)
        self.generate_assign(self.visit(nodes.NULL_obj), lhs)
        node.llvm_temp = lhs

        # Assign value
        self.builder.position_at_end(bb)
        rhs = self.visit(node.node)
        self.generate_assign(rhs, lhs)

        # goto error if NULL
        self.object_coercer.check_err(rhs)

        # Generate Py_XDECREF(temp) (call Py_DecRef only if not NULL)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.current_cleanup_bb)
        sig, py_decref = self.function_cache.function_by_name('Py_DecRef')

        def cleanup(b, bb_true, bb_endif):
            b.call(py_decref, [self.builder.load(lhs)])
            b.branch(bb_endif)

        self.object_coercer.check_err(self.builder.load(lhs),
                                      callback=cleanup,
                                      cmp=llvm.core.ICMP_NE)
        self.current_cleanup_bb = self.builder.basic_block

        self.builder.position_at_end(bb)
        return self.builder.load(lhs, name=name + '_load')

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

def if_badval(translator, llvm_result, badval, callback, cmp):
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
        self.translator = translator
        self.builder = translator.builder
        sig, self.py_buildvalue = translator.function_cache.function_by_name(
                                                              'Py_BuildValue')
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

    def convert_single(self, type, llvm_result, name=''):
        "Generate code to convert an LLVM value to a Python object"
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
