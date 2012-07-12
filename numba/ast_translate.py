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
from ._ext import make_ufunc
from .cfg import ControlFlowGraph
from .llvm_types import _plat_bits, _int1, _int8, _int32, _intp, _intp_star, \
    _void_star, _float, _double, _complex64, _complex128, _pyobject_head, \
    _trace_refs_, _head_len, _numpy_struct,  _numpy_array, \
    _numpy_array_field_ofs
from .multiarray_api import MultiarrayAPI
from .symtab import Variable
from . import _numba_types as _types
from ._numba_types import Complex64, Complex128, BuiltinType

from . import visitors, nodes, llvm_types
from .minivect import minitypes

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

class _LLVMCaster(object):
    # NOTE: Using a class to lower namespace polution here.  The
    # following would be class methods, but we'd have to index them
    # using "class.method" in the cast dictionary to get the proper
    # binding, and that'd only succeed after the class has been built.

    def __init__(self, builder):
        self.builder = builder

    def cast(self, lvalue, dst_ltype):
        src_ltype = lvalue.type
        return self.build_cast(self.builder, lvalue, dst_ltype)

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
            print("Warning: Perfoming downcast.  May lose information.")
            ret_val = builder.trunc(lval1, lty2)
        return ret_val

    def build_float_cast(_, builder, lval1, lty2):
        raise NotImplementedError("FIXME")

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
        lc.TYPE_INTEGER : build_int_cast,
        lc.TYPE_FLOAT : build_float_cast,
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
            map_index = (lkind1, lkind2)
            if map_index in cls.CAST_MAP:
                ret_val = cls.CAST_MAP[map_index](cls, builder, lval1, lty2,
                                                  *args, **kws)
        return ret_val


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

        fpm.add(lp.PASS_PROMOTE_MEMORY_TO_REGISTER)
        fpm.add(lp.PASS_DEAD_CODE_ELIMINATION)

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
        try:
            fn = getattr(self, 'visit_%s' % type(node).__name__)
        except AttributeError as e:
            logger.exception(e)
            logger.error('Unhandled visit to %s', ast.dump(node))
            raise InternalError(node, 'Not yet implemented.')
        else:
            try:
                self._nodes.append(node) # push current node
                return fn(node)
            except Exception as e:
                logger.exception(e)
                raise
            finally:
                self._nodes.pop() # pop current node

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

    def visit_Subscript(self, node):
        dptr, strides, ndim = self.visit(node.value)
        indices = self.visit(node.slice)
        offset = self.generate_constant_int(0)

        for i, index in zip(range(ndim), reversed(indices)):
            # why is the indices reversed?
            stride_ptr = self.builder.gep(strides,
                                          [self.generate_constant_int(i)])
            stride = self.builder.load(stride_ptr)
            index = self.caster.cast(index, stride.type)
            offset = self.caster.cast(offset, stride.type)
            offset = self.builder.add(offset, self.builder.mul(index, stride))

        data_ty = node.value.variable.type.dtype.to_llvm(self.context)
        data_ptr_ty = lc.Type.pointer(data_ty)

        dptr_plus_offset = self.builder.gep(dptr, [offset])


        ptr = self.builder.bitcast(dptr_plus_offset, data_ptr_ty)

        if isinstance(node.ctx, ast.Load): # load the value
            return self.builder.load(ptr)
        elif isinstance(node.ctx, ast.Store): # return a pointer for storing
            return ptr
        else:
            # unreachable
            raise AssertionError("Unknown subscript context: %s" % node.ctx)


    def visit_DataPointerNode(self, node):
        dptr, strides = node.data_descriptors(self.builder)
        ndim = node.ndim
        return dptr, strides, ndim

    def visit_ExtSlice(self, node):
        return self.visitlist(node.dims)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load): # load
            return self.generate_load_symbol(node.id)
        elif isinstance(node.ctx, ast.Store): # store
            return self.generate_store_symbol(node.id)
        # unreachable
        raise AssertionError('unreachable')

    def visit_If(self, node):
        test = self.visit(node.test)
        iftrue_body = node.body
        orelse_body = node.orelse
        self.generate_if(test, iftrue_body, orelse_body)

    def visit_For(self, node):
        if node.orelse:
            # FIXME
            raise NotImplementedError('Else in for-loop is not implemented.')

        if node.iter.type.is_range:
            self.generate_for_range(node, node.target, node.iter, node.body)
        else:
            raise NotImplementedError(node.iter, node.iter.type)

    def visit_BoolOp(self, node):
        if len(node.values)!=2: raise AssertionError
        return self.generate_boolop(node.op, node.values[0], node.values[1])

    def generate_boolop(self, op_class, lhs, rhs):
        raise NotImplementedError

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return self.generate_not(operand)
        raise NotImplementedError(ast.dump(node))

    # __________________________________________________________________________

    def setup_func(self):
        # Seems not necessary
        # convert (numpy) array arguments to a pointer to the dtype
        #        self.func_signature = minitypes.FunctionType(
        #            return_type=self.func_signature.return_type,
        #            args=[arg_type.dtype.pointer() if arg_type.is_array else arg_type
        #                      for arg_type in self.func_signature.args])

        self.lfunc_type = self.to_llvm(self.func_signature)
        self.lfunc = self.mod.add_function(self.lfunc_type, self.func_name)
        self.nlocals = len(self.fco.co_varnames)
        # Local variables with LLVM types
        self._locals = [None] * self.nlocals

        # Add entry block for alloca.
        entry = self.append_basic_block('entry')
        self.builder = lc.Builder.new(entry)
        self.caster = _LLVMCaster(self.builder)

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
                self.builder.ret_void()

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

        prototype = ctypes.CFUNCTYPE(restype,
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

    # __________________________________________________________________________

    def visit_ConstNode(self, node):
        return node.value(self.builder)

    def generate_load_symbol(self, name):
        var = self.symtab[name]
        if var.is_local:
            return self.builder.load(var.lvalue)
        else:
            raise NotImplementedError(var)

    def generate_store_symbol(self, name):
        return self.symtab[name].lvalue

    def visit_Compare(self, node):
        lhs, rhs = node.left, node.comparators[0]
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

    def generate_if(self, test, iftrue_body, orelse_body):
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
        if lvalue.type != ltarget.type:
            lvalue = self.caster.cast(lvalue, lvalue.type)

        self.builder.store(lvalue, ltarget)

    def visit_Return(self, node):
        if node.value is not None:
            self.builder.ret(self.visit(node.value))

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
        assert isinstance(target.ctx, ast.Store)
        ctstore = self.visit(target)

        start, stop, step = self.visitlist(iternode.args)

        bb_cond = self.append_basic_block('for.cond')
        bb_incr = self.append_basic_block('for.incr')
        bb_body = self.append_basic_block('for.body')
        bb_exit = self.append_basic_block('for.exit')

        # generate initializer
        self.generate_assign(start, ctstore)
        self.builder.branch(bb_cond)

        # generate condition
        self.builder.position_at_end(bb_cond)
        op = _compare_mapping_sint['<']
        ctvalue = self.generate_load_symbol(target.id)
        cond = self.builder.icmp(op, ctvalue, stop)
        self.builder.cbranch(cond, bb_body, bb_exit)

        # generate increment
        self.builder.position_at_end(bb_incr)
        ctvalue = self.generate_load_symbol(target.id)
        ctvalue_plus_step = self.builder.add(ctvalue, step)
        self.builder.store(ctvalue_plus_step, ctstore)
        self.builder.branch(bb_cond)

        # generate body
        self.builder.position_at_end(bb_body)
        for stmt in body:
            self.visit(stmt)

        if not self.is_block_terminated():
            self.builder.branch(bb_incr)

        # move to exit block
        self.builder.position_at_end(bb_exit)

    def generate_constant_int(self, val, ty=_types.int_):
        lconstant = lc.Constant.int(ty.to_llvm(self.context), val)
        return lconstant

    _binops = {
        ast.Add: ('fadd', 'add'),
        ast.Sub: ('fsub', 'sub'),
        ast.Mult: ('fmul', 'mul'),
        ast.Div: ('fdiv', 'div'),
    }

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = type(node.op)

        if (node.type.is_int or node.type.is_float) and op in self._binops:
            llvm_method_name = self._binops[op][node.type.is_int]
            meth = getattr(self.builder, llvm_method_name)
            result = meth(lhs, rhs)
        else:
            raise Exception(op_class, type, lhs, rhs)

        return result

    def visit_CoercionNode(self, node):
        val = self.visit(node.node)
        if node.node.type != node.dst_type:
            val = self.caster.cast(val, node.dst_type.to_llvm(self.context))

        return val

    def visit_Call(self, node):
        raise Exception("This node should have been replaced")

    def visit_ObjectCallNode(self, node):
        args_tuple = self.visitlist(node.args)
        kwargs_dict = self.visit(node.kwargs)
        function = self.visit(node.function)
        largs = [function, args_tuple, kwargs_dict]
        PyObject_Call = self.function_cache.function_by_name('PyObject_Call')
        return self.builder.call(PyObject_Call, largs)

    def visit_NativeCallNode(self, node):
        # TODO: Refcounts + error check
        largs = self.visitlist(node.args)
        return self.builder.call(node.llvm_func, largs)

    def visit_TempNode(self, node):
        rhs = self.visit(node.node)
        lhs = self.builder.alloca(llvm_types._pyobject_head_struct_p)
        self.generate_assign(rhs, lhs)
        node.llvm_temp = lhs
        return lhs

class DisposalVisitor(visitors.NumbaVisitor):
    # TODO: handle errors, check for NULL before calling DECREF

    def __init__(self, context, func, ast, builder):
        super(DisposalVisitor, self).__init__(context, func, ast)
        self.builder = builder

    def visit_TempNode(self, node):
        self.visit(node.node)
        lfunc = self.function_cache.function_by_name('Py_DecRef')
        self.builder.call(lfunc, node.llvm_temp)
