import ast
import ctypes

import llvm
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le

from .llvm_types import    _int32, _intp, _LLVMCaster
from .multiarray_api import MultiarrayAPI # not used
from .symtab import Variable
from . import _numba_types as _types
from ._numba_types import BuiltinType

from numba import *
from . import visitors, nodes, llvm_types, utils
from .minivect import minitypes, llvm_codegen
from numba import ndarray_helpers, translate, error, extension_types
from numba._numba_types import is_obj, promote_closest
from numba.utils import dump

import logging
logger = logging.getLogger(__name__)

_int32_zero = lc.Constant.int(_int32, 0)

debug_conversion = False

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

def _create_methoddef(py_func, func_pointer):
    # struct PyMethodDef {
    #     const char  *ml_name;   /* The name of the built-in function/method */
    #     PyCFunction  ml_meth;   /* The C function that implements it */
    #     int      ml_flags;      /* Combination of METH_xxx flags, which mostly
    #                                describe the args expected by the C func */
    #     const char  *ml_doc;    /* The __doc__ attribute, or NULL */
    # };
    PyMethodDef = struct([('name', c_string_type),
                          ('method', void.pointer()),
                          ('flags', int_),
                          ('doc', c_string_type)])
    c_PyMethodDef = PyMethodDef.to_ctypes()

    PyCFunction_NewEx = ctypes.pythonapi.PyCFunction_NewEx
    PyCFunction_NewEx.argtypes = [ctypes.POINTER(c_PyMethodDef),
                                  ctypes.py_object,
                                  ctypes.c_void_p]
    PyCFunction_NewEx.restype = ctypes.py_object

    # It is paramount to put these into variables first, since every
    # access may return a new string object!
    name = py_func.__name__
    doc = py_func.__doc__
    py_func.live_objects.extend((name, doc))

    methoddef = c_PyMethodDef()
    methoddef.name = name
    methoddef.doc = doc
    methoddef.method = ctypes.c_void_p(func_pointer)
    methoddef.flags = 1 # METH_VARARGS

    return methoddef

def numbafunction_new(py_func, func_pointer, wrapped_lfunc_pointer,
                      wrapped_signature):
    methoddef = _create_methoddef(py_func, func_pointer)

    # Create PyCFunctionObject, pass in the methoddef struct as the m_self
    # attribute
    #methoddef_p = ctypes.byref(methoddef)
    #NULL = ctypes.c_void_p()
    #result = PyCFunction_NewEx(methoddef_p, methoddef, NULL)
    #return result
    wrapper = extension_types.create_function(
            methoddef, py_func, wrapped_lfunc_pointer, wrapped_signature)
    return methoddef, wrapper

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

    def __init__(self, opt=3):
        self._initialize(opt=opt)

    @classmethod
    def _initialize(cls, opt=3):
        if not cls._mods: # no modules yet
            # Create default module
            default_mod = cls._init_module(cls._DEFAULT_MODULE, opt=opt)

            # Create execution engine
            # NOTE: EE owns all registered modules
            cls._ee = le.ExecutionEngine.new(default_mod)

    @classmethod
    def _init_module(cls, name, opt=3):
        '''
        Initialize a module with the given name;
        Prepare pass managers for it.
        '''
        mod = lc.Module.new(name)
        cls._mods[name] = mod
        
        pmb = lp.PassManagerBuilder.new()
        pmb.opt_level = opt 
        fpm = lp.FunctionPassManager.new(mod)
        pmb.populate(fpm)
        
        cls._fpass[mod] = fpm

        return mod

    def create_module(self, name, opt=3):
        '''
        Create a llvm Module and add it to the execution engine.

        NOTE: Will we ever need this?
        '''
        mod = self._init_module(name, opt)
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
            #fpm.initialize()  # not necessary

        return fpm

    def get_module(self, name):
        return self._mods[name]

    def get_execution_engine(self):
        return self._ee

class ComplexSupportMixin(object):
    "Support for complex numbers"

    def _generate_complex_op(self, op, arg1, arg2):
        (r1, i1), (r2, i2) = self._extract(arg1), self._extract(arg2)
        real, imag = op(r1, i1, r2, i2)
        return self._create_complex(real, imag)

    def _extract(self, value):
        "Extract the real and imaginary parts of the complex value"
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1))

    def _create_complex(self, real, imag):
        assert real.type == imag.type
        complex = lc.Constant.undef(llvm.core.Type.struct([real.type, real.type]))
        complex = self.builder.insert_value(complex, real, 0)
        complex = self.builder.insert_value(complex, imag, 1)
        return complex

    def _promote_complex(self, src_type, dst_type, value):
        "Promote a complex value to value with a larger or smaller complex type"
        real, imag = self._extract(value)

        if dst_type.is_complex:
            dst_type = dst_type.base_type
        dst_ltype = dst_type.to_llvm(self.context)

        real = self.caster.cast(real, dst_ltype)
        imag = self.caster.cast(imag, dst_ltype)
        return self._create_complex(real, imag)

    def _complex_add(self, arg1r, arg1i, arg2r, arg2i):
        return (self.builder.fadd(arg1r, arg2r),
                self.builder.fadd(arg1i, arg2i))

    def _complex_sub(self, arg1r, arg1i, arg2r, arg2i):
        return (self.builder.fsub(arg1r, arg2r),
                self.builder.fsub(arg1i, arg2i))

    def _complex_mul(self, arg1r, arg1i, arg2r, arg2i):
        return (self.builder.fsub(self.builder.fmul(arg1r, arg2r),
                                  self.builder.fmul(arg1i, arg2i)),
                self.builder.fadd(self.builder.fmul(arg1i, arg2r),
                                  self.builder.fmul(arg1r, arg2i)))

    def _complex_div(self, arg1r, arg1i, arg2r, arg2i):
        divisor = self.builder.fadd(self.builder.fmul(arg2r, arg2r),
                                    self.builder.fmul(arg2i, arg2i))
        return (self.builder.fdiv(
                        self.builder.fadd(self.builder.fmul(arg1r, arg2r),
                                          self.builder.fmul(arg1i, arg2i)),
                        divisor),
                self.builder.fdiv(
                        self.builder.fsub(self.builder.fmul(arg1i, arg2r),
                                          self.builder.fmul(arg1r, arg2i)),
                        divisor))

    def _complex_floordiv(self, arg1r, arg1i, arg2r, arg2i):
        real, imag = self._complex_div(arg1r, arg1i, arg2r, arg2i)
        long_type = long_.to_llvm(self.context)
        real = self.caster.cast(real, long_type)
        imag = self.caster.cast(imag, long_type)
        real = self.caster.cast(real, arg1r.type)
        imag = self.caster.cast(imag, arg1r.type)
        return real, imag


class RefcountingMixin(object):

    def decref(self, value, func='Py_DecRef'):
        "Py_DECREF a value"
        assert not self.nopython
        object_ltype = object_.to_llvm(self.context)
        sig, py_decref = self.function_cache.function_by_name(func)
        b = self.builder
        return b.call(py_decref, [b.bitcast(value, object_ltype)])

    def incref(self, value):
        "Py_INCREF a value"
        assert not self.nopython
        return self.decref(value, func='Py_IncRef')

    def xdecref_temp(self, temp, decref=None):
        "Py_XDECREF a temporary"
        assert not self.nopython
        decref = decref or self.decref

        def cleanup(b, bb_true, bb_endif):
            decref(b.load(temp))
            b.branch(bb_endif)

        self.object_coercer.check_err(self.builder.load(temp),
                                      callback=cleanup,
                                      cmp=llvm.core.ICMP_NE)

    def xincref_temp(self, temp):
        "Py_XINCREF a temporary"
        assert not self.nopython
        return self.xdecref_temp(temp, decref=self.incref)

    def xdecref_temp_cleanup(self, temp):
        "Cleanup a temp at the end of the function"
        
        assert not self.nopython
        bb = self.builder.basic_block

        self.builder.position_at_end(self.current_cleanup_bb)
        self.xdecref_temp(temp)
        self.current_cleanup_bb = self.builder.basic_block

        self.builder.position_at_end(bb)


class LLVMCodeGenerator(visitors.NumbaVisitor, ComplexSupportMixin,
                        RefcountingMixin, visitors.NoPythonContextMixin):
    """
    Translate a Python AST to LLVM. Each visit_* method should directly
    return an LLVM value.
    """

    multiarray_api = MultiarrayAPI()

    def __init__(self, context, func, ast, func_signature, symtab,
                 optimize=True, nopython=False,
                 llvm_module=None, llvm_ee=None,
                 refcount_args=True, **kwds):

        super(LLVMCodeGenerator, self).__init__(
                    context, func, ast, func_signature=func_signature,
                    nopython=nopython, symtab=symtab)

        self.func_name = kwds.get('func_name', func_signature.name
                                            or func.__name__)
        self.func_signature = func_signature
        self.blocks = {} # stores id => basic-block

        # code generation attributes
        self.mod = llvm_module or LLVMContextManager().get_default_module()
        self.ee = llvm_ee or LLVMContextManager().get_execution_engine()

        self.refcount_args = refcount_args

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

    def _allocate_arg_local(self, name, argtype, larg):
        """
        Allocate a local variable on the stack.
        """
        stackspace = self.alloca(argtype)
        stackspace.name = name

        if (minitypes.pass_by_ref(argtype) and
                self.func_signature.struct_by_reference):
            larg = self.builder.load(larg)

        self.builder.store(larg, stackspace)
        return stackspace

    def _init_args(self):
        for larg, argname, argtype in zip(self.lfunc.args, self.argnames,
                                          self.func_signature.args):
            larg.name = argname
            variable = self.symtab[argname]

            if not variable.need_arg_copy:
                assert not is_obj(argtype)
                variable.lvalue = larg
                continue

            # Store away arguments in locals
            stackspace = self._allocate_arg_local(argname, argtype, larg)
            variable.lvalue = stackspace

            # TODO: incref objects in structs
            if not self.nopython:
                if is_obj(variable.type) and self.refcount_args:
                    self.incref(self.builder.load(stackspace))

    def _allocate_locals(self):
        for name, var in self.symtab.items():
            # FIXME: 'None' should be handled as a special case (probably).
            if name not in self.argnames and var.is_local:
                # Not argument and not builtin type.
                # Allocate storage for all variables.
                name = 'var_%s' % var.name
                if is_obj(var.type):
                    stackspace = self._null_obj_temp(name, type=var.ltype)
                else:
                    stackspace = self.builder.alloca(var.ltype, name=name)

                if var.type.is_struct:
                    # TODO: memset struct to 0
                    pass

                var.lvalue = stackspace

    def setup_func(self):
        self.lfunc_type = self.to_llvm(self.func_signature)

        self.lfunc = self.mod.add_function(self.lfunc_type, self.func_name)
        assert self.func_name == self.lfunc.name, \
               "Redefinition of function %s" % self.func_name
        
        # Add entry block for alloca.
        entry = self.append_basic_block('entry')
        self.builder = lc.Builder.new(entry)
        self.caster = _LLVMCaster(self.builder)
        self.object_coercer = ObjectCoercer(self)
        self.multiarray_api.set_PyArray_API(self.mod)

        self._init_args()
        self._allocate_locals()

        # TODO: Put current function into symbol table for recursive call
        self.setup_return()

        self.in_loop = 0
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
        # Try to find the function of the specified name in the current module.
        # If it is found, we can return immediately.
        # Otherwise, continue to translate.
        try:
            self.lfunc = self.mod.get_function_named(self.func_name)
        except llvm.LLVMException:
            pass
        else:
            assert not self.lfunc.is_declaration
            return # we are done, escape now.
        try:
            self.setup_func()

            if isinstance(self.ast, ast.FunctionDef):
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
                self.visit(self.ast)

            if not self.is_block_terminated():
                # self.builder.ret_void()
                self.builder.branch(self.cleanup_label)

            self.terminate_cleanup_blocks()

            # Done code generation
            del self.builder  # release the builder to make GC happy

            logger.debug("ast translated function: %s" % self.lfunc)
            # Verify code generation
            self.mod.verify()  # only Module level verification checks everything.
            if self.optimize:
                fp = LLVMContextManager().get_function_pass_manager(self.mod)
                fp.run(self.lfunc)
        except:
            # Delete the function to prevent an invalid function from living in the module
            self.lfunc.delete()
            raise

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
            return prototype(self.lfunc_pointer)
        else:
            return prototype(self.func)

    def _build_wrapper_function_ast(self, fake_pyfunc):
        wrapper = nodes.FunctionWrapperNode(self.lfunc,
                                            self.func_signature,
                                            self.func,
                                            fake_pyfunc)
        wrapper.pipeline = self.ast.pipeline
        wrapper.cellvars = []
        return wrapper

    def build_wrapper_function(self, get_lfunc=False):
        # PyObject *(*)(PyObject *self, PyObject *args)
        def func(self, args):
            pass
        func.live_objects = self.func.live_objects

        # Create wrapper code generator and wrapper AST
        func.__name__ = '__numba_wrapper_%s' % self.func_name
        signature = minitypes.FunctionType(return_type=object_,
                                           args=[void.pointer(), object_])
        symtab = dict(self=Variable(object_, is_local=True),
                      args=Variable(object_, is_local=True))
        wrapper_call = self._build_wrapper_function_ast(func)
        error_return = ast.Return(nodes.CoercionNode(nodes.NULL_obj,
                                                     object_))
        wrapper_call.error_return = error_return
        t = LLVMCodeGenerator(self.context, func, wrapper_call, signature,
                              symtab, llvm_module=self.mod, llvm_ee=self.ee,
                              refcount_args=False)
        t.translate()

        func.live_objects.append(t.lfunc)

        # Return a PyCFunctionObject holding the wrapper
        func_pointer = t.lfunc_pointer
        methoddef, wrapper = numbafunction_new(self.func, func_pointer,
                                   self.lfunc_pointer, self.func_signature)

        if get_lfunc:
            return wrapper, t.lfunc, methoddef

        return wrapper

    def insert_closure_scope_arg(self, args, node):
        """
        Retrieve the closure from the NumbaFunction passed in as m_self
        """
        closure_scope_type = node.signature.args[0]
        offset = extension_types.numbafunc_closure_field_offset
        closure = nodes.LLVMValueRefNode(void.pointer(), self.lfunc.args[0])
        closure = nodes.CoercionNode(closure, char.pointer())
        closure = nodes.pointer_add(closure, nodes.const(offset, size_t))
        closure = nodes.CoercionNode(closure, closure_scope_type.pointer())
        closure = nodes.DereferenceNode(closure)
        args.insert(0, closure)

    def visit_FunctionWrapperNode(self, node):
        from numba import ast_type_inference, pipeline

        args_tuple = self.lfunc.args[1]
        arg_types = [object_] * len(node.signature.args)

        if self.is_closure(node.signature):
            # Wrapper expects closure scope as first argument, which we
            # have as the m_self attribute
            arg_types.pop()

        # Unpack tuple into arguments
        types, lstr = self.object_coercer.lstr(arg_types)
        #if debug_conversion:
        #    self.puts("args tuple:")
        #    nodes.print_llvm(self, object_, args_tuple)

        largs = self.object_coercer.parse_tuple(lstr, args_tuple, arg_types)

        # Call wrapped function
        args = [nodes.LLVMValueRefNode(arg_type, larg)
                    for arg_type, larg in zip(arg_types, largs)]

        if self.is_closure(node.signature):
            # Insert m_self as scope argument type
            logger.debug("Closure:")
            self.insert_closure_scope_arg(args, node)

        func_call = nodes.NativeCallNode(node.signature, args,
                                         node.wrapped_function)

        if not is_obj(node.signature.return_type):
            # Check for error using PyErr_Occurred()
            # TODO: make this an option in CheckErrorNode
            check_err = nodes.CheckErrorNode(
                    nodes.ptrtoint(self.function_cache.call('PyErr_Occurred')),
                    goodval=nodes.ptrtoint(nodes.NULL))
            func_call = func_call.cloneable
            func_call = nodes.ExpressionNode(stmts=[func_call, check_err],
                                             expr=func_call.clone)

        # Coerce and return result
        if node.signature.return_type.is_void:
            node.body = func_call
            result_node = nodes.ObjectInjectNode(None)
        else:
            node.body = None
            result_node = func_call

        # self.puts("calling wrapped function %r" % node.orig_py_func.__name__)
        node.return_result = ast.Return(
                    value=nodes.CoercionNode(result_node, object_))

        # We need to specialize the return statement, since it may be a
        # non-trivial coercion (e.g. call a ctypes pointer type for pointer
        # return types, etc)
        pipeline_ = pipeline.Pipeline(self.context, node.fake_pyfunc,
                                      node, self.func_signature,
                                      order=['late_specializer'])
        sig, symtab, return_stmt_ast = pipeline_.run_pipeline()
        self.generic_visit(return_stmt_ast)

    @property
    def lfunc_pointer(self):
        return self.ee.get_pointer_to_function(self.lfunc)

    def _null_obj_temp(self, name, type=None):
        lhs = self.llvm_alloca(type or llvm_types._pyobject_head_struct_p,
                               name=name, change_bb=False)
        self.generate_assign(self.visit(nodes.NULL_obj), lhs)
        return lhs

    def puts(self, msg):
        const = nodes.ConstNode(msg, c_string_type)
        self.visit(self.function_cache.call('puts', const))

    def puts_llvm(self, llvm_string):
        const = nodes.LLVMValueRefNode(c_string_type, llvm_string)
        self.visit(self.function_cache.call('puts', const))

    def setup_return(self):
        # Assign to this value which will be returned
        self.is_void_return = \
                self.func_signature.actual_signature.return_type.is_void
        if self.func_signature.struct_by_reference:
            self.return_value = self.lfunc.args[-1]
        elif not self.is_void_return:
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

        # Decref local variables
        if not self.nopython:
            for name, var in self.symtab.iteritems():
                if var.is_local and is_obj(var.type):
                    if self.refcount_args or not name in self.argnames:
                        self.xdecref_temp(var.lvalue)

        if self.is_void_return:
            self.builder.ret_void()
        else:
            ret_type = self.func_signature.return_type
            self.builder.ret(self.builder.load(self.return_value))

    # __________________________________________________________________________

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Pass(self, node):
        pass

    def visit_ConstNode(self, node):
        return node.value(self)

    def visit_Attribute(self, node):
        result = self.visit(node.value)
        if node.value.type.is_complex:
            if node.attr == 'real':
                return self.builder.extract_value(result, 0)
            elif node.attr == 'imag':
                return self.builder.extract_value(result, 1)

        raise error.NumbaError("This node should have been replaced")

    def visit_StructAttribute(self, node):
        result = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            result = self.builder.extract_value(result, node.field_idx)
        else:
            result = self.builder.gep(
                result, [llvm_types.constant_int(0),
                         llvm_types.constant_int(node.field_idx)])

        return result

    def visit_Assign(self, node):
        target_node = node.targets[0]

        target = self.visit(target_node)
        value = self.visit(node.value)

        object = is_obj(target_node.type)
        self.generate_assign(value, target, decref=object, incref=object)
        if object:
            self.incref(value)

    def generate_assign(self, lvalue, ltarget, decref=False, incref=False):
        '''
        Generate assignment operation and automatically cast value to
        match the target type.
        '''
        if lvalue.type != ltarget.type.pointee:
            lvalue = self.caster.cast(lvalue, ltarget.type.pointee)

        if decref:
            # Py_XDECREF any previous object
            self.xdecref_temp(ltarget)

        self.builder.store(lvalue, ltarget)

    def visit_Num(self, node):
        if node.type.is_int:
            return self.generate_constant_int(node.n)
        elif node.type.is_float:
            return self.generate_constant_real(node.n)
        else:
            assert node.type.is_complex
            return self.generate_constant_complex(node.n)

    def visit_Name(self, node):
        if not node.variable.is_local:
            raise error.NumbaError(node, "global variables:", node.id)

        lvalue = self.symtab[node.id].lvalue
        return self._handle_ctx(node, lvalue)

    def _handle_ctx(self, node, lptr, name=''):
        if isinstance(node.ctx, ast.Load):
            return self.builder.load(lptr, name=name and 'load_' + name)
        else:
            return lptr

    def visit_For(self, node):
        if node.orelse:
            # FIXME
            raise error.NumbaError(node, 'Else in for-loop is not implemented.')

        if node.iter.type.is_range:
            self.generate_for_range(node, node.target, node.iter, node.body)
        else:
            raise error.NumbaError(node, "Looping over iterables")

    def visit_BoolOp(self, node):
        # NOTE: Can have >2 values
        assert len(node.values) >= 2
        assert isinstance(node.op, ast.And) or isinstance(node.op, ast.Or)

        count = len(node.values)

        if isinstance(node.op, ast.And):
            bb_true = self.append_basic_block('and.true')
            bb_false = self.append_basic_block('and.false')
            bb_next = [self.append_basic_block('and.rhs')
                       for i in range(count - 1)] + [bb_true]
            bb_done = self.append_basic_block('and.done')

            for i in range(count):
                value = self.visit(node.values[i])
                self.builder.cbranch(value, bb_next[i], bb_false)
                self.builder.position_at_end(bb_next[i])

            assert self.builder.basic_block is bb_true
            self.builder.branch(bb_done)

            self.builder.position_at_end(bb_false)
            self.builder.branch(bb_done)

            self.builder.position_at_end(bb_done)
        elif isinstance(node.op, ast.Or):
            bb_true = self.append_basic_block('or.true')
            bb_false = self.append_basic_block('or.false')
            bb_next = [self.append_basic_block('or.rhs')
                       for i in range(count - 1)] + [bb_false]
            bb_done = self.append_basic_block('or.done')

            for i in range(count):
                value = self.visit(node.values[i])
                self.builder.cbranch(value, bb_true, bb_next[i])
                self.builder.position_at_end(bb_next[i])

            assert self.builder.basic_block is bb_false
            self.builder.branch(bb_done)

            self.builder.position_at_end(bb_true)
            self.builder.branch(bb_done)

            self.builder.position_at_end(bb_done)
        else:
            raise Exception("internal erorr")

        booltype = lc.Type.int(1)
        phi = self.builder.phi(booltype)
        phi.add_incoming(lc.Constant.int(booltype, 1), bb_true)
        phi.add_incoming(lc.Constant.int(booltype, 0), bb_false)

        return phi

    def visit_UnaryOp(self, node):
        operand_type = node.operand.type
        operand = self.visit(node.operand)
        operand_ltype = operand.type
        op = node.op
        if isinstance(op, ast.Not) and (operand_type.is_bool or
                                        operand_type.is_int_like):
            bb_false = self.builder.basic_block
            bb_true = self.append_basic_block('not.true')
            bb_done = self.append_basic_block('not.done')
            self.builder.cbranch(
                self.builder.icmp(lc.ICMP_NE, operand,
                                  lc.Constant.null(operand_ltype)),
                bb_true, bb_done)
            self.builder.position_at_end(bb_true)
            self.builder.branch(bb_done)
            self.builder.position_at_end(bb_done)
            phi = self.builder.phi(operand_ltype)
            phi.add_incoming(lc.Constant.int(operand_ltype, 1), bb_false)
            phi.add_incoming(lc.Constant.int(operand_ltype, 0), bb_true)
            return phi
        elif isinstance(op, ast.USub) and operand_type.is_numeric:
            if operand_type.is_float:
                return self.builder.fsub(lc.Constant.null(operand_ltype),
                                         operand)
            elif operand_type.is_int_like and operand_type.signed:
                return self.builder.sub(lc.Constant.null(operand_ltype),
                                        operand)
        elif isinstance(op, ast.UAdd) and operand_type.is_numeric:
            return operand
        elif isinstance(op, ast.Invert) and operand_type.is_int_like:
            return self.builder.xor(lc.Constant.int(operand_ltype, -1), operand)
        raise error.NumbaError(node, "Unary operator %s" % node.op)

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
            if not lhs.type.signed:
                error.NumbaError(
                        node, 'Unsigned comparison has not been implemented')
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
        #bb = self.lfunc.append_basic_block('%s_%d'%(name, idx))
        bb = self.lfunc.append_basic_block(name)
        self.blocks[idx] = bb
        return bb

    def visit_Return(self, node):
        if node.value is not None:
            value_type = node.value.type
            rettype = self.func_signature.return_type

            retval = self.visit(node.value)
            if is_obj(rettype) or rettype.is_pointer:
                retval = self.builder.bitcast(retval,
                                              self.return_value.type.pointee)

            if not retval.type == self.return_value.type.pointee:
                dump(node)
                logger.debug('%s != %s (in node %s)' % (
                        self.return_value.type.pointee, retval.type,
                        utils.pformat_ast(node)))
                raise error.NumbaError(
                    node, 'Expected %s type in return, got %s!' %
                    (self.return_value.type.pointee, retval.type))

            self.builder.store(retval, self.return_value)

            ret_type = self.func_signature.return_type
            if is_obj(rettype):
                self.xincref_temp(self.return_value)

        if not self.is_block_terminated():
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
        self.in_loop += 1

    def teardown_loop(self):
        self.loop_beginnings.pop()
        self.loop_exits.pop()
        self.in_loop -= 1

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

    _opnames = {
        ast.Mult: 'mul',
    }

    def opname(self, op):
        if op in self._opnames:
            return self._opnames[op]
        else:
            return op.__name__.lower()

    def _handle_mod(self, node, lhs, rhs):
        _, func = self.function_cache.function_by_name(
            'PyModulo', arg_types = (node.type, node.type),
            return_type = node.type)
        return self.builder.call(func, (lhs, rhs))

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = type(node.op)

        if (node.type.is_int or node.type.is_float) and op in self._binops:
            llvm_method_name = self._binops[op][node.type.is_int]
            if node.type.is_int:
                llvm_method_name = llvm_method_name[node.type.signed]
            meth = getattr(self.builder, llvm_method_name)
            if not lhs.type == rhs.type:
                print ast.dump(node)
                assert False
            result = meth(lhs, rhs)
        elif (node.type.is_int or node.type.is_float) and op == ast.Mod:
            return self._handle_mod(node, lhs, rhs)
        elif node.type.is_complex:
            opname = self.opname(op)
            if opname in ('add', 'sub', 'mul', 'div', 'floordiv'):
                m = getattr(self, '_complex_' + opname)
                result = self._generate_complex_op(m, lhs, rhs)
            else:
                raise error.NumbaError("Unsupported binary operation "
                                       "for complex numbers: %s" % opname)
        else:
            logging.debug('Unrecognized node type "%s"' % node.type)
            logging.debug(ast.dump(node))
            raise error.NumbaError(
                    node, "Binary operations %s on values typed %s and %s "
                          "not (yet) supported)" % (self.opname(op),
                                                    node.left.type,
                                                    node.right.type))

        return result

    def visit_CoercionNode(self, node, val=None):
        if val is None:
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
        elif dst_type.is_pointer and node_type.is_pointer:
            val = self.builder.bitcast(val, ldst_type)
        elif dst_type.is_complex and node_type.is_complex:
            val = self._promote_complex(node_type, dst_type, val)
        elif dst_type.is_complex and node_type.is_numeric:
            ldst_base_type = dst_type.base_type.to_llvm(self.context)
            real = val
            if node_type != dst_type.base_type:
                real = self.caster.cast(real, ldst_base_type)
            imag = llvm.core.Constant.real(ldst_base_type, 0.0)
            val = self._create_complex(real, imag)
        else:
            val = self.caster.cast(val, node.dst_type.to_llvm(self.context))

        if debug_conversion:
            self.puts("Coercing %s to %s" % (node_type, dst_type))

        return val

    def visit_CoerceToObject(self, node):
        from_type = node.node.type
        result = self.visit(node.node)
        if not is_obj(from_type):
            result = self.object_coercer.convert_single(from_type, result,
                                                        name=node.name)
        return result

    def visit_CoerceToNative(self, node):
        assert node.node.type.is_tuple
        val = self.visit(node.node)
        return self.object_coercer.to_native(node.dst_type, val,
                                             name=node.name)

    def visit_DereferenceNode(self, node):
        result = self.visit(node.pointer)
        return self.builder.load(result)

    def visit_PointerFromObject(self, node):
        return self.visit(node.node)

    def visit_ComplexNode(self, node):
        real = self.visit(node.real)
        imag = self.visit(node.imag)
        return self._create_complex(real, imag)

    def visit_Subscript(self, node):
        value_type = node.value.type
        if not (value_type.is_carray or value_type.is_c_string or
                    value_type.is_sized_pointer):
            raise error.InternalError(node, "Unsupported type:", node.value.type)

        value = self.visit(node.value)
        index = self.visit(node.slice)
        lptr = self.builder.gep(value, [index])
        if node.slice.type.is_int:
            lptr = self._handle_ctx(node, lptr)

        return lptr

    def visit_DataPointerNode(self, node):
        assert node.node.type.is_array
        lvalue = self.visit(node.node)
        lindices = self.visit(node.slice)
        lptr = node.subscript(self, lvalue, lindices)
        return self._handle_ctx(node, lptr)

    #def visit_Index(self, node):
    #    return self.visit(node.value)

    def visit_ExtSlice(self, node):
        return self.visitlist(node.dims)

    def visit_Call(self, node):
        raise error.InternalError(node, "This node should have been replaced")

    def visit_List(self, node):
        types = [n.type for n in node.elts]
        largs = self.visitlist(node.elts)
        return self.object_coercer.build_list(types, largs)

    def visit_Tuple(self, node):
        raise error.InternalError(node, "This node should have been replaced")

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

    def visit_NoneNode(self, node):
        try:
            self.mod.add_global_variable(object_.to_llvm(self.context),
                                         "Py_None")
        except llvm.LLVMException:
            pass

        return self.mod.get_global_variable_named("Py_None")

    def visit_NativeCallNode(self, node, largs=None):
        if largs is None:
            largs = self.visitlist(node.args)

        return_value = llvm_codegen.handle_struct_passing(
                            self.builder, self.alloca, largs, node.signature)

        if hasattr(node.llvm_func, 'module') and node.llvm_func.module != self.mod:
            lfunc = self.mod.get_or_insert_function(node.llvm_func.type.pointee,
                                                    node.llvm_func.name)
        else:
            lfunc = node.llvm_func

        result = self.builder.call(lfunc, largs, name=node.name)

        if node.signature.struct_by_reference:
            if minitypes.pass_by_ref(node.signature.return_type):
                result = self.builder.load(return_value)

        return result

    def visit_NativeFunctionCallNode(self, node):
        lfunc = self.visit(node.function)
        node.llvm_func = lfunc
        return self.visit_NativeCallNode(node)

    def visit_LLMacroNode (self, node):
        return node.macro(self.function_cache, self.builder,
                          *self.visitlist(node.args))

    def visit_LLVMExternalFunctionNode(self, node):
        lfunc_type = node.signature.to_llvm(self.context)
        return self.mod.get_or_insert_function(lfunc_type, node.fname)

    def visit_LLVMIntrinsicNode(self, node):
        intr = getattr(llvm.core, 'INTR_' + node.func_name)
        largs = self.visitlist(node.args)
        node.llvm_func = llvm.core.Function.intrinsic(
                        self.mod, intr, [largs[0].type])
        return self.visit_NativeCallNode(node, largs=largs)

    def visit_MathCallNode(self, node):
        try:
            lfunc = self.mod.get_function_named(node.name)
        except llvm.LLVMException:
            lfunc_type = node.signature.to_llvm(self.context)
            lfunc = self.mod.add_function(lfunc_type, node.name)

        node.llvm_func = lfunc
        return self.visit_NativeCallNode(node)

    def visit_CTypesCallNode(self, node):
        node.llvm_func = self.visit(node.function)
        return self.visit_NativeCallNode(node)

    def visit_ClosureCallNode(self, node):
        lfunc = node.closure_type.closure.lfunc
        assert lfunc is not None
        node.llvm_func = lfunc
        return self.visit_NativeCallNode(node)

    def visit_ComplexConjugateNode(self, node):
        lcomplex = self.visit(node.complex_node)

        elem_ltyp = node.type.base_type.to_llvm(self.context)
        zero = llvm.core.Constant.real(elem_ltyp, 0)
        imag = self.builder.extract_value(lcomplex, 1)
        new_imag_lval = self.builder.fsub(zero, imag)

        assert hasattr(self.builder, 'insert_value'), (
            "llvm-py support for LLVMBuildInsertValue() required to build "
            "code for complex conjugates.")

        return self.builder.insert_value(lcomplex, new_imag_lval, 1)

    def visit_MultiArrayAPINode(self, node):
        meth = getattr(self.multiarray_api, 'load_' + node.func_name)
        lfunc = meth(self.mod, self.builder)
        lsignature = node.signature.pointer().to_llvm(self.context)
        node.llvm_func = self.builder.bitcast(lfunc, lsignature)
        result = self.visit_NativeCallNode(node)
        return result

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
        lhs = self._null_obj_temp(name)
        node.llvm_temp = lhs

        # Assign value
        self.builder.position_at_end(bb)
        rhs = self.visit(node.node)
        self.generate_assign(rhs, lhs, decref=self.in_loop)

        # goto error if NULL
        # self.puts("checking error... %s" % error.format_pos(node))
        self.object_coercer.check_err(rhs)
        # self.puts("all good at %s" % error.format_pos(node))

        if node.incref:
            self.incref(self.builder.load(lhs))

        # Generate Py_XDECREF(temp) at end-of-function cleanup path
        self.xdecref_temp_cleanup(lhs)
        result = self.builder.load(lhs, name=name + '_load')

        if not node.type == object_:
            dst_type = node.type.to_llvm(self.context)
            result = self.builder.bitcast(result, dst_type)

        return result

    def visit_PropagateNode(self, node):
        # self.puts("ERROR! %s" % error.format_pos(node))
        self.builder.branch(self.error_label)

    def visit_ObjectTempRefNode(self, node):
        return node.obj_temp_node.llvm_temp

    def visit_LLVMValueRefNode(self, node):
        return node.llvm_value

    def visit_CloneNode(self, node):
        return node.llvm_value

    def visit_CloneableNode(self, node):
        llvm_value = self.visit(node.node)
        for clone_node in node.clone_nodes:
            clone_node.llvm_value = llvm_value

        return llvm_value

    def visit_ArrayAttributeNode(self, node):
        array = self.visit(node.array)
        acc = ndarray_helpers.PyArrayAccessor(self.builder, array)

        attr_name = node.attr_name
        if attr_name == 'shape':
            attr_name = 'dimensions'

        result = getattr(acc, attr_name)
        ltype = node.type.to_llvm(self.context)
        if node.attr_name == 'data':
            result = self.builder.bitcast(result, ltype)

        return result

    def visit_ShapeAttributeNode(self, node):
        # hurgh, no dispatch on superclasses?
        return self.visit_ArrayAttributeNode(node)

    def visit_IncrefNode(self, node):
        obj = self.visit(node.value)
        self.incref(obj)
        return obj

    def visit_DecrefNode(self, node):
        obj = self.visit(node.value)
        self.decref(obj)
        return obj

    def visit_ExpressionNode(self, node):
        self.visitlist(node.stmts)
        return self.visit(node.expr)


def llvm_alloca(lfunc, builder, ltype, name='', change_bb=True):
    "Use alloca only at the entry bock of the function"
    if change_bb:
        bb = builder.basic_block
    builder.position_at_beginning(lfunc.get_entry_basic_block())
    lstackvar = builder.alloca(ltype, name)
    if change_bb:
        builder.position_at_end(bb)
    return lstackvar

def if_badval(translator, llvm_result, badval, callback,
              cmp=llvm.core.ICMP_EQ, name='cleanup'):
    # Use llvm_cbuilder :(
    b = translator.builder

    bb_true = translator.append_basic_block('%s.if.true' % name)
    bb_endif = translator.append_basic_block('%s.if.end' % name)

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
        char.pointer() : "s",
    }

    def __init__(self, translator):
        self.context = translator.context
        self.translator = translator
        self.builder = translator.builder
        sig, self.py_buildvalue = translator.function_cache.function_by_name(
                                                              'Py_BuildValue')
        sig, self.pyarg_parsetuple = translator.function_cache.function_by_name(
                                                              'PyArg_ParseTuple')
        sig, self.pyerr_clear = translator.function_cache.function_by_name(
                                                            'PyErr_Clear')
        self.function_cache = translator.function_cache
        self.NULL = self.translator.visit(nodes.NULL_obj)

    def check_err(self, llvm_result, callback=None, cmp=llvm.core.ICMP_EQ):
        """
        Check for errors. If the result is NULL, and error should have been set
        Jumps to translator.error_label if an exception occurred.
        """
        assert llvm_result.type.kind == llvm.core.TYPE_POINTER
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

    def _create_llvm_string(self, str):
        return self.translator.visit(nodes.ConstNode(str, c_string_type))

    def lstr(self, types, fmt=None):
        "Get an llvm format string for the given types"
        typestrs = []
        result_types = []
        for type in types:
            if is_obj(type):
                type = object_
            elif type.is_int:
                type = promote_closest(self.context, type,
                                       minitypes.native_integral)

            result_types.append(type)
            typestrs.append(self.type_to_buildvalue_str[type])

        str = "".join(typestrs)
        if fmt is not None:
            str = fmt % str

        if debug_conversion:
            self.translator.puts("fmt: %s" % str)

        result = self._create_llvm_string(str)
        return result_types, result

    def buildvalue(self, types, *largs, **kwds):
        # The caller should check for errors using check_err or by wrapping
        # its node in an ObjectTempNode
        name = kwds.get('name', '')
        fmt = kwds.get('fmt', None)
        types, lstr = self.lstr(types, fmt)
        largs = (lstr,) + largs

        if debug_conversion:
            self.translator.puts("building... %s" % name)
        # func_type = object_(*types).pointer()
        # py_buildvalue = self.builder.bitcast(
        #         self.py_buildvalue, func_type.to_llvm(self.context))
        py_buildvalue = self.py_buildvalue
        result = self.builder.call(py_buildvalue, largs, name=name)

        if debug_conversion:
            self.translator.puts("done building... %s" % name)

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

    def convert_single_struct(self, llvm_result, type):
        types = []
        largs = []
        for i, (field_name, field_type) in enumerate(type.fields):
            types.extend((c_string_type, field_type))
            largs.append(self._create_llvm_string(field_name))
            struct_attr = self.builder.extract_value(llvm_result, i)
            largs.append(struct_attr)

        return self.buildvalue(types, *largs, name='struct', fmt="{%s}")

    def convert_single(self, type, llvm_result, name=''):
        "Generate code to convert an LLVM value to a Python object"
        llvm_result, type = self.npy_intp_to_py_ssize_t(llvm_result, type)
        if type.is_struct:
            return self.convert_single_struct(llvm_result, type)
        elif type.is_complex:
            # We have a Py_complex value, construct a Py_complex * temporary
            new_result = llvm_alloca(self.translator.lfunc, self.builder,
                                     llvm_result.type, name='complex_temp')
            self.builder.store(llvm_result, new_result)
            llvm_result = new_result

        return self.buildvalue([type], llvm_result, name=name)

    def build_tuple(self, types, llvm_values):
        "Build a tuple from a bunch of LLVM values"
        assert len(types) == len(llvm_values)
        return self.buildvalue(lstr, *llvm_values, name='tuple', fmt="(%s)")

    def build_list(self, types, llvm_values):
        "Build a tuple from a bunch of LLVM values"
        assert len(types) == len(llvm_values)
        return self.buildvalue(types, *llvm_values, name='list',  fmt="[%s]")

    def build_dict(self, key_types, value_types, llvm_keys, llvm_values):
        "Build a dict from a bunch of LLVM values"
        types = []
        largs = []
        for k, v, llvm_key, llvm_value in zip(key_types, value_types,
                                              llvm_keys, llvm_values):
            types.append(k)
            types.append(v)
            largs.append(llvm_key)
            largs.append(llvm_value)

        return self.buildvalue(types, *largs, name='dict', fmt="{%s}")

    def parse_tuple(self, lstr, llvm_tuple, types, name=''):
        "Unpack a Python tuple into typed llvm variables"
        lresults = []
        for i, type in enumerate(types):
            var = llvm_alloca(self.translator.lfunc, self.builder,
                              type.to_llvm(self.context),
                              name=name and "%s%d" % (name, i))
            lresults.append(var)

        largs = [llvm_tuple, lstr] + lresults

        if debug_conversion:
            self.translator.puts("parsing tuple... %s" % (types,))
            nodes.print_llvm(self.translator, object_, llvm_tuple)

        parse_result = self.builder.call(self.pyarg_parsetuple, largs)
        self.check_err_int(parse_result, 0)

        # Some conversion functions don't reset the exception state...
        # self.builder.call(self.pyerr_clear, [])

        if debug_conversion:
            self.translator.puts("successfully parsed tuple...")

        return map(self.builder.load, lresults)

    def to_native(self, type, llvm_tuple, name=''):
        "Generate code to convert a Python object to an LLVM value"
        types, lstr = self.lstr([type])
        lresult, = self.parse_tuple(lstr, llvm_tuple, [type], name=name)
        return lresult
