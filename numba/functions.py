import ast

from numba import *
from . import naming
from .minivect import minitypes
import numba.ast_translate as translate
import numba.ast_type_inference as type_inference
from numba import nodes

import meta.decompiler
import llvm.core

def is_numba_func(func):
    return getattr(func, '_is_numba_func', False)

def fix_ast_lineno(tree):
    # NOTE: A hack to fix assertion error in debug mode due to bad lineno.
    #       Lineno must increase monotonically for co_lnotab,
    #       the "line number table" to work correctly.
    #       This script just set all lineno to 1 and col_offset = to 0.
    #       This makes it impossible to do traceback, but it is not possible
    #       anyway since we are dynamically changing the source code.
    for node in ast.walk(tree):
        # only ast.expr and ast.stmt and their subclass has lineno and col_offset.
        # if isinstance(node,  ast.expr) or isinstance(node, ast.stmt):
        node.lineno = 1
        node.col_offset = 0

    return tree

def _get_ast(func):
    return meta.decompiler.decompile_func(func)

def _infer_types(context, func, restype=None, argtypes=None):
    ast = _get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
    return type_inference.run_pipeline(context, func, ast, func_signature)


def _compile(context, func, restype=None, argtypes=None, **kwds):
    """
    Compile a numba annotated function.

        - decompile function into a Python ast
        - run type inference using the given input types
        - compile the function to LLVM
    """
    func_signature, symtab, ast = _infer_types(context, func, restype, argtypes)
    func_name = naming.specialized_mangle(func.__name__, func_signature.args)

    t = translate.LLVMCodeGenerator(
        context, func, ast, func_signature=func_signature,
        func_name=func_name, symtab=symtab, **kwds)
    t.translate()

    return func_signature, t.lfunc, t.get_ctypes_func(kwds.get('llvm', True))

class FunctionCache(object):
    """
    Cache for compiler functions, declared external functions and constants.
    """
    def __init__(self, context):
        self.context = context
        self.module = context.llvm_context.get_default_module()

        # All numba-compiled functions
        # (py_func, arg_types) -> (signature, llvm_func, ctypes_func)
        self.compiled_functions = {}

        # All external functions: (py_func or name) -> (signature, llvm_func)
        # Only callable from numba-compiled functions
        self.external_functions = {}

        self.string_constants = {}

    def get_function(self, py_func, argtypes=None):
        result = None

        if argtypes is not None:
            result = self.compiled_functions.get((py_func, tuple(argtypes)))

        if result is None and py_func in self.external_functions:
            signature, lfunc = self.external_functions[py_func]
            result = signature, lfunc, None

        return result

    def compile_function(self, func, argtypes, restype=None, **kwds):
        """
        Compile a python function given the argument types. Compile only
        if not compiled already, and only if an annotated numba function
        (in the future, use heuristics to determine whether a function
        should be compiled or not).

        Returns a triplet of (signature, llvm_func, python_callable)
        `python_callable` may be the original function, or a ctypes callable
        if the function was compiled.
        """
        if func is not None:
            result = self.get_function(func, argtypes)
            if result is not None:
                return result

            if is_numba_func(func):
                func = getattr(func, '_numba_func', func)
                # numba function, compile
                func_signature, lfunc, ctypes_func = _compile(
                                self.context, func, restype, argtypes, **kwds)
                self.compiled_functions[func, tuple(func_signature.args)] = (
                                            func_signature, lfunc, ctypes_func)
                return func_signature, lfunc, ctypes_func

        # print func, getattr(func, '_is_numba_func', False)
        # create a signature taking N objects and returning an object
        signature = ofunc(argtypes=ofunc.arg_types * len(argtypes)).signature
        return signature, None, func

    def function_by_name(self, name):
        """
        Return the signature and LLVM function given a name. The function must
        either already be compiled, or it must be defined in this module.
        """
        if name in self.external_functions:
            return self.external_functions[name]
        else:
            declared_func = globals()[name]()
            lfunc = self.build_function(declared_func)
            return declared_func.signature, lfunc

    def call(self, name, *args, **kw):
        temp_name = kw.get('temp_name', '')
        function_cls = globals()[name]
        sig, lfunc = self.function_by_name(name)
        return nodes.NativeCallNode(sig, args, lfunc, name=temp_name)

    def build_function(self, external_function):
        """
        Build a function given it's signature information. See the
        `ExternalFunction` class.
        """
        try:
            lfunc = self.module.get_function_named(external_function.name)
        except llvm.LLVMException:
            func_type = minitypes.FunctionType(
                    return_type=external_function.return_type,
                    args=external_function.arg_types,
                    is_vararg=external_function.is_vararg)
            lfunc_type = func_type.to_llvm(self.context)
            lfunc = self.module.add_function(lfunc_type, external_function.name)

            if external_function.linkage == llvm.core.LINKAGE_INTERNAL:
                entry = lfunc.append_basic_block('entry')
                builder = lc.Builder.new(entry)
                lfunc.linkage = external_function.linkage
                external_function.implementation(
                                        self.module, builder, lfunc)

        return lfunc

    def get_string_constant(self, const_str):
        if (module, const_str) in self.string_constants:
            ret_val = self.string_constants[(module, const_str)]
        else:
            lconst_str = lc.Constant.stringz(const_str)
            ret_val = module.add_global_variable(lconst_str.type, "__STR_%d" %
                                                 (len(self.string_constants),))
            ret_val.initializer = lconst_str
            ret_val.linkage = lc.LINKAGE_INTERNAL
            self.string_constants[(module, const_str)] = ret_val

        return ret_val


class _LLVMModuleUtils(object):
    # TODO: rewrite print statements w/ native types to printf during type
    # TODO: analysis to PrintfNode. Use PrintfNode for debugging

    @classmethod
    def build_print_string_constant(cls, translator, out_val):
        # FIXME: This is just a hack to get things going.  There is a
        # corner case where formatting markup in the string can cause
        # undefined behavior, and for 100% correctness we'd have to
        # escape string formatting sequences.
        llvm_printf = cls.get_printf(translator.mod)
        return translator.builder.call(llvm_printf, [
            translator.builder.gep(cls.get_string_constant(translator.mod,
                                                           out_val),
                [_int32_zero, _int32_zero])])

    @classmethod
    def build_print_number(cls, translator, out_var):
        llvm_printf = cls.get_printf(translator.mod)
        if out_var.typ[0] == 'i':
            if int(out_var.typ[1:]) < 64:
                fmt = "%d"
                typ = "i32"
            else:
                fmt = "%ld"
                typ = "i64"
        elif out_var.typ[0] == 'f':
            if int(out_var.typ[1:]) < 64:
                fmt = "%f"
                typ = "f32"
            else:
                fmt = "%lf"
                typ = "f64"
        else:
            raise NotImplementedError("FIXME (type %r not supported in "
                                      "build_print_number())" % (out_var.typ,))
        return translator.builder.call(llvm_printf, [
            translator.builder.gep(
                cls.get_string_constant(translator.mod, fmt),
                [_int32_zero, _int32_zero]),
            out_var.llvm(typ, builder = translator.builder)])

    @classmethod
    def build_debugout(cls, translator, args):
        if translator.optimize:
            print("Warning: Optimization turned on, debug output code may "
                  "be optimized out.")
        res = cls.build_print_string_constant(translator, "debugout: ")
        for arg in args:
            arg_type = arg.typ
            if arg.typ is None:
                arg_type = type(arg.val)
            if isinstance(arg.val, str):
                res = cls.build_print_string_constant(translator, arg.val)
            elif typ_isa_number(arg_type):
                res = cls.build_print_number(translator, arg)
            else:
                raise NotImplementedError("Don't know how to output stuff of "
                                          "type %r at present." % arg_type)
        res = cls.build_print_string_constant(translator, "\n")
        return res, None

    @classmethod
    def build_len(cls, translator, args):
        if (len(args) == 1 and
            args[0].lvalue is not None and
            args[0].lvalue.type == _numpy_array):
            lfunc = None
            shape_ofs = _numpy_array_field_ofs['shape']
            res = translator.builder.load(
                translator.builder.load(
                    translator.builder.gep(args[0]._llvm, [
                        _int32_zero, lc.Constant.int(_int32, shape_ofs)])))
        else:
            return cls.build_object_len(translator, args)

        return res, None

    @classmethod
    def build_zeros_like(cls, translator, args):
        assert (len(args) == 1 and
                args[0]._llvm is not None and
                args[0]._llvm.type == _numpy_array), (
            "Expected Numpy array argument to numpy.zeros_like().")
        translator.init_multiarray()
        larr = args[0]._llvm
        largs = [translator.builder.load(
            translator.builder.gep(larr, [
                _int32_zero,
                lc.Constant.int(_int32,
                                _numpy_array_field_ofs[field_name])]))
                 for field_name in ('ndim', 'shape', 'descr')]
        largs.append(_int32_zero)
        lfunc = translator.ma_obj.load_PyArray_Zeros(translator.mod,
                                                     translator.builder)
        res = translator.builder.bitcast(
            translator.builder.call(
                cls.get_py_incref(translator.mod),
                [translator.builder.call(lfunc, largs)]),
            _numpy_array)
        if __debug__:
            print "build_zeros_like(): lfunc =", str(lfunc)
            print "build_zeros_like(): largs =", [str(arg) for arg in largs]
        return res, args[0].typ

    @classmethod
    def build_conj(cls, translator, args):
        print args
        assert ((len(args) == 1) and (args[0]._llvm is not None) and
                (args[0]._llvm.type in (_complex64, _complex128)))
        larg = args[0]._llvm
        elem_ltyp = _float if larg.type == _complex64 else _double
        new_imag_lval = translator.builder.fmul(
            lc.Constant.real(elem_ltyp, -1.),
            translator.builder.extract_value(larg, 1))
        assert hasattr(translator.builder, 'insert_value'), (
            "llvm-py support for LLVMBuildInsertValue() required to build "
            "code for complex conjugates.")
        res = translator.builder.insert_value(larg, new_imag_lval, 1)
        return res, None

#
### Define external and internal functions
#

class ExternalFunction(object):
    func_name = None
    arg_types = None
    return_type = None
    linkage = None
    is_vararg = False

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    @property
    def signature(self):
        return minitypes.FunctionType(return_type=self.return_type,
                                      args=self.arg_types,
                                      is_vararg=self.is_vararg)

    @property
    def name(self):
        if self.func_name is None:
            return type(self).__name__
        else:
            return self.func_name

    def implementation(self, module, builder, lfunc):
        return None

class InternalFunction(ExternalFunction):
    linkage = llvm.core.LINKAGE_INTERNAL

class ofunc(ExternalFunction):
    arg_types = [object_]
    return_type = object_

class printf(ExternalFunction):
    arg_types = [void.pointer()]
    return_type = int32
    is_vararg = True

class puts(ExternalFunction):
    arg_types = [c_string_type]
    return_type = int32

class Py_IncRef(ofunc):
    # TODO: rewrite calls to Py_IncRef/Py_DecRef to direct integer
    # TODO: increments/decrements
    return_type = void

class Py_DecRef(Py_IncRef):
    pass

class PyObject_Length(ofunc):
    return_type = Py_ssize_t

class PyObject_Call(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = object_

class PyObject_CallMethod(ExternalFunction):
    arg_types = [object_, c_string_type, c_string_type]
    return_type = object_
    is_vararg = True

class PyObject_Type(ExternalFunction):
    '''
    Added to aid debugging
    '''
    arg_types = [object_]
    return_type = object_

class PyTuple_Pack(ExternalFunction):
    arg_types = [Py_ssize_t]
    return_type = object_
    is_vararg = True

class Py_BuildValue(ExternalFunction):
    arg_types = [c_string_type]
    return_type = object_
    is_vararg = True

class PyArg_ParseTuple(ExternalFunction):
    arg_types = [object_, c_string_type]
    return_type = int_
    is_vararg = True

class PyObject_Print(ExternalFunction):
    arg_types = [object_, void.pointer(), int_]
    return_type = int_

class PyObject_Str(ExternalFunction):
    arg_types = [object_]
    return_type = object_

class PyObject_GetAttrString(ExternalFunction):
    arg_types = [object_, c_string_type]
    return_type = object_

class PyObject_GetItem(ExternalFunction):
    arg_types = [object_, object_]
    return_type = object_

class PyObject_SetItem(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = int_

class PySlice_New(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = object_

class PyModulo(InternalFunction):

    @property
    def name(self):
        return naming.specialized_mangle('__py_modulo_%s', self.arg_types)

    def implementation(self, module, builder, ret_val):
        different_sign_block = ret_val.append_basic_block('different_sign')
        join_block = ret_val.append_basic_block('join')
        arg2 = ret_val.args[1]
        srem = builder.srem(ret_val.args[0], arg2)
        width = int(typ[1:])
        lbits_shifted = lc.Constant.int(str_to_llvmtype(typ), width - 1)
        srem_sign = builder.lshr(srem, lbits_shifted)
        arg2_sign = builder.lshr(arg2, lbits_shifted)
        same_sign = builder.icmp(lc.ICMP_EQ, srem_sign, arg2_sign)
        builder.cbranch(same_sign, join_block, different_sign_block)
        builder.position_at_end(different_sign_block)
        different_sign_res = builder.add(srem, arg2)
        builder.branch(join_block)
        builder.position_at_end(join_block)
        res = builder.phi(ltyp)
        res.add_incoming(srem, entry_block)
        res.add_incoming(different_sign_res, different_sign_block)
        builder.ret(res)
        return ret_val
