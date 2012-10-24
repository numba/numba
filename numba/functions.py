import ast, inspect, os

from numba import *
from . import naming
from .minivect import minitypes
import numba.ast_translate as translate
from numba import nodes
from numba.llnumba.byte_translator import LLVMTranslator

import logging
import traceback
logger = logging.getLogger(__name__)

try:
    from meta.decompiler import decompile_func
except Exception, exn:
    logger.warn("Could not import Meta - AST translation will not work:\n%s" %
                traceback.format_exc())
    decompile_func = None

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

## Fixme: 
##  This should be changed to visit the AST and fix-up where a None object
##  is present as this will likely not work for all AST.
def _fix_ast(myast):
    import _ast

    # Remove Pass nodes from the end of the ast
    while isinstance(myast.body[-1], _ast.Pass):
        del myast.body[-1]
    # Add a return node at the end of the ast if not present
    if len(myast.body) < 1 or not isinstance(myast.body[-1], _ast.Return):
        name = _ast.Name(id='None',ctx=_ast.Load(), lineno=0, col_offset=0)
        myast.body.append(ast.Return(name))
    # remove _decorator list which sometimes confuses ast visitor
    try:
        indx = myast._fields.index('decorator_list')
    except ValueError:
        return
    else:
        myast.decorator_list = []

def _get_ast(func):
    if os.environ.get('NUMBA_FORCE_META_AST'):
        func_def = decompile_func(func)
        assert isinstance(func_def, ast.FunctionDef)
        return func_def
    try:
        source = inspect.getsource(func)
    except IOError:
        return decompile_func(func)
    else:
        module_ast = ast.parse(source)
        assert len(module_ast.body) == 1
        func_def = module_ast.body[0]
        _fix_ast(func_def)
        assert isinstance(func_def, ast.FunctionDef)
        return func_def

def _infer_types(context, func, restype=None, argtypes=None, **kwargs):
    import numba.ast_type_inference as type_inference

    ast = _get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
    return type_inference.run_pipeline(context, func, ast,
                                       func_signature, **kwargs)


def _compile(context, func, restype=None, argtypes=None, ctypes=False,
             compile_only=False, name=None, **kwds):
    """
    Compile a numba annotated function.

        - decompile function into a Python ast
        - run type inference using the given input types
        - compile the function to LLVM
    """
    func_signature, symtab, ast = _infer_types(context, func,
                                               restype, argtypes, **kwds)
    func_name = name or naming.specialized_mangle(func.__name__, func_signature.args)
    func_signature.name = func_name

    t = translate.LLVMCodeGenerator(
        context, func, ast, func_signature=func_signature,
        symtab=symtab, **kwds)
    t.translate()

    if compile_only:
        return func_signature, t.lfunc, None
    if ctypes:
        ctypes_func = t.get_ctypes_func(kwds.get('llvm', True))
        return func_signature, t.lfunc, ctypes_func
    else:
        return func_signature, t.lfunc, t.build_wrapper_function()

class FunctionCache(object):
    """
    Cache for compiler functions, declared external functions and constants.
    """
    def __init__(self, context, module=None):
        self.context = context
        self.module = module or context.llvm_context.get_default_module()

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

    def compile_function(self, func, argtypes, restype=None,
                         ctypes=False, **kwds):
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
                from numba import pipeline

                func = getattr(func, '_numba_func', func)
                compile_only = getattr(func, '_numba_compile_only', False)
                kwds['compile_only'] = kwds.get('compile_only', compile_only)
                # numba function, compile
                func_signature, translator, ctypes_func = pipeline.compile(
                                self.context, func, restype, argtypes,
                                ctypes=ctypes, **kwds)
                self.compiled_functions[func, tuple(func_signature.args)] = (
                                      func_signature, translator, ctypes_func)
                return func_signature, translator.lfunc, ctypes_func

        # print func, getattr(func, '_is_numba_func', False)
        # create a signature taking N objects and returning an object
        signature = ofunc(argtypes=ofunc.arg_types * len(argtypes)).signature
        return signature, None, func

    def function_by_name(self, name, **kws):
        """
        Return the signature and LLVM function given a name. The function must
        either already be compiled, or it must be defined in this module.
        """
        if name in self.external_functions:
            return self.external_functions[name]
        else:
            declared_func = globals()[name](**kws)
            lfunc = self.build_function(declared_func)
            return declared_func.signature, lfunc

    def call(self, name, *args, **kw):
        temp_name = kw.pop('temp_name', name)
        sig, lfunc = self.function_by_name(name, **kw)
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
                lfunc.linkage = external_function.linkage
                external_function.implementation(
                                        self.module, lfunc)

        return lfunc

    def get_string_constant(self, const_str):
        if (module, const_str) in self.string_constants:
            ret_val = self.string_constants[(module, const_str)]
        else:
            lconst_str = llvm.core.Constant.stringz(const_str)
            ret_val = module.add_global_variable(lconst_str.type, "__STR_%d" %
                                                 (len(self.string_constants),))
            ret_val.initializer = lconst_str
            ret_val.linkage = llvm.core.LINKAGE_INTERNAL
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
                        _int32_zero, llvm.core.Constant.int(_int32, shape_ofs)])))
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
                llvm.core.Constant.int(_int32,
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
            llvm.core.Constant.real(elem_ltyp, -1.),
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

    def implementation(self, module, lfunc):
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
        return naming.specialized_mangle('__py_modulo', self.arg_types)

    def implementation(self, module, ret_val):
        _rtype = ret_val.type.pointee.return_type
        _rem = (llvm.core.Builder.srem
               if _rtype.kind == llvm.core.TYPE_INTEGER
               else llvm.core.Builder.frem)
        def _py_modulo (arg1, arg2):
            ret_val = rem(arg1, arg2)
            if ret_val < rtype(0):
                if arg2 > rtype(0):
                    ret_val += arg2
            elif arg2 < rtype(0):
                ret_val += arg2
            return ret_val
        LLVMTranslator(module).translate(
            _py_modulo, llvm_function = ret_val,
            env = {'rtype' : _rtype, 'rem' : _rem})
        return ret_val

class CStringSlice2 (InternalFunction):
    arg_types = [c_string_type, c_string_type, size_t, size_t]
    return_type = void

    def implementation(self, module, ret_val):
        # logger.debug((module, str(ret_val)))
        def _py_c_string_slice(out_string, in_string, lower, upper):
            in_str_len = strlen(in_string)
            zero = lc_size_t(0)
            if lower < zero:
                lower += in_str_len
            if upper < zero:
                upper += in_str_len
            elif upper > in_str_len:
                upper = in_str_len
            temp_len = upper - lower
            if temp_len < zero:
                temp_len = zero
            strncpy(out_string, in_string + lower, temp_len)
            out_string[temp_len] = li8(0)
            return
        LLVMTranslator(module).translate(_py_c_string_slice,
                                         llvm_function = ret_val)
        return ret_val

class CStringSlice2Len(InternalFunction):
    arg_types = [c_string_type, size_t, size_t]
    return_type = size_t

    def implementation(self, module, ret_val):
        def _py_c_string_slice_len(in_string, lower, upper):
            in_str_len = strlen(in_string)
            zero = lc_size_t(0)
            if lower < zero:
                lower += in_str_len
            if upper < zero:
                upper += in_str_len
            elif upper > in_str_len:
                upper = in_str_len
            temp_len = upper - lower
            if temp_len < zero:
                temp_len = zero
            return temp_len + lc_size_t(1)
        LLVMTranslator(module).translate(_py_c_string_slice_len,
                                         llvm_function = ret_val)
        return ret_val

class labs(ExternalFunction):
    arg_types = [long_]
    return_type = long_

class llabs(ExternalFunction):
    arg_types = [longlong]
    return_type = longlong

class atoi(ExternalFunction):
    arg_types = [c_string_type]
    return_type = int_

class atol(ExternalFunction):
    arg_types = [c_string_type]
    return_type = long_

class atoll(ExternalFunction):
    arg_types = [c_string_type]
    return_type = longlong

class atof(ExternalFunction):
    arg_types = [c_string_type]
    return_type = double

#
### Object conversions to native types
#
def create_func(name, restype, argtype, d):
    class PyLong_FromLong(ExternalFunction):
        arg_types = [argtype]
        return_type = restype

    PyLong_FromLong.__name__ = name
    if restype.is_object:
        type = argtype
    else:
        type = restype

    d[type] = PyLong_FromLong
    globals()[name] = PyLong_FromLong

_as_long = {}
def as_long(name, type):
    create_func(name, type, object_, _as_long)

as_long('PyLong_AsLong', long_)
as_long('PyLong_AsUnsignedLong', ulong)
as_long('PyLong_AsLongLong', longlong)
as_long('PyLong_AsUnsignedLongLong', ulonglong)
#as_long('PyLong_AsSize_t', size_t) # new in py3k
as_long('PyLong_AsSsize_t', Py_ssize_t)

class PyFloat_FromDouble(ExternalFunction):
    arg_types = [double]
    return_type = object_

class PyComplex_FromCComplex(ExternalFunction):
    arg_types = [complex128]
    return_type = object_

class PyInt_FromString(ExternalFunction):
    arg_types = [c_string_type, c_string_type.pointer(), int_]
    return_type = object_

class PyFloat_FromString(ExternalFunction):
    arg_types = [object_, c_string_type.pointer()]
    return_type = object_

#
### Conversion of native types to object
#
_from_long = {}
def from_long(name, type):
    create_func(name, object_, type, _from_long)


from_long('PyLong_FromLong', long_)
from_long('PyLong_FromUnsignedLong', ulong)
from_long('PyLong_FromLongLong', longlong)
from_long('PyLong_FromUnsignedLongLong', ulonglong)
from_long('PyLong_FromSize_t', size_t) # new in 2.6
from_long('PyLong_FromSsize_t', Py_ssize_t)

class PyFloat_AsDouble(ExternalFunction):
    arg_types = [object_]
    return_type = double

class PyComplex_AsCComplex(ExternalFunction):
    arg_types = [object_]
    return_type = complex128
