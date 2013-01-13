import ast, inspect, os
import textwrap

from collections import defaultdict
from numba import *
from . import naming
from .minivect import minitypes
import llvm.core
import logging
import traceback
import numba.decorators

logger = logging.getLogger(__name__)

try:
    from meta.decompiler import decompile_func
except Exception, exn:
    logger.warn("Could not import Meta - AST translation will not work "
                "if the source is not available:\n%s" %
                traceback.format_exc())
    decompile_func = None

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
    while len(myast.body) > 0  and isinstance(myast.body[-1], _ast.Pass):
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
    if int(os.environ.get('NUMBA_FORCE_META_AST', 0)):
        func_def = decompile_func(func)
        assert isinstance(func_def, ast.FunctionDef)
        return func_def
    try:
        source = inspect.getsource(func)
    except IOError:
        return decompile_func(func)
    else:
        source = textwrap.dedent(source)
        # Split off decorators
        decorators = 0
        while not source.startswith('def'): # decorator can have multiple lines
            decorator, sep, source = source.partition('\n')
            decorators += 1
        module_ast = ast.parse(source)

        # fix line numbering
        lineoffset = func.func_code.co_firstlineno + decorators
        ast.increment_lineno(module_ast, lineoffset)

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
    import numba.ast_translate as translate

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

live_objects = [] # These are never collected

def keep_alive(py_func, obj):
    """
    Keep an object alive for the lifetime of the translated unit.

    This is a HACK. Make live objects part of the function-cache

    NOTE: py_func may be None, so we can't make it a function attribute
    """
    live_objects.append(obj)

class FunctionCache(object):
    """
    Cache for compiler functions, declared external functions and constants.
    """
    def __init__(self, context):
        self.context = context

        # All numba-compiled functions
        # (py_func) -> (arg_types, flags) -> (signature, llvm_func, ctypes_func)
        self.__compiled_funcs = defaultdict(dict)

    def get_function(self, py_func, argtypes, flags):
        '''Get a compiled function in the the function cache.
        The function must not be an external function.
            
        For an external function, is_registered() must return False.
        '''
        result = None

        assert argtypes is not None
        flags = None # TODO: stub
        argtypes_flags = tuple(argtypes), flags
        if py_func in self.__compiled_funcs:
            result = self.__compiled_funcs[py_func].get(argtypes_flags)

        # DEAD CODE?
        #if result is None and py_func in self.external_functions:
        #    signature, lfunc = self.external_functions[py_func]
        #    result = signature, lfunc, None

        return result

    def is_registered(self, func):
        '''Check if a function is registed to the FunctionCache instance.
        '''
        if isinstance(func, numba.decorators.NumbaFunction):
            return func.py_func in self.__compiled_funcs
        return False

    def register(self, func):
        '''Register a function to the FunctionCache.  

        It is necessary before calling compile_function().
        '''
        self.__compiled_funcs[func]
        
    def compile_function(self, func, argtypes, restype=None,
                         ctypes=False, **kwds):
        """
        Compile a python function given the argument types. Compile only
        if not compiled already, and only if it is registered to the function
        cache.

        Returns a triplet of (signature, llvm_func, python_callable)
        `python_callable` may be the original function, or a ctypes callable
        if the function was compiled.
        """
        # For NumbaFunction, we get the original python function.
        func = getattr(func, 'py_func', func)
        assert func in self.__compiled_funcs, func

        # get the compile flags
        flags = None # stub

        # Search in cache
        result = self.get_function(func, argtypes, flags)
        if result is not None:
            sig, trans, pycall = result
            return sig, trans.lfunc, pycall

        # Compile the function
        from numba import pipeline

        compile_only = getattr(func, '_numba_compile_only', False)
        kwds['compile_only'] = kwds.get('compile_only', compile_only)

        assert kwds.get('llvm_module') is None, kwds.get('llvm_module')

        compiled = pipeline.compile(self.context, func, restype, argtypes,
                                    ctypes=ctypes, **kwds)
        func_signature, translator, ctypes_func = compiled
    
        argtypes_flags = tuple(func_signature.args), flags
        self.__compiled_funcs[func][argtypes_flags] = compiled
        return func_signature, translator.lfunc, ctypes_func

    ## Does not belong to function-cache functionality.
    ## Any caller can easily implement this.
    #    def get_signature(self, argtypes):
    #        '''Get the signature base on the argtypes
    #
    #        create a signature taking N objects and returning an object
    #        '''
    #        assert False
    #        return minitypes.FunctionType(args=(object_,) * len(argtypes),
    #                                      return_type=object_)
    #

    ### Does not need to be here. Too tightly coupled.
    #    def external_function_by_name(self, name, module, **kws):
    #        """
    #        Return the signature and LLVM function declaration given a external 
    #        function name.  The function name can be external or intrinsic 
    #        functions.  The linker is responsible to link the intrinsic library
    #        into the module.  All intrinsics have linkage LINKONCE_ODR; thus,
    #        they are safe to appear in multiple linking modules.
    #            """
    #        assert module is not None
    #        try:
    #            sig, lfunc = self.context.external_library.declare(module,
    #                                                               name,
    #                                                               **kws)
    #        except KeyError:
    #            sig, lfunc = self.context.intrinsic_library.declare(module,
    #                                                                name,
    #                                                                **kws)
    #        return sig, lfunc

    ### Does not need to be here.  Too tightly coupled.
    #    def external_call(self, name, *args, **kw):
    #        '''Builds a call node for an external function.
    #        '''
    #        temp_name = kw.pop('temp_name', name)
    #        llvm_module = kw.pop('llvm_module')
    #        sig, lfunc = self.external_function_by_name(name, llvm_module, **kw)
    #
    #        if name in self.context.external_library:
    #            external_func = self.context.external_library.get(name)
    #            exc_check = dict(badval=external_func.badval,
    #                             goodval=external_func.goodval,
    #                             exc_msg=external_func.exc_msg,
    #                             exc_type=external_func.exc_type,
    #                             exc_args=external_func.exc_args)
    #        else:
    #            exc_check = {}
    #
    #        result = nodes.NativeCallNode(sig, args, lfunc, name=temp_name,
    #                                      **exc_check)
    #        return result

    ### Unused
    #    def build_external_function(self, external_function, module):
    #        """
    #        Build an external function given it's signature information. 
    #        See the `ExternalFunction` class.
    #        """
    #        # TODO: Remove this function
    #        assert False, "Unused"
    #        assert module is not None
    #        try:
    #            lfunc = module.get_function_named(external_function.name)
    #        except llvm.LLVMException:
    #            func_type = minitypes.FunctionType(
    #                    return_type=external_function.return_type,
    #                    args=external_function.arg_types,
    #                    is_vararg=external_function.is_vararg)
    #            lfunc_type = func_type.to_llvm(self.context)
    #            lfunc = module.add_function(lfunc_type, external_function.name)
    #
    #            if isinstance(external_function, InternalFunction):
    #                lfunc.linkage = external_function.linkage
    #                external_function.implementation(module, lfunc)
    #
    #        return lfunc

    # DEAD CODE?
    #def get_string_constant(self, const_str):
    #    if (module, const_str) in self.string_constants:
    #        ret_val = self.string_constants[(module, const_str)]
    #    else:
    #        lconst_str = llvm.core.Constant.stringz(const_str)
    #        ret_val = module.add_global_variable(lconst_str.type, "__STR_%d" %
    #                                             (len(self.string_constants),))
    #        ret_val.initializer = lconst_str
    #        ret_val.linkage = llvm.core.LINKAGE_LINKEONCE_ODR
    #        self.string_constants[(module, const_str)] = ret_val
    #
    #    return ret_val

### Duplicated and unused class? See translate.py
#
#class _LLVMModuleUtils(object):
#    # TODO: rewrite print statements w/ native types to printf during type
#    # TODO: analysis to PrintfNode. Use PrintfNode for debugging
#
#    @classmethod
#    def build_print_string_constant(cls, translator, out_val):
#        # FIXME: This is just a hack to get things going.  There is a
#        # corner case where formatting markup in the string can cause
#        # undefined behavior, and for 100% correctness we'd have to
#        # escape string formatting sequences.
#        llvm_printf = cls.get_printf(translator.mod)
#        return translator.builder.call(llvm_printf, [
#            translator.builder.gep(cls.get_string_constant(translator.mod,
#                                                           out_val),
#                [_int32_zero, _int32_zero])])
#
#    @classmethod
#    def build_print_number(cls, translator, out_var):
#        llvm_printf = cls.get_printf(translator.mod)
#        if out_var.typ[0] == 'i':
#            if int(out_var.typ[1:]) < 64:
#                fmt = "%d"
#                typ = "i32"
#            else:
#                fmt = "%ld"
#                typ = "i64"
#        elif out_var.typ[0] == 'f':
#            if int(out_var.typ[1:]) < 64:
#                fmt = "%f"
#                typ = "f32"
#            else:
#                fmt = "%lf"
#                typ = "f64"
#        else:
#            raise NotImplementedError("FIXME (type %r not supported in "
#                                      "build_print_number())" % (out_var.typ,))
#        return translator.builder.call(llvm_printf, [
#            translator.builder.gep(
#                cls.get_string_constant(translator.mod, fmt),
#                [_int32_zero, _int32_zero]),
#            out_var.llvm(typ, builder = translator.builder)])
#
#    @classmethod
#    def build_debugout(cls, translator, args):
#        if translator.optimize:
#            print("Warning: Optimization turned on, debug output code may "
#                  "be optimized out.")
#        res = cls.build_print_string_constant(translator, "debugout: ")
#        for arg in args:
#            arg_type = arg.typ
#            if arg.typ is None:
#                arg_type = type(arg.val)
#            if isinstance(arg.val, str):
#                res = cls.build_print_string_constant(translator, arg.val)
#            elif typ_isa_number(arg_type):
#                res = cls.build_print_number(translator, arg)
#            else:
#                raise NotImplementedError("Don't know how to output stuff of "
#                                          "type %r at present." % arg_type)
#        res = cls.build_print_string_constant(translator, "\n")
#        return res, None
#
#    @classmethod
#    def build_len(cls, translator, args):
#        if (len(args) == 1 and
#            args[0].lvalue is not None and
#            args[0].lvalue.type == _numpy_array):
#            lfunc = None
#            shape_ofs = _numpy_array_field_ofs['shape']
#            res = translator.builder.load(
#                translator.builder.load(
#                    translator.builder.gep(args[0]._llvm, [
#                        _int32_zero, llvm.core.Constant.int(_int32, shape_ofs)])))
#        else:
#            return cls.build_object_len(translator, args)
#
#        return res, None
#
#    @classmethod
#    def build_zeros_like(cls, translator, args):
#        assert (len(args) == 1 and
#                args[0]._llvm is not None and
#                args[0]._llvm.type == _numpy_array), (
#            "Expected Numpy array argument to numpy.zeros_like().")
#        translator.init_multiarray()
#        larr = args[0]._llvm
#        largs = [translator.builder.load(
#            translator.builder.gep(larr, [
#                _int32_zero,
#                llvm.core.Constant.int(_int32,
#                                _numpy_array_field_ofs[field_name])]))
#                 for field_name in ('ndim', 'shape', 'descr')]
#        largs.append(_int32_zero)
#        lfunc = translator.ma_obj.load_PyArray_Zeros(translator.mod,
#                                                     translator.builder)
#        res = translator.builder.bitcast(
#            translator.builder.call(
#                cls.get_py_incref(translator.mod),
#                [translator.builder.call(lfunc, largs)]),
#            _numpy_array)
#        if __debug__:
#            print "build_zeros_like(): lfunc =", str(lfunc)
#            print "build_zeros_like(): largs =", [str(arg) for arg in largs]
#        return res, args[0].typ
#
#    @classmethod
#    def build_conj(cls, translator, args):
#        print args
#        assert ((len(args) == 1) and (args[0]._llvm is not None) and
#                (args[0]._llvm.type in (_complex64, _complex128)))
#        larg = args[0]._llvm
#        elem_ltyp = _float if larg.type == _complex64 else _double
#        new_imag_lval = translator.builder.fmul(
#            llvm.core.Constant.real(elem_ltyp, -1.),
#            translator.builder.extract_value(larg, 1))
#        assert hasattr(translator.builder, 'insert_value'), (
#            "llvm-py support for LLVMBuildInsertValue() required to build "
#            "code for complex conjugates.")
#        res = translator.builder.insert_value(larg, new_imag_lval, 1)
#        return res, None
#
