import opcode
import sys
import types
import __builtin__

import numpy as np

from llvm import _ObjectCache, WeakValueDictionary
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le

from utils import itercode, debugout, Complex64, Complex128
#from ._ext import make_ufunc
from .cfg import ControlFlowGraph
from .llvm_types import _plat_bits, _int1, _int8, _int32, _intp, _intp_star, \
    _void_star, _float, _double, _complex64, _complex128, _pyobject_head, \
    _trace_refs_, _head_len, _numpy_struct,  _numpy_array, \
    _numpy_array_field_ofs
from .multiarray_api import MultiarrayAPI

from .minivect import minitypes

if __debug__:
    import pprint

_int32_zero = lc.Constant.int(_int32, 0)

# Translate Python bytecode to LLVM IR

# For type-inference we need a mapping showing what the output type
# is from any operation and the input types.  We can assume if it is
# not in this table that the output type is the same as the input types 

typemaps = {
}

#hasconst
#hasname
#hasjrel
#haslocal
#hascompare
#hasfree


# Convert llvm Type object to kind-bits string
def llvmtype_to_strtype(typ):
    if typ.kind == lc.TYPE_FLOAT:
        return 'f32'
    elif typ.kind == lc.TYPE_DOUBLE:
        return 'f64'
    elif typ.kind == lc.TYPE_INTEGER:
        return 'i%d' % typ.width
    elif typ == _numpy_array: # NOTE: Checks for Numpy arrays need to
                              # happen before the more general
                              # recognition of pointer types,
                              # otherwise it won't ever be used.
        # FIXME: Need to find a way to stash dtype somewhere in here.
        return 'arr[]'
    elif typ == _complex64:
        return 'c64'
    elif typ == _complex128:
        return 'c128'
    elif typ.kind == lc.TYPE_POINTER:
        if typ.pointee.kind == lc.TYPE_FUNCTION:
            return ['func'] + typ.pointee.args
        else:
            n_pointer = 1
            typ = typ.pointee
            while typ.kind == lc.TYPE_POINTER:
                n_pointer += 1
                typ = typ.pointee
            return "%s%s" % (llvmtype_to_strtype(typ), "*" * n_pointer)
    raise NotImplementedError("FIXME: LLVM type conversions for '%s'!" % (typ,))

# We don't support all types....
def pythontype_to_strtype(typ):
    if issubclass(typ, float):
        return 'f64'
    elif issubclass(typ, int):
        return 'i%d' % _plat_bits
    elif issubclass(typ, complex):
        return 'c128'
    elif issubclass(typ, (types.BuiltinFunctionType, types.FunctionType)):
        return ["func"]
    elif issubclass(typ, tuple):
        return 'tuple'

def map_to_strtype(type):
    "Map a minitype or str type to a str type"
    if isinstance(type, minitypes.Type):
        if __debug__:
            print 'CONVERTING', type
        if type.is_float:
            if type.itemsize == 4:
                return 'f'
            elif type.itemsize == 8:
                return 'd'
            else:
                # hurgh
                raise NotImplementedError
            #type = 'f%d' % (type.itemsize * 8,)
        elif type.is_int:
            if type == minitypes.int_:
                type = 'i'
            elif type == minitypes.long_:
                type = 'l'
            else:
                type = 'i%d' % (type.itemsize * 8,)
        elif type.is_function:
            type = ["func"]
        elif type.is_tuple:
            type = 'tuple'
        elif type.is_array:
            type = [map_to_strtype(type.dtype)]
        elif type.is_pointer:
            type = "%s *" % (map_to_strtype(type.base_type))
        else:
            raise NotImplementedError
        if __debug__:
            print type

    return type

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
            ret_val = Variable(lc.Constant.int(_intp, 0))
        return ret_val

    def get_inc(self):
        if len(self.args) > 2:
            ret_val = self.args[2]
        else:
            # FIXME: Need to infer case where this might be over floats.
            ret_val = Variable(lc.Constant.int(_intp, 1))
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
        ret_val = None
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

# Variables placed on the stack. 
#  They allow an indirection
#  So, that when used in an operation, the correct
#  LLVM type can be inserted.  
class Variable(object):
    def __init__(self, val, argtyp = None, exact_typ = None):
        if isinstance(val, Variable):
            self.val = val.val
            self._llvm = val._llvm
            self.typ = val.typ
            return 
        self.val = val
        if isinstance(val, lc.Value):
            self._llvm = val
            if argtyp is None:
                if exact_typ is None:
                    self.typ = llvmtype_to_strtype(val.type)
                else:
                    self.typ = exact_typ
            else:
                self.typ = convert_to_strtype(argtyp)
        else:
            self._llvm = None
            self.typ = pythontype_to_strtype(type(val))

    def __repr__(self):
        return ('<Variable(val=%r, _llvm=%r, typ=%r)>' %
                (self.val, self._llvm, self.typ))

    def llvm(self, typ=None, mod=None, builder = None):
        if self._llvm:
            ret_val = self._llvm
            if typ is not None and typ != self.typ:
                ret_val = None
                ltyp = str_to_llvmtype(typ)
                if builder:
                    ret_val = _LLVMCaster.build_cast(builder, self._llvm, ltyp)
                if ret_val is None:
                    raise ValueError("Unsupported cast (from %r.typ to %r)" %
                                     (self, typ))
            return ret_val
        else:
            if typ is None:
                typ = 'f64' if self.typ is None else self.typ
            if typ == 'f64':
                res = lc.Constant.real(lc.Type.double(), float(self.val))
            elif typ == 'f32':
                res = lc.Constant.real(lc.Type.float(), float(self.val))
            elif typ[0] == 'i':
                res = lc.Constant.int(lc.Type.int(int(typ[1:])),
                                      int(self.val))
            elif typ[0] == 'c':
                ltyp = _float if typ[1:] == '64' else _double
                res = lc.Constant.struct([
                        lc.Constant.real(ltyp, self.val.real),
                        lc.Constant.real(ltyp, self.val.imag)])
            elif typ[0] == 'func':
                res = map_to_function(self.val, typ[1:], mod)
            return res

    def is_phi(self):
        return isinstance(self._llvm, lc.PHINode)

    def is_module(self):
        return isinstance(self.val, types.ModuleType)

# Add complex, unsigned, and bool 
def str_to_llvmtype(str):
    if __debug__:
        print("str_to_llvmtype(): str = %r" % (str,))
    n_pointer = 0
    if str.endswith('*'):
        n_pointer = str.count('*')
        str = str[:-n_pointer]
    if str[0] == 'f':
        if str[1:] == '32':
            ret_val = lc.Type.float()
        elif str[1:] == '64':
            ret_val = lc.Type.double()
    elif str[0] == 'i':
        num = int(str[1:])
        ret_val = lc.Type.int(num)
    elif str[0] == 'c':
        if str[1:] == '64':
            ret_val = _complex64
        elif str[1:] == '128':
            ret_val = _complex128
    elif str.startswith('arr['):
        ret_val = _numpy_array
    else:
        raise TypeError, "Invalid Type: %s" % str
    for _ in xrange(n_pointer):
        ret_val = lc.Type.pointer(ret_val)
    return ret_val

def _handle_list_conversion(typ):
    assert isinstance(typ, list)
    crnt_elem = typ[0]
    dimcount = 1
    while isinstance(crnt_elem, list):
        crnt_elem = crnt_elem[0]
        dimcount += 1
    return dimcount, crnt_elem

def _dtypeish_to_str(typ):
    n_pointer = 0
    if typ.endswith('*'):
        n_pointer = typ.count('*')
        typ = typ[:-n_pointer]
    try:
        dt = np.dtype(typ)
        return ("%s%s%s" % (dt.kind, 8*dt.itemsize, "*" * n_pointer))
    except TypeError:  # already the right form
        return ("%s%s" % (typ, '*' * n_pointer))
    

def convert_to_llvmtype(typ):
    n_pointer = 0
    if isinstance(typ, list):
        return _numpy_array
    return str_to_llvmtype(_dtypeish_to_str(typ))

def _getdim_and_type(typ):
    dim = 1
    res = typ[0]
    while isinstance(res, list):
        res = res[0]
        dim += 1
    return np.dtype(res), dim

def convert_to_ctypes(typ):
    import ctypes
    from numpy.ctypeslib import _typecodes
    if isinstance(typ, list):
        # FIXME: At some point we should add a type check to the
        # wrapper code s.t. it ensures the given argument conforms to
        # the following:
        #     np.ctypeslib.ndpointer(dtype = np.dtype(crnt_elem),
        #                            ndim = dimcount,
        #                            flags = 'C_CONTIGUOUS')
        # For now, we'll just allow any Python objects, and hope for the best.
        return ctypes.py_object
    n_pointer = 0
    if typ.endswith('*'):
        n_pointer = typ.count('*')
        typ = typ[:-n_pointer]
        if __debug__:
            print("convert_to_ctypes(): n_pointer = %d, typ' = %r" %
                  (n_pointer, typ))
    dtype_str = np.dtype(typ).str
    if dtype_str[1] == 'c':
        if dtype_str[2:] == '8':
            ret_val = Complex64
        elif dtype_str[2:] == '16':
            ret_val = Complex128
        else:
            raise NotImplementedError("Numba does not currently support "
                                      "ctypes wrapping for type %r." % (typ,))
    else:
        ret_val = _typecodes[np.dtype(typ).str]
    for _ in xrange(n_pointer):
        ret_val = ctypes.POINTER(ret_val)
    return ret_val

def convert_to_strtype(typ):
    # FIXME: This current mapping preserves dtype information, but
    # loses ndims.
    arr = 0
    if isinstance(typ, list):
        arr = 1
        _, typ = _handle_list_conversion(typ)
    return '%s%s%s' % ('arr[' * arr, _dtypeish_to_str(typ), ']' * arr)

# Add complex, unsigned, and bool
def typcmp(type1, type2):
    if type1==type2:
        return 0
    kind1 = type1[0]
    kind2 = type2[0]
    if kind1 == kind2:
        return cmp(int(type1[1:]),int(type2[1:]))
    if kind1 == 'f':
        return 1
    else:
        return -1

def typ_isa_number(typ):
    ret_val = ((not typ.endswith('*')) and
               (typ.startswith(('i', 'f', 'c'))))
    return ret_val

# Both inputs are Variable objects
#  Resolves types on one of them. 
#  Won't work if both need resolving
# Currently delegates casting to Variable.llvm(), but only in the
# presence of a builder instance.
def resolve_type(arg1, arg2, builder = None):
    if __debug__:
        print("resolve_type(): arg1 = %r, arg2 = %r" % (arg1, arg2))
    typ = None
    typ1 = None
    typ2 = None
    if arg1._llvm is not None:
        typ1 = arg1.typ
    else:
        try:
            str_to_llvmtype(arg1.typ)
            typ1 = arg1.typ
        except TypeError:
            pass
    if arg2._llvm is not None:
        typ2 = arg2.typ
    else:
        try:
            str_to_llvmtype(arg2.typ)
            typ2 = arg2.typ
        except TypeError:
            pass
    if typ1 is None and typ2 is None:
        raise TypeError, "Both types not valid"
    # The following is trying to enforce C-like upcasting rules where
    # we try to do the following:
    # * Use higher precision if the types are identical in kind.
    # * Use floats instead of integers.
    if typ1 is None:
        typ = typ2
    elif typ2 is None:
        typ = typ1
    else:
        # FIXME: What about arithmetic operations on arrays?  (Though
        # we'd need to support code generation for them first...)
        if typ_isa_number(typ1) and typ_isa_number(typ2):
            if typ1[0] == typ2[0]:
                if int(typ1[1:]) > int(typ2[1:]):
                    typ = typ1
                else:
                    typ = typ2
            elif typ1[0] == 'c': # Complex values trump all...
                typ = typ1
            elif typ2[0] == 'c':
                typ = typ2
            elif typ1[0] == 'f': # Floating point values trump integers...
                typ = typ1
            elif typ2[0] == 'f':
                typ = typ2
        else:
            # Fall-through case: just use the left hand operand's type...
            typ = typ1
    if __debug__:
        print("resolve_type() ==> %r" % (typ,))
    return (typ,
            arg1.llvm(typ, builder = builder),
            arg2.llvm(typ, builder = builder))

# This won't convert any llvm types.  It assumes 
#  the llvm types in args are either fixed or not-yet specified.
def func_resolve_type(mod, func, args):
    # already an llvm function
    if func.val and func.val is func._llvm:
        typs = [llvmtype_to_strtype(x) for x in func._llvm.type.pointee.args]
        lfunc = func._llvm
    else:
        # we need to generate the function including the types
        typs = [arg.typ if arg._llvm is not None else '' for arg in args]
        # pick first one as choice
        choicetype = None
        for typ in typs:
            if typ is not None:
                choicetype = typ
                break
        if choicetype is None:
            raise TypeError, "All types are unspecified"
        typs = [choicetype if x is None else x for x in typs]
        lfunc = map_to_function(func.val, typs, mod)

    llvm_args = [arg.llvm(typ) for typ, arg in zip(typs, args)]
    return lfunc, llvm_args

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

class _LLVMModuleUtils(object):
    __string_constants = {}

    @classmethod
    def get_printf(cls, module):
        try:
            ret_val = module.get_function_named('printf')
        except:
            ret_val = module.add_function(
                lc.Type.function(_int32, [_void_star], True),
                'printf')
        return ret_val

    @classmethod
    def get_string_constant(cls, module, const_str):
        if (module, const_str) in cls.__string_constants:
            ret_val = cls.__string_constants[(module, const_str)]
        else:
            lconst_str = lc.Constant.stringz(const_str)
            ret_val = module.add_global_variable(lconst_str.type, "__STR_%d" % 
                                                 (len(cls.__string_constants),))
            ret_val.initializer = lconst_str
            ret_val.linkage = lc.LINKAGE_INTERNAL
            cls.__string_constants[(module, const_str)] = ret_val
        return ret_val

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
                fmt_args = [out_var.llvm("i32", builder = translator.builder)]
            else:
                fmt = "%ld"
                fmt_args = [out_var.llvm("i64", builder = translator.builder)]
        elif out_var.typ[0] == 'f':
            if int(out_var.typ[1:]) < 64:
                fmt = "%f"
                fmt_args = [out_var.llvm("f32", builder = translator.builder)]
            else:
                fmt = "%lf"
                fmt_args = [out_var.llvm("f64", builder = translator.builder)]
        elif out_var.typ[0] == 'c':
            if int(out_var.typ[1:]) < 128:
                fmt = "(%f%+fj)"
                lvar = out_var.llvm(out_var.typ, builder = translator.builder)
            else:
                fmt = "(%lf%+lfj)"
                lvar = out_var.llvm(out_var.typ, builder = translator.builder)
            fmt_args = [translator.builder.extract_value(lvar, 0),
                        translator.builder.extract_value(lvar, 1)]
        else:
            raise NotImplementedError("FIXME (type %r not supported in "
                                      "build_print_number())" % (out_var.typ,))
        return translator.builder.call(llvm_printf, [
                translator.builder.gep(
                    cls.get_string_constant(translator.mod, fmt),
                    [_int32_zero, _int32_zero])] + fmt_args)

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
            args[0]._llvm is not None and
            args[0]._llvm.type == _numpy_array):
            lfunc = None
            shape_ofs = _numpy_array_field_ofs['shape']
            res = translator.builder.load(
                translator.builder.load(
                    translator.builder.gep(args[0]._llvm, [
                            _int32_zero, lc.Constant.int(_int32, shape_ofs)])))
        else:
            raise NotImplementedError("Currently unable to handle calls to "
                                      "len() for arguments that are not Numpy "
                                      "arrays.")
        return res, None

    @classmethod
    def get_py_incref(cls, module):
        try:
            ret_val = module.get_function_named('Py_IncRef')
        except:
            ret_val = module.add_function(
                lc.Type.function(_void_star, [_void_star]), 'Py_IncRef')
        return ret_val

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
    def get_py_modulo(cls, module, typ):
        modulo_fn = '__py_modulo_%s' % typ
        try:
            ret_val = module.get_function_named(modulo_fn)
        except:
            ltyp = str_to_llvmtype(typ)
            ret_val = module.add_function(
                lc.Type.function(ltyp, [ltyp, ltyp]), modulo_fn)
            ret_val.linkage = lc.LINKAGE_INTERNAL
            entry_block = ret_val.append_basic_block('entry')
            different_sign_block = ret_val.append_basic_block('different_sign')
            join_block = ret_val.append_basic_block('join')
            builder = lc.Builder.new(entry_block)
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

PY_CALL_TO_LLVM_CALL_MAP = {
    debugout : _LLVMModuleUtils.build_debugout,
    len : _LLVMModuleUtils.build_len,
    np.zeros_like : _LLVMModuleUtils.build_zeros_like,
    np.complex64.conj : _LLVMModuleUtils.build_conj,
    np.complex128.conj : _LLVMModuleUtils.build_conj,
    np.complex64.conjugate : _LLVMModuleUtils.build_conj,
    np.complex128.conjugate : _LLVMModuleUtils.build_conj,
}

class LLVMControlFlowGraph (ControlFlowGraph):
    def __init__ (self, translator = None):
        self.translator = translator
        super(LLVMControlFlowGraph, self).__init__()

    def add_block (self, key, value = None):
        if self.translator is not None:
            if key not in self.translator.blocks:
                lfunc = self.translator.lfunc
                lblock = lfunc.append_basic_block('BLOCK_%d' % key)
                assert isinstance(lblock, lc.BasicBlock), (
                    "Expected %r from llvm-py, got instance of type %r, "
                    "however." % (lc.BasicBlock, type(lblock)))
                self.translator.blocks[key] = lblock
            else:
                lblock = self.translator.blocks[key]
            if value is None:
                value = lblock
        return super(LLVMControlFlowGraph, self).add_block(key, value)

    # The following overloaded methods implement a state machine
    # intended to recognize the opcode sequence: GET_ITER, FOR_ITER,
    # STORE_FAST.  Any other sequence is (currently) rejected in
    # control-flow analysis.

    def op_GET_ITER (self, i, op, arg):
        self.saw_get_iter_at = (self.crnt_block, i)
        return False

    def op_FOR_ITER (self, i, op, arg):
        if (hasattr(self, "saw_get_iter_at") and
            self.saw_get_iter_at[1] == i - 1):
            self.add_block(i)
            self.add_block(i + 3)
            self.add_edge(self.crnt_block, i + 3)
            self.add_edge(i, i + 3)
            self.add_block(i + arg + 3)
            self.add_edge(i + 3, i + arg + 3)
            # The following is practically meaningless since we are
            # hijacking normal control flow, and injecting a synthetic
            # basic block at i + 3, but still correct if we want to
            # enforce some weird loop invariant over the symbolic
            # execution loop.
            self.crnt_block = i
        else:
            raise NotImplementedError("Unable to handle FOR_ITER appearing "
                                      "after any opcode other than GET_ITER.")
        return True

    def op_STORE_FAST (self, i, op, arg):
        if hasattr(self, "saw_get_iter_at"):
            get_iter_block, get_iter_index = self.saw_get_iter_at
            del self.saw_get_iter_at
            if get_iter_index == i - 4:
                self.blocks_writes[get_iter_block].add(arg)
                self.blocks_writer[get_iter_block][arg] = get_iter_index
                self.blocks_writes[i - 3].add(arg)
                self.blocks_writer[i - 3][arg] = i - 3
                self.blocks_reads[i].add(arg)
                self.add_block(i + 3)
                self.add_edge(i, i + 3)
            else:
                # FIXME: (?) Are there corner cases where this will fail to
                # eventually detect a pattern miss?
                raise NotImplementedError(
                    "Detected GET_ITER, FOR_ITER opcodes not immediately "
                    "followed by STORE_FAST at instruction index %d." %
                    (get_iter_index,))
            ret_val = False
        else:
            ret_val = super(LLVMControlFlowGraph, self).op_STORE_FAST(
                i, op, arg)
        return ret_val

    def compute_dataflow (self):
        """Overload the base class to induce a writer update for phi
        nodes, otherwise later phi node calculations won't work."""
        ret_val = super(LLVMControlFlowGraph, self).compute_dataflow()
        self.update_for_ssa()
        return ret_val

class Translate(object):
    def __init__(self, func, ret_type='d', arg_types=['d'], **kws):
        self.func = func
        self.fco = func.func_code
        self.names = self.fco.co_names
        self.varnames = self.fco.co_varnames
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code
        # Just the globals we will use
        self._myglobals = {}
        for name in self.names:
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                # Assumption here is that any name not in globals or
                # builtins is an attribtue.
                self._myglobals[name] = getattr(__builtin__, name, None)

        ret_type, arg_types = self.map_types(ret_type, arg_types)

        # NOTE: Was seeing weird corner case where
        # llvm.core.Module.new() was not returning a module object,
        # thinking this was caused by compiling the same function
        # twice while the module was being garbage collected, and
        # llvm.core.Module.new() would return whatever was left lying
        # around.  Using the translator address in the module name
        # might fix this.

        # NOTE: It doesn't.  Have to manually flush the llvm-py object cache
        # since not even forcing garbage collection is reliable.

        global _ObjectCache
        setattr(_ObjectCache, '_ObjectCache__instances', WeakValueDictionary())

        self.mod = lc.Module.new('%s_mod_%x' % (func.__name__, id(self)))
        assert isinstance(self.mod, lc.Module), (
            "Expected %r from llvm-py, got instance of type %r, however." %
            (lc.Module, type(self.mod)))
        self._delaylist = [range, xrange, enumerate]
        self.ret_type = ret_type
        self.arg_types = arg_types
        self.setup_func()
        self.ee = None
        self.ma_obj = None
        self.optimize = kws.pop('optimize', True)
        self.flags = kws

    def map_types(self, ret_type, arg_types):
        return map_to_strtype(ret_type), [map_to_strtype(arg_type)
                                              for arg_type in arg_types]



    def setup_func(self):
        # The return type will not be known until the return
        #   function is created.   So, we will need to 
        #   walk through the code twice....
        #   Once to get the type of the return, and again to 
        #   emit the instructions.
        # For now, we assume the function has been called already
        #   or the return type is otherwise known and passed in
        self.ret_ltype = convert_to_llvmtype(self.ret_type)
        # The arg_ltypes we will be able to get from what is passed in
        argnames = self.fco.co_varnames[:self.fco.co_argcount]
        self.arg_ltypes = [convert_to_llvmtype(x) for x in self.arg_types]
        ty_func = lc.Type.function(self.ret_ltype, self.arg_ltypes)        
        self.lfunc = self.mod.add_function(ty_func, self.func.func_name)
        assert isinstance(self.lfunc, lc.Function), (
            "Expected %r from llvm-py, got instance of type %r, however." %
            (lc.Function, type(self.lfunc)))
        self.nlocals = len(self.fco.co_varnames)
        self._locals = [None] * self.nlocals
        for i, (name, typ) in enumerate(zip(argnames, self.arg_types)):
            assert isinstance(self.lfunc.args[i], lc.Argument), (
                "Expected %r from llvm-py, got instance of type %r, however." %
                (lc.Argument, type(self.lfunc.args[i])))
            self.lfunc.args[i].name = name
            # Store away arguments in locals
            self._locals[i] = Variable(self.lfunc.args[i], typ)
        entry = self.lfunc.append_basic_block('Entry')
        assert isinstance(entry, lc.BasicBlock), (
            "Expected %r from llvm-py, got instance of type %r, however." %
            (lc.BasicBlock, type(entry)))
        self.blocks = {0:entry}
        self.cfg = None
        self.blocks_locals = {}
        self.pending_phis = {}
        self.pending_blocks = {}
        self.stack = []
        self.loop_stack = []

    def translate(self):
        """Translate the function
        """
        self.cfg = LLVMControlFlowGraph.build_cfg(self.fco, self)
        self.cfg.compute_dataflow()
        if __debug__:
            self.cfg.pprint()
        for i, op, arg in itercode(self.costr):
            name = opcode.opname[op]
            # Change the builder if the line-number 
            # is in the list of blocks.
            if i in self.blocks.keys():
                if i > 0:
                    # Emit a branch to link blocks up if the previous
                    # block was not explicitly branched out of...
                    bb_instrs = self.builder.basic_block.instructions
                    if ((len(bb_instrs) == 0) or
                        (not bb_instrs[-1].is_terminator)):
                        self.builder.branch(self.blocks[i])

                    # Copy the locals exiting the soon to be
                    # preceeding basic block.
                    self.blocks_locals[self.crnt_block] = self._locals[:]

                    # Ensure we are playing with locals that might
                    # actually precede the next block.
                    self.check_locals(i)

                self.crnt_block = i
                self.builder = lc.Builder.new(self.blocks[i])
                self.build_phi_nodes(self.crnt_block)
            getattr(self, 'op_'+name)(i, op, arg)

        # Perform code optimization
        if self.optimize:
            fpm = lp.FunctionPassManager.new(self.mod)
            fpm.initialize()
            #fpm.add(lp.PASS_DEAD_CODE_ELIMINATION)
            #fpm.run(self.lfunc)
            #fpm.finalize()

        if __debug__:
            print self.mod

    def _get_ee(self):
        if self.ee is None:
            self.ee = le.ExecutionEngine.new(self.mod)
        return self.ee

    def build_call_to_translated_function(self, target_translator, args):
        # FIXME: At some point, I assume we'll actually want to index
        # by argument types, so this will grab the best suited
        # translation of a function based on the caller circumstance.
        if len(args) != len(self.arg_types):
            raise TypeError("Mismatched argument count in call to translated "
                            "function (%r)." % (self.func))
        ee = self._get_ee()
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

    def has_pending_phi(self, instr_index, local_index):
        return ((instr_index in self.pending_phis) and 
                (local_index in self.pending_phis[instr_index]))

    def add_pending_phi(self, instr_index, local_index, phi, pred):
        if instr_index not in self.pending_phis:
            locals_map = {}
            self.pending_phis[instr_index] = locals_map
        else:
            locals_map = self.pending_phis[instr_index]
        if local_index not in locals_map:
            # Note that the same reaching definition might "arrive"
            # via more than one predecessor block, so we keep a list
            # of predecessors, not just one.
            locals_map[local_index] = (phi, [pred])
        else:
            assert locals_map[local_index][0] == phi, (
                "Internal compiler error!")
            locals_map[local_index][1].append(pred)

    def handle_pending_phi(self, instr_index, local_index, value):
        phi, pred_lblocks = self.pending_phis[instr_index][local_index]
        if isinstance(value, Variable):
            value = value.llvm(llvmtype_to_strtype(phi.type))
        else:
            assert isinstance(value, lc.Value), "Internal compiler error!"
        for pred_lblock in pred_lblocks:
            phi.add_incoming(value, pred_lblock)

    def add_phi_incomming(self, phi, crnt_block, pred, local):
        '''Take one of three actions:

        1. If the predecessor block has already been visited, add its
        exit value for the given local to the phi node under
        construction.

        2. If the predecessor has not been visited, but the block that
        defines the reaching definition for that local value, add the
        definition value to the phi node under construction.

        3. If the reaching definition has not been visited, add a
        pending call to PHINode.add_incoming() which will be caught by
        op_STORE_LOCAL().
        '''
        if pred in self.blocks_locals and pred not in self.pending_blocks:
            pred_locals = self.blocks_locals[pred]
            assert pred_locals[local] is not None, ("Internal error.  "
                "Local value definition missing from block that has "
                "already been visited.")
            phi.add_incoming(pred_locals[local].llvm(
                    llvmtype_to_strtype(phi.type)), self.blocks[pred])
        else:
            reaching_defs = self.cfg.get_reaching_definitions(crnt_block)
            if __debug__:
                print("add_phi_incomming(): reaching_defs = %s\n    "
                      "crnt_block=%r, pred=%r, local=%r" %
                      (pprint.pformat(reaching_defs), crnt_block, pred, local))
            definition_block = reaching_defs[pred][local]
            if ((definition_block in self.blocks_locals) and
                (definition_block not in self.pending_blocks)):
                defn_locals = self.blocks_locals[definition_block]
                assert defn_locals[local] is not None, ("Internal error.  "
                    "Local value definition missing from block that has "
                    "already been visited.")
                phi.add_incomming(defn_locals[local].llvm(
                        llvmtype_to_strtype(phi.type)), self.blocks[pred])
            else:
                definition_index = self.cfg.blocks_writer[definition_block][
                    local]
                self.add_pending_phi(definition_index, local, phi,
                                     self.blocks[pred])

    def build_phi_nodes(self, crnt_block):
        '''Determine if any phi nodes need to be created, and if so,
        do it.'''
        preds = self.cfg.blocks_in[crnt_block]
        if len(preds) > 1:
            phis_needed = self.cfg.phi_needed(crnt_block)
            if len(phis_needed) > 0:
                reaching_defs = self.cfg.get_reaching_definitions(crnt_block)
                for local in phis_needed:
                    # Infer type from current local value.
                    oldlocal = self._locals[local]
                    # NOTE: Also seeing builder.phi returning
                    # non-PHINode instances intermittently (see NOTE
                    # above for llvm.core.Module.new()).
                    phi = self.builder.phi(str_to_llvmtype(oldlocal.typ))
                    assert isinstance(phi, lc.PHINode), (
                        "Intermittent llvm-py error encountered (builder.phi()"
                        " result type was %r, not %r)." %
                        (type(phi), lc.PHINode))
                    newlocal = Variable(phi)
                    self._locals[local] = newlocal
                    for pred in preds:
                        self.add_phi_incomming(phi, crnt_block, pred, local)
                    # This is a local write, even if it is synthetic,
                    # so check to see if we are responsible for back
                    # patching any pending phis.
                    if self.has_pending_phi(crnt_block, local):
                        # FIXME: There may be the potential for a
                        # corner case where a STORE_FAST occurs at the
                        # top of a join.  This will cause multiple,
                        # ambiguous, calls to PHINode.add_incomming()
                        # (once here, and once in op_STORE_FAST()).
                        # Currently checking for this in
                        # numba.cfg.ControlFlowGraph._writes_local().
                        # Assertion should fail when
                        # LLVMControlFlowGraph calls
                        # self.update_for_ssa().
                        self.handle_pending_phi(crnt_block, local, phi)

    def get_preceding_locals(self, preds):
        '''Given an iterable set of preceding basic blocks, check to
        see if one of them has already been symbolically executed.  If
        so, return the symbolic locals recorded as leaving that basic
        block.  Returns None otherwise.'''
        pred_list = list(preds)
        pred_list.sort()
        pred_list.reverse()
        next_locals = None
        for next_pred in pred_list:
            if next_pred in self.blocks_locals:
                next_locals = self.blocks_locals[next_pred]
                break
        return next_locals

    def check_locals(self, i):
        '''Given the instruction index of the next block, determine if
        the current block is in the set of the next block's
        predecessors.  If not, change out the locals to those of a
        predecessor that has already been symbolically run.
        '''
        if self.crnt_block not in self.cfg.blocks_in[i]:
            next_locals = self.get_preceding_locals(self.cfg.blocks_in[i])
            if next_locals is None:
                if (len(self.stack) > 0 and
                    isinstance(self.stack[-1].val, DelayedObj)):
                    # When we detect that we are in a for loop over a
                    # simple range, fallback to the block dominator so we
                    # at least have type information for the locals.
                    assert next_locals is None, "Internal compiler error!"
                    next_locals = self.get_preceding_locals(
                        self.cfg.blocks_dom[i])
                elif len(self.cfg.blocks_in[i]) == 0:
                    # Ignore unreachable basic blocks (this happens when
                    # the Python compiler doesn't know that all paths have
                    # already returned something).
                    assert i != 0, ("Translate.check_locals() should not be "
                                    "called for the entry block.")
                    next_locals = self._locals
                else:
                    assert next_locals is not None, "Internal compiler error!"
            self._locals = next_locals[:]

    def get_ctypes_func(self, llvm=True):
        ee = self._get_ee()
        import ctypes
        restype = convert_to_ctypes(self.ret_type)
        prototype = ctypes.CFUNCTYPE(restype,
                                     *[convert_to_ctypes(x)
                                       for x in self.arg_types])
        if hasattr(restype, 'make_ctypes_prototype_wrapper'):
            # See numba.utils.ComplexMixin for an example of
            # make_ctypes_prototype_wrapper().
            prototype = restype.make_ctypes_prototype_wrapper(prototype)
        if llvm:
            PY_CALL_TO_LLVM_CALL_MAP[self.func] = \
                self.build_call_to_translated_function
            return prototype(ee.get_pointer_to_function(self.lfunc))
        else:
            return prototype(self.func)

    def make_ufunc(self, name=None):
        raise NotImplementedError('Extension module for make_ufunc() is '
                                  'currently disabled.')
        #if self.ee is None:
        #    self.ee = le.ExecutionEngine.new(self.mod)
        #if name is None:
        #    name = self.func.func_name
        #return make_ufunc(self.ee.get_pointer_to_function(self.lfunc), 
        #                       0, name)

    # This won't convert any llvm types.  It assumes 
    #  the llvm types in args are either fixed or not-yet specified.
    def func_resolve_type(self, func, args):
        # already an llvm function
        if func.val and func.val is func._llvm:
            typs = [llvmtype_to_strtype(x) for x in func._llvm.type.pointee.args]
            lfunc = func._llvm
        # The function is one of the delayed list
        elif func.val in self._delaylist:
            return None, DelayedObj(func.val, args)
        else:
            # Assume we are calling into an intrinsic function...
            # we need to generate the function including the types
            typs = [arg.typ if arg._llvm is not None else '' for arg in args]
            # pick first one as choice
            choicetype = None
            for typ in typs:
                if typ is not None:
                    choicetype = typ
                    break
            if choicetype is None:
                raise TypeError, "All types are unspecified"
            typs = [choicetype if x is None else x for x in typs]
            lfunc = map_to_function(func.val, typs, self.mod)

        llvm_args = [arg.llvm(typ) for typ, arg in zip(typs, args)]
        return lfunc, llvm_args

    def init_multiarray(self):
        '''Builds the MultiarrayAPI object and adds a PyArray_API
        variable to the current module under construction.

        Should be called by code generators (like
        _LLVMModuleUtils.build_zeros_like()) that build Numpy-like
        functionality to ensure the ma_obj member has been
        initialized.'''
        if self.ma_obj is None:
            self.ma_obj = MultiarrayAPI()
            self.ma_obj.set_PyArray_API(self.mod)

    def _revisit_block(self, block_index):
        block_state = (self.crnt_block, self.builder, self._locals[:])
        self.crnt_block = block_index
        self.builder = lc.Builder.new(self.blocks[block_index])
        self.builder.position_at_beginning(self.blocks[block_index])
        return block_state

    def _restore_block(self, block_state):
        self.blocks_locals[self.crnt_block] = self._locals[:]
        self.crnt_block, self.builder, self._locals = block_state

    def _generate_for_loop(self, i , op, arg, delayer):
        '''Generates code for a simple for loop (a loop over range,
        xrange, or arange).'''
        false_jump_target = self.pending_blocks.pop(i - 3)
        crnt_block_data = self._revisit_block(i - 3)
        inc_variable = delayer.val.get_inc()
        self.op_LOAD_FAST(i - 3, None, arg)
        self.stack.append(inc_variable)
        self.op_INPLACE_ADD(i - 3, None, None)
        self.op_STORE_FAST(i - 3, None, arg)
        self._restore_block(crnt_block_data)
        self.op_LOAD_FAST(i, None, arg)
        self.stack.append(delayer.val.get_stop())
        # FIXME: This should really test to see if we are increasing
        # the iteration variable (inc > 0) or decreasing (inc < 0),
        # and select the comparison operator based on that.  This currently
        # only works if the increment is a constant integer value.
        cmp_op_str = '<'
        llvm_inc = inc_variable._llvm
        # FIXME: Handle other types.
        if hasattr(inc_variable, 'as_int') and llvm_inc.as_int() < 0:
            cmp_op_str = '>='
        self.op_COMPARE_OP(i, None, opcode.cmp_op.index(cmp_op_str))
        self.op_POP_JUMP_IF_FALSE(i, None, false_jump_target)

    def _generate_complex_op(self, op, arg1, arg2):
        rres, ires = op(self.builder.extract_value(arg1, 0),
                        self.builder.extract_value(arg1, 1),
                        self.builder.extract_value(arg2, 0),
                        self.builder.extract_value(arg2, 1))
        ret_val = self.builder.insert_value(
            self.builder.insert_value(lc.Constant.undef(arg1.type), rres, 0),
            ires, 1)
        return ret_val

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
                                          self.builder.fmul(arg1i, arg2r)),
                        divisor))

    def op_LOAD_FAST(self, i, op, arg):
        self.stack.append(Variable(self._locals[arg]))

    def op_STORE_FAST(self, i, op, arg):
        oldval = self._locals[arg]
        newval = self.stack.pop(-1)
        if isinstance(newval.val, DelayedObj):
            self._generate_for_loop(i, op, arg, newval)
        else:
            if self.has_pending_phi(i, arg):
                self.handle_pending_phi(i, arg, newval)
            self._locals[arg] = newval

    def op_LOAD_GLOBAL(self, i, op, arg):
        self.stack.append(Variable(self._myglobals[self.names[arg]]))

    def op_LOAD_CONST(self, i, op, arg):
        const = Variable(self.constants[arg])
        self.stack.append(const)
    
    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        if __debug__:
            print("op_BINARY_ADD(): %r + %r" % (arg1, arg2))
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'f':
            res = self.builder.fadd(arg1, arg2)
        elif typ[0] == 'i':
            res = self.builder.add(arg1, arg2)
        elif typ[0] == 'c':
            res = self._generate_complex_op(self._complex_add, arg1, arg2)
        else:
            raise NotImplementedError("FIXME")
        self.stack.append(Variable(res))

    def op_INPLACE_ADD(self, i, op, arg):
        # FIXME: Trivial inspection seems to illustrate a mostly
        # identical semantics to BINARY_ADD for numerical inputs.
        # Verify this, or figure out what the corner cases are that
        # require a separate symbolic execution procedure.
        return self.op_BINARY_ADD(i, op, arg)

    def op_BINARY_SUBTRACT(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'f':
            res = self.builder.fsub(arg1, arg2)
        elif typ[0] == 'i':
            res = self.builder.sub(arg1, arg2)
        elif typ[0] == 'c':
            res = self._generate_complex_op(self._complex_sub, arg1, arg2)
        else:
            raise NotImplementedError("FIXME")
        self.stack.append(Variable(res))

    def op_INPLACE_SUBTRACT(self, i, op, arg):
        return self.op_BINARY_SUBTRACT(i, op, arg)

    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'f':
            res = self.builder.fmul(arg1, arg2)
        elif typ[0] == 'i':
            res = self.builder.mul(arg1, arg2)
        elif typ[0] == 'c':
            res = self._generate_complex_op(self._complex_mul, arg1, arg2)
        else:
            raise NotImplementedError("FIXME")
        self.stack.append(Variable(res))

    def op_INPLACE_MULTIPLY(self, i, op, arg):
        # FIXME: See note for op_INPLACE_ADD
        return self.op_BINARY_MULTIPLY(i, op, arg)

    def op_BINARY_DIVIDE(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'f':
            res = self.builder.fdiv(arg1, arg2)
        elif typ[0] == 'i':
            res = self.builder.sdiv(arg1, arg2)
            # XXX: FIXME-need udiv as well.
        elif typ[0] == 'c':
            res = self._generate_complex_op(self._complex_div, arg1, arg2)
        else:
            raise NotImplementedError("FIXME")
        self.stack.append(Variable(res))

    def op_INPLACE_DIVIDE(self, i, op, arg):
        return self.op_BINARY_DIVIDE(i, op, arg)

    def op_BINARY_FLOOR_DIVIDE(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'i':
            res = self.builder.sdiv(arg1, arg2)
        else:
            raise NotImplementedError('// for type %r' % typ)
        self.stack.append(Variable(res))

    def op_BINARY_MODULO(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'f':
            res = self.builder.frem(arg1, arg2)
        elif typ[0] == 'i': # typ[0] == 'i'
            # NOTE: There is a discrepancy between LLVM remainders and
            # Python's modulo operator.  See:
            # http://llvm.org/docs/LangRef.html#i_srem
            # We handle this using a special function.
            mod_fn = _LLVMModuleUtils.get_py_modulo(self.mod, typ)
            res = self.builder.call(mod_fn, [arg1, arg2])
            # FIXME: Add unsigned integer modulo (which should be the
            # same as urem).
        else:
            raise NotImplementedError('% for type %r' % (typ,))
        self.stack.append(Variable(res))

    def op_BINARY_POWER(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        args = [arg1.llvm(arg1.typ), arg2.llvm(arg2.typ)]
        if arg2.typ[0] == 'i':
            INTR = getattr(lc, 'INTR_POWI')
        elif arg2.typ[0] == 'f': # make sure it's float
            INTR = getattr(lc, 'INTR_POW')
        else:
            raise NotImplementedError('** for type %r' % (typ,))
        typs = [str_to_llvmtype(x.typ) for x in [arg1, arg2]]
        func = lc.Function.intrinsic(self.mod, INTR, typs)
        res = self.builder.call(func, args)
        self.stack.append(Variable(res))
        

    def op_RETURN_VALUE(self, i, op, arg):
        val = self.stack.pop(-1)
        if val.val is None:
            self.builder.ret(lc.Constant.real(self.ret_ltype, 0))
        else:
            self.builder.ret(val.llvm(llvmtype_to_strtype(self.ret_ltype),
                                      builder = self.builder))
        # Add a new block at the next instruction if not at end
        if i+1 < len(self.costr) and i+1 not in self.blocks.keys():
            blk = self.lfunc.append_basic_block("RETURN_%d" % i)
            self.blocks[i+1] = blk


    def op_COMPARE_OP(self, i, op, arg):
        cmpop = opcode.cmp_op[arg]
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2, self.builder)
        if typ[0] == 'f':
            res = self.builder.fcmp(_compare_mapping_float[cmpop], 
                                    arg1, arg2)
        else: # integer FIXME: need unsigned as well...
            res = self.builder.icmp(_compare_mapping_sint[cmpop], 
                                    arg1, arg2)
        self.stack.append(Variable(res))

    def op_POP_JUMP_IF_FALSE(self, i, op, arg):
        # We need to create two blocks.
        #  One for the next instruction (just past the jump)
        #  and another for the block to be jumped to.
        if (i + 3) not in self.blocks:
            cont = self.lfunc.append_basic_block("CONT_%d"% i )
            self.blocks[i+3]=cont
        else:
            cont = self.blocks[i+3]
        if arg not in self.blocks:
            if_false = self.lfunc.append_basic_block("IF_FALSE_%d" % i)
            self.blocks[arg]=if_false
        else:
            if_false = self.blocks[arg]
        arg1 = self.stack.pop(-1)
        self.builder.cbranch(arg1.llvm(), cont, if_false)

    def op_CALL_FUNCTION(self, i, op, arg):
        # number of arguments is arg
        args = [self.stack[-i] for i in range(arg,0,-1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop(-1)
        ret_typ = None
        if __debug__:
            print("op_CALL_FUNCTION():", func)
        if func.val in PY_CALL_TO_LLVM_CALL_MAP:
            res, ret_typ = PY_CALL_TO_LLVM_CALL_MAP[func.val](self, args)
        elif isinstance(func.val, MethodReference):
            res, ret_val = PY_CALL_TO_LLVM_CALL_MAP[func.val.py_method](
                self, [func.val.object_var])
        else:
            func, args = self.func_resolve_type(func, args)
            if func is None: # A delayed-result (i.e. range or xrange)
                res = args
            else:
                res = self.builder.call(func, args)
        self.stack.append(Variable(res, exact_typ = ret_typ))

    def op_GET_ITER(self, i, op, arg):
        iterable = self.stack[-1].val
        if isinstance(iterable, DelayedObj):
            # This is a dirty little hack since we are not popping the
            # iterable off the stack, and pushing an iterator value
            # on.  Instead, we're going to branch to a synthetic
            # basic block, and hope there is a FOR_ITER to handle this
            # mess.
            self.stack.append(iterable.get_start())
            iter_local = None
            block_writers = self.cfg.blocks_writer[self.crnt_block]
            for local_index, instr_index in block_writers.iteritems():
                if instr_index == i:
                    iter_local = local_index
                    break
            assert iter_local is not None, "Internal compiler error!"
            self.op_STORE_FAST(i, None, iter_local)
            self.builder.branch(self.blocks[i + 4])
        else:
            raise NotImplementedError(
                "Numba can not currently handle iteration over anything other "
                "than range, xrange, or arange (got %r)." % (iterable,))

    def op_DUP_TOPX(self, i, op, arg):
        self.stack.extend(self.stack[-arg:])

    def op_ROT_THREE(self, i, op, arg):
        A, B, C = self.stack[-3:]
        self.stack.extend([C, A, B])

    def op_FOR_ITER(self, i, op, arg):
        iterable = self.stack[-1].val
        # Note that we don't actually generate any code here when
        # rewriting a simple for loop.  Code generation is deferred to
        # the STORE_FAST that should immediately follow this FOR_ITER
        # (we need to know the phi node for the iteration local).
        if isinstance(iterable, DelayedObj):
            self.pending_blocks[i] = i + arg + 3
        else:
            raise NotImplementedError(
                "Numba can not currently handle iteration over anything other "
                "than range, xrange, or arange (got %r)." % (iterable,))

    def op_SETUP_LOOP(self, i, op, arg):
        self.loop_stack.append((i, arg))
        if (i + 3) not in self.blocks:
            loop_entry = self.lfunc.append_basic_block("LOOP_%d" % i)
            self.blocks[i+3] = loop_entry
            # Connect blocks up if this was not an anticipated change
            # in the basic block structure.
            predecessor = self.builder.block
            self.builder.position_at_end(predecessor)
            self.builder.branch(loop_entry)
            self.builder.position_at_end(loop_entry)
        else:
            loop_entry = self.blocks[i+3]

    def op_LOAD_ATTR(self, i, op, arg):
        objarg = self.stack.pop(-1)
        if __debug__:
            print "op_LOAD_ATTR():", i, op, self.names[arg], objarg, objarg.typ
        if objarg.is_module():
            res = getattr(objarg.val, self.names[arg])
        else:
            # Make this a map on types in the future (thinking this is
            # what typemap was destined to do...)
            objarg_llvm_val = objarg.llvm()
            res = None
            if __debug__:
                print "op_LOAD_ATTR():", objarg_llvm_val.type
            if objarg_llvm_val.type == _numpy_array:
                field_index = _numpy_array_field_ofs[self.names[arg]]
                field_indices = [_int32_zero,
                                 lc.Constant.int(_int32, field_index)]
            elif objarg_llvm_val.type in (_complex64, _complex128):
                # Re-wrap the instance, if we had to convert to an LLVM value.
                objarg = objarg if objarg._llvm else Variable(objarg_llvm_val)
                field_name = self.names[arg]
                if field_name == 'real':
                    res = self.builder.extract_value(objarg_llvm_val, 0)
                elif field_name == 'imag':
                    res = self.builder.extract_value(objarg_llvm_val, 1)
                elif objarg_llvm_val.type == _complex64:
                    res = MethodReference(objarg,
                                          getattr(np.complex64, field_name))
                else:
                    res = MethodReference(objarg,
                                          getattr(np.complex128, field_name))
            else:
                raise NotImplementedError(
                    'LOAD_ATTR does not presently support LLVM type %r.' %
                    (str(objarg_llvm_val.type),))
            if res is None:
                res_addr = self.builder.gep(objarg_llvm_val, field_indices)
                res = self.builder.load(res_addr)
        self.stack.append(Variable(res))

    def op_JUMP_ABSOLUTE(self, i, op, arg):
        self.builder.branch(self.blocks[arg])

    def op_POP_BLOCK(self, i, op, arg):
        self.loop_stack.pop(-1)

    def op_JUMP_FORWARD(self, i, op, arg):
        target_i = i + arg + 3
        if target_i not in self.blocks:
            target = self.lfunc.append_basic_block("TARGET_%d" % target_i)
            self.blocks[target_i] = target
        else:
            target = self.blocks[target_i]
        self.builder.branch(target)

    def op_UNPACK_SEQUENCE(self, i, op, arg):
        objarg = self.stack.pop(-1)
        if isinstance(objarg.val, tuple):
            raise NotImplementedError("FIXME")
        else:
            objarg_llvm_val = objarg.llvm()
            # FIXME: Is there some type checking we can do so a bad call
            # to getelementptr doesn't kill the whole process (assuming
            # asserts are live in LLVM)?!
            llvm_vals = [
                self.builder.load(
                    self.builder.gep(objarg_llvm_val,
                                     [lc.Constant.int(_int32, index)]))
                for index in xrange(arg)]
            llvm_vals.reverse()
            for llvm_val in llvm_vals:
                self.stack.append(Variable(llvm_val))

    def _get_index_as_llvm_value(self, index, index_ltyp = None):
        if index_ltyp is None:
            index_ltyp = _intp
        if isinstance(index, Variable):
            ret_val = index.llvm(llvmtype_to_strtype(index_ltyp),
                                 builder = self.builder)
        else:
            ret_val = lc.Constant.int(index_ltyp, int(index))
        return ret_val

    def _build_index_lval(self, arr_lval, index_var):
        '''Given an LLVM pointer to a Numpy array and a Numba index
        variable (may either be representable as an LLVM i32, or a
        tuple of Numba index variables for n-dimensional arrays),
        build the necessary calculations for an index into the data
        array.'''
        if index_var.typ == 'tuple':
            if len(index_var.val) < 1:
                raise NotImplementedError("Indexing by a tuple of size zero!")
            strides = self.builder.load(
                self.builder.gep(
                    arr_lval,
                    [_int32_zero,
                     lc.Constant.int(_int32,
                                     _numpy_array_field_ofs['strides'])]))
            index_lval = None
            index_ltyp = strides.type.pointee
            for stride_index, dim_index in enumerate(index_var.val[:-1]):
                dim_index_lval = self._get_index_as_llvm_value(dim_index,
                                                               index_ltyp)
                stride_lval = self.builder.load(
                    self.builder.gep(
                        strides, [lc.Constant.int(_int32, stride_index)]))
                dim_lval = self.builder.mul(dim_index_lval, stride_lval)
                if index_lval is None:
                    index_lval = dim_lval
                else:
                    index_lval = self.builder.add(index_lval, dim_lval)
            ret_val = (index_lval, self._get_index_as_llvm_value(
                    index_var.val[-1], index_ltyp))
        else:
            try:
                ret_val = (None, index_var.llvm('i32', builder = self.builder))
            except:
                raise NotImplementedError('Index value calculation for %r' %
                                          (index_var,))
        return ret_val

    def _build_pointer_into_arr_data(self, arr_var, index_var):
        lval = arr_var._llvm
        ltype = lval.type
        assert ((arr_var.typ is not None) and (arr_var.typ.startswith('arr[')))
        byte_ofs, index_lval = self._build_index_lval(lval, index_var)
        data_ofs = _numpy_array_field_ofs['data']
        result_type = str_to_llvmtype(arr_var.typ[4:-1])
        data_ptr = self.builder.load(
            self.builder.gep(lval,
                             [_int32_zero,
                              lc.Constant.int(_int32, data_ofs)]))
        if byte_ofs:
            data_ptr = self.builder.gep(data_ptr, [byte_ofs])
        ret_val = self.builder.gep(
            self.builder.bitcast(data_ptr,
                                 lc.Type.pointer(result_type)),
            [index_lval])
        return ret_val

    def op_BINARY_SUBSCR(self, i, op, arg):
        index_var = self.stack.pop(-1)
        arr_var = self.stack.pop(-1)
        result_val = None
        if arr_var._llvm is not None:
            lval = arr_var._llvm
            ltype = lval.type
            if ltype == _numpy_array:
                if __debug__:
                    print("op_BINARY_SUBSCR(): arr_var.typ = %s" %
                          (arr_var.typ,))
                result_val = self.builder.load(
                    self._build_pointer_into_arr_data(arr_var, index_var))
                if __debug__:
                    print result_val
            elif ltype.kind == lc.TYPE_POINTER:
                result_val = self.builder.load(
                    self.builder.gep(lval, [index_var.llvm(
                                'i32', builder = self.builder)]))
            else:
                if __debug__:
                    print("op_BINARY_SUBSCR(): %r arr_var = %r (%r)\n%s" %
                              ((i, op, arg), arr_var, str(arr_var._llvm),
                               self.mod))
                raise NotImplementedError(
                    "Unable to handle indexing on LLVM type '%s'." %
                    (str(ltype),))
        elif isinstance(arr_var.val, tuple):
            raise NotImplementedError("FIXME")
        else:
            if __debug__:
                print("op_BINARY_SUBSCR(): %r arr_var = %r (%r)\n%s" %
                          ((i, op, arg), arr_var, str(arr_var._llvm),
                           self.mod))
            raise NotImplementedError("Unable to handle indexing on objects "
                                      "of type %r." % (type(arr_var.val),))
        self.stack.append(Variable(result_val))

    def op_BUILD_TUPLE(self, i, op, arg):
        tval = tuple(self.stack[-arg:])
        del self.stack[-arg:]
        self.stack.append(Variable(tval))

    def op_STORE_SUBSCR(self, i, op, arg):
        index_var = self.stack.pop(-1)
        arr_var = self.stack.pop(-1)
        store_var = self.stack.pop(-1)
        if __debug__:
            print "op_STORE_SUBSCR():", i, op, arg
            print("op_STORE_SUBSCR(): %r[%r] = %r" % (arr_var, index_var,
                                                      store_var))
        if arr_var._llvm is not None:
            arr_lval = arr_var._llvm
            arr_ltype = arr_lval.type
            if __debug__:
                print("op_STORE_SUBSCR(): arr_lval = '%s', arr_ltype = '%s'" %
                      (arr_lval, arr_ltype))
            if arr_ltype == _numpy_array:
                target_addr = self._build_pointer_into_arr_data(arr_var,
                                                                index_var)
                self.builder.store(
                    # Note: Using type checks in
                    # _build_pointer_into_arr_data() to ensure the
                    # following slice on arr_var.typ is valid.
                    store_var.llvm(arr_var.typ[4:-1], builder = self.builder),
                    target_addr)
            elif arr_ltype.kind == lc.TYPE_POINTER:
                # FIXME: The following implementation is easy enough
                # that I'm writing it out, but this allows normally
                # invalid statements like "arr.shape[1] = 42" to
                # actually work.
                self.builder.store(
                    store_var.llvm(llvmtype_to_strtype(arr_ltype.pointee),
                                   builder = self.builder),
                    self.builder.gep(
                        lval, [index_var.llvm('i32', builder = self.builder)]))
            else:
                raise NotImplementedError("Unable to handle indexing on LLVM "
                                          "type '%s'." % (str(arr_ltype),))
        else:
            raise NotImplementedError('Indexing on objects/values of type %r.' %
                                      (type(arr_var.val),))

    def op_POP_TOP(self, i, op, arg):
        self.stack.pop(-1)
