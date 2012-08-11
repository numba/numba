import opcode
import sys
import types
import __builtin__

import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le

if sys.maxint > 2**33:
    _plat_bits = 64
else:
    _plat_bits = 32


_pyobject_head = [lc.Type.int(_plat_bits), lc.Type.pointer(lc.Type.int(32))]
if hasattr(sys, 'getobjects'):
    _trace_refs_ = True
    _pyobject_head = [lc.Type.pointer(lc.Type.int(32)),
                      lc.Type.pointer(lc.Type.int(32))] + \
                      _pyobject_head
else:
    _trace_refs_ = False

_head_len = len(_pyobject_head)
_intp = lc.Type.int(_plat_bits)
_intp_star = lc.Type.pointer(_intp)
_void_star = lc.Type.pointer(lc.Type.int(8))
_numpy_struct = lc.Type.struct(_pyobject_head+\
      [_void_star,          # data
       lc.Type.int(32),     # nd
       _intp_star,          # dimensions
       _intp_star,          # strides
       _void_star,          # base
       _void_star,          # descr
       lc.Type.int(32),     # flags
       _void_star,          # weakreflist 
       _void_star,          # maskna_dtype
       _void_star,          # maskna_data
       _intp_star,          # masna_strides
      ])
_numpy_array = lc.Type.pointer(_numpy_struct)

# Translate Python bytecode to LLVM IR

#hasconst
#hasname
#hasjrel
#haslocal
#hascompare
#hasfree

from translate import itercode

# Convert llvm Type object to kind-bits string
def llvmtype_to_strtype(typ):
    if typ.kind == lc.TYPE_FLOAT:
        return 'f32'
    elif typ.kind == lc.TYPE_DOUBLE:
        return 'f64'
    elif typ.kind == lc.TYPE_INTEGER:
        return 'i%d' % typ.width
    elif typ.kind == lc.TYPE_POINTER and \
         typ.pointee.kind == lc.TYPE_FUNCTION:
        return ['func'] + typ.pointee.args
    elif typ.kind == lc.TYPE_POINTER and \
        typ.pointee.kind == lc.TYPE_STRUCT:
        return ['arr[]']

# We don't support all types....
def pythontype_to_strtype(typ):
    if issubclass(typ, float):
        return 'f64'
    elif issubclass(typ, int):
        return 'i%d' % _plat_bits
    elif isinstance(typ, (types.BuiltinFunctionType, types.FunctionType)):
        return ["func"]
    elif isinstance(typ, numpy.dtype):
        return 'arr[%s%d]' % (typ.dtype, typ.itemsize*8)

def map_to_function(func, typs, mod):
    typs = [str_to_llvmtype(x) if isinstance(x, str) else x for x in typs]
    INTR = getattr(lc, 'INTR_%s' % func.__name__.upper())
    return lc.Function.intrinsic(mod, INTR, typs)

class DelayedObj(object):
    def __init__(self, base, args):
        self.base = base
        self.args = args


# Variables placed on the stack. 
#  They allow an indirection
#  So, that when used in an operation, the correct
#  LLVM type can be inserted.  
class Variable(object):
    def __init__(self, val):
        if isinstance(val, Variable):
            self.val = val.val
            self._llvm = val._llvm
            self.typ = val.typ
            return 
        self.val = val
        if isinstance(val, lc.Value):
            self._llvm = val
            self.typ = llvmtype_to_strtype(val.type)
        else:
            self._llvm = None
            if isinstance(val, numpy.ndarray):
                self.typ = pythontype_to_strtype(val.dtype)
            else:
                self.typ = pythontype_to_strtype(type(val))
    
    def llvm(self, typ=None, mod=None):
        if self._llvm:
            if typ is not None and typ != self.typ:
                raise ValueError, "type mismatch"
            return self._llvm
        else:
            if typ is None:
                typ = 'f64'
            if typ == 'f64':
                res = lc.Constant.real(lc.Type.double(), float(self.val))
            elif typ == 'f32':
                res = lc.Constant.real(lc.Type.float(), float(self.val))
            elif typ[0] == 'i':
                res = lc.Constant.int(lc.Type.int(int(typ[1:])), 
                                      int(self.val))
            elif typ[0] == 'func':
                res = map_to_function(self.val, typ[1:], mod)
            elif typ[:3] == 'arr':
                pass
            return res

# Add complex, unsigned, and bool 
def str_to_llvmtype(str):
    if str[0] == 'f':
        if str[1:] == '32':
            return lc.Type.float()
        elif str[1:] == '64':
            return lc.Type.double()
    elif str[0] == 'i':
        num = int(str[1:])
        return lc.Type.int(num)
    raise TypeError, "Invalid Type"

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

# Both inputs are Variable objects
#  Resolves types on one of them. 
#  Won't work if both need resolving
#  Does not up-cast llvm types
def resolve_type(arg1, arg2):
    if arg1._llvm is not None:
        typ = arg1.typ
    elif arg2._llvm is not None:
        typ = arg2.typ
    else:
        raise TypeError, "Both types not valid"
                
    return typ, arg1.llvm(typ), arg2.llvm(typ)

# This won't convert any llvm types.  It assumes 
#  the llvm types in args are either fixed or not-yet specified.
def func_resolve_type(mod, func, args):
    # already an llvm function
    if func.val and func.val is func._llvm:
        typs = [llvmtype_to_str(x) for x in func._llvm.type.pointee.args]
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

class Translate(object):
    def __init__(self, func):
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
                self._myglobals[name] = __builtin__.__getattribute__(name)

        self.mod = lc.Module.new(func.func_name+'_mod')        

        self._delaylist = [range, xrange, enumerate]
        self.setup_func()

    def setup_func(self):
        # XXX: Fix the typing here.
        double = lc.Type.double()
        # The return type will not be known until the return
        #   function is created.   So, we will need to 
        #   walk through the code twice....
        #   Once to get the type of the return, and again to 
        #   emit the instructions. 
        #   Or, we assume the function has been called already
        #   and the return type is transknown and passed in. 
        self.ret_ltype = double
        # The arg_ltypes we will be able to get from what is passed in
        argnames = self.fco.co_varnames[:self.fco.co_argcount]
        self.arg_ltypes = [double for arg in argnames]
        ty_func = lc.Type.function(self.ret_ltype, self.arg_ltypes)        
        self.lfunc = self.mod.add_function(ty_func, self.func.func_name)
        self._locals = [None]*len(self.fco.co_varnames)
        for i, name in enumerate(argnames):
            self.lfunc.args[i].name = name
            # Store away arguments in locals
            self._locals[i] = self.lfunc.args[i]
        entry = self.lfunc.append_basic_block('Entry')
        self.blocks = {0:entry}
        self.stack = []

    def translate(self):
        """Translate the function
        """
        for i, op, arg in itercode(self.costr):
            name = opcode.opname[op]
            # Change the builder if the line-number 
            # is in the list of blocks. 
            if i in self.blocks.keys():
                self.builder = lc.Builder.new(self.blocks[i])
            getattr(self, 'op_'+name)(i, op, arg)
    
        # Perform code optimization
        #fpm = lp.FunctionPassManager.new(self.mod)
        #fpm.initialize()
        #fpm.add(lp.PASS_DEAD_CODE_ELIMINATION)
        #fpm.run(self.lfunc)
        #fpm.finalize()

    def make_ufunc(self, name=None):
        import llvm._core as core
        ee = le.ExecutionEngine.new(self.mod)
        if name is None:
            name = self.func.func_name
        return core.make_ufunc(ee.get_pointer_to_function(self.lfunc), 
                               name)

    # This won't convert any llvm types.  It assumes 
    #  the llvm types in args are either fixed or not-yet specified.
    def func_resolve_type(self, func, args):
        # already an llvm function
        if func.val and func.val is func._llvm:
            typs = [llvmtype_to_str(x) for x in func._llvm.type.pointee.args]
            lfunc = func._llvm
        # The function is one of the delayed list
        elif func.val in self._delaylist:
            return None, DelayedObj(func.val, args)
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
            lfunc = map_to_function(func.val, typs, self.mod)

        llvm_args = [arg.llvm(typ) for typ, arg in zip(typs, args)]
        return lfunc, llvm_args


    def op_LOAD_FAST(self, i, op, arg):
        self.stack.append(Variable(self._locals[arg]))

    def op_STORE_FAST(self, i, op, arg):
        self._locals[arg] = self.stack.pop(-1)

    def op_LOAD_GLOBAL(self, i, op, arg):
        self.stack.append(Variable(self._myglobals[self.names[arg]]))

    def op_LOAD_CONST(self, i, op, arg):
        const = Variable(self.constants[arg])
        self.stack.append(const)        
    
    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2)
        if typ[0] == 'f':
            res = self.builder.fadd(arg1, arg2)
        else: # typ[0] == 'i'
            res = self.builder.add(arg1, arg2)
        self.stack.append(Variable(res))

    def op_BINARY_SUBTRACT(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2)
        if typ[0] == 'f':
            res = self.builder.fsub(arg1, arg2)
        else: # typ[0] == 'i'
            res = self.builder.sub(arg1, arg2)
        self.stack.append(Variable(res))
    
    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2)
        if typ[0] == 'f':
            res = self.builder.fmul(arg1, arg2)
        else: # typ[0] == 'i'
            res = self.builder.mul(arg1, arg2)
        self.stack.append(Variable(res))

    def op_BINARY_DIVIDE(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2)
        if typ[0] == 'f':
            res = self.builder.fdiv(arg1, arg2)
        else: # typ[0] == 'i'
            res = self.builder.sdiv(arg1, arg2)
            # XXX: FIXME-need udiv as
        self.stack.append(Variable(res))

    def op_BINARY_MODULO(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2)
        if typ[0] == 'f':
            res = self.builder.frem(arg1, arg2)
        else: # typ[0] == 'i'
            res = self.builder.srem(arg1, arg2)
            # FIXME:  Add urem
        self.stack.append(Variable(res))

    def op_RETURN_VALUE(self, i, op, arg):
        val = self.stack.pop(-1)
        if val.val is None:
            self.builder.ret(lc.Constant.real(self.ret_ltype, 0))
        else:
            self.builder.ret(val.llvm())
        # Add a new block at the next instruction if not at end
        if i+1 < len(self.costr) and i+1 not in self.blocks.keys():
            blk = self.lfunc.append_basic_block("RETURN_%d" % i)
            self.blocks[i+1] = blk


    def op_COMPARE_OP(self, i, op, arg):
        cmpop = opcode.cmp_op[arg]
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        typ, arg1, arg2 = resolve_type(arg1, arg2)
        if typ[0] == 'f':
            res = self.builder.fcmp(_compare_mapping_float[cmpop], 
                                    arg1, arg2)
        else: # integer FIXME: need unsigned as well...
            res = self.builer.icmp(_compare_mapping_sint[cmpop], 
                                    arg1, arg2)
        self.stack.append(Variable(res))

    def op_POP_JUMP_IF_FALSE(self, i, op, arg):
        # We need to create two blocks.
        #  One for the next instruction (just past the jump)
        #  and another for the block to be jumped to.
        cont = self.lfunc.append_basic_block("CONT_%d"% i )
        if_false = self.lfunc.append_basic_block("IF_FALSE_%d" % i)
        self.blocks[i+3]=cont
        self.blocks[arg]=if_false
        arg1 = self.stack.pop(-1)
        self.builder.cbranch(arg1.llvm(), cont, if_false)

    def op_CALL_FUNCTION(self, i, op, arg):
        # number of arguments is arg
        args = [self.stack[-i] for i in range(arg,0,-1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop(-1)
        func, args = self.func_resolve_type(func, args)
        if func is None: # A delayed-result (i.e. range or xrange)
            res = args
        else:
            res = self.builder.call(func, args)
        self.stack.append(Variable(res))

    def op_GET_ITER(self, i, op, arg):
        pass






    


        

