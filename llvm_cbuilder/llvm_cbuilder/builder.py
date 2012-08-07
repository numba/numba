#
# TODO: Add support for vector.
#

import contextlib
import llvm.core as lc
import llvm.ee as le
from llvm import LLVMException
from . import shortnames as types

def _is_int(ty):
    return isinstance(ty, lc.IntegerType)

def _is_real(ty):
    tys = [ lc.Type.float(),
            lc.Type.double(),
            lc.Type.x86_fp80(),
            lc.Type.fp128(),
            lc.Type.ppc_fp128() ]
    return any(ty == x for x in tys)

def _is_pointer(ty):
    return isinstance(ty, lc.PointerType)

def _is_block_terminated(bb):
    instrs = bb.instructions
    return len(instrs) > 0 and instrs[-1].is_terminator

def _is_cstruct(ty):
    try:
        return issubclass(ty, CStruct)
    except TypeError:
        return False

def _list_values(iterable):
    return [i.value for i in iterable]

@contextlib.contextmanager
def _change_block_temporarily(builder, bb):
    origbb = builder.basic_block
    builder.position_at_end(bb)
    yield
    builder.position_at_end(origbb)

@contextlib.contextmanager
def _change_block_temporarily_dummy(*args):
    yield

class _IfElse(object):
    def __init__(self, parent, cond):
        self.parent = parent
        self.cond = cond
        self._to_close = []

    @contextlib.contextmanager
    def then(self):
        self._bbif = self.parent.function.append_basic_block('if.then')
        self._bbelse = self.parent.function.append_basic_block('if.else')

        builder = self.parent.builder
        builder.cbranch(self.cond.value, self._bbif, self._bbelse)

        builder.position_at_end(self._bbif)

        yield

        self._to_close.extend([self._bbif, self._bbelse, builder.basic_block])

    @contextlib.contextmanager
    def otherwise(self):
        builder = self.parent.builder
        builder.position_at_end(self._bbelse)
        yield
        self._to_close.append(builder.basic_block)

    def close(self):
        self._to_close.append(self.parent.builder.basic_block)
        bbend = self.parent.function.append_basic_block('if.end')
        builder = self.parent.builder
        for bb in self._to_close:
            if not _is_block_terminated(bb):
                with _change_block_temporarily(builder, bb):
                    builder.branch(bbend)
        builder.position_at_end(bbend)

class _Loop(object):
    def __init__(self, parent):
        self.parent = parent

    @contextlib.contextmanager
    def condition(self):
        builder = self.parent.builder
        self._bbcond = self.parent.function.append_basic_block('loop.cond')
        self._bbbody = self.parent.function.append_basic_block('loop.body')
        self._bbend = self.parent.function.append_basic_block('loop.end')

        builder.branch(self._bbcond)

        builder.position_at_end(self._bbcond)

        def setcond(cond):
            builder.cbranch(cond.value, self._bbbody, self._bbend)

        yield setcond

    @contextlib.contextmanager
    def body(self):
        builder = self.parent.builder
        builder.position_at_end(self._bbbody)

        yield self

        if not _is_block_terminated(builder.basic_block):
            builder.branch(self._bbcond)

    def break_loop(self):
        self.parent.builder.branch(self._bbend)

    def continue_loop(self):
        self.parent.builder.branch(self._bbcond)

    def close(self):
        builder = self.parent.builder
#        if not _is_block_terminated(builder.basic_block):
#            with _change_block_temporarily(builder, builder.basic_block):
#                builder.branch(self._bbend)
        builder.position_at_end(self._bbend)

class CBuilder(object):
    '''
    A wrapper class for features in llvm-py package
    to allow user to use C-like high-level language contruct easily.
    '''
    def __init__(self, function):
        self.function = function
        self.declare_block = self.function.append_basic_block('decl')
        self.first_body_block = self.function.append_basic_block('body')
        self.builder = lc.Builder.new(self.first_body_block)
        self.target_data = le.TargetData.new(self.function.module.data_layout)

        # prepare arguments
        self.args = []
        for arg in function.args:
            var = self.var(arg.type, arg, name=arg.name)
            self.args.append(var)

    @staticmethod
    def new_function(mod, name, ret, args):
        functype = lc.Type.function(ret, args)
        func = mod.add_function(functype, name=name)
        return CBuilder(func)

    def depends(self, fndecl):
        return CFunc(self, fndecl(self.function.module))

    def printf(self, fmt, *args):
        mod = self.function.module
        functype = lc.Type.function(types.int, [types.char_p], True)
        printf = mod.get_or_insert_function(functype, name='printf')
        ret = self.builder.call(printf, [fmt.value]+_list_values(args))
        return CTemp(self, ret)

    def debug(self, *args):
        type_mapper = {
            'i8' : '%c',
            'i16': '%hd',
            'i32': '%d',
            'i64': '%ld',
            'double': '%e',
        }
        itemsfmt = []
        items = []
        for i in args:
            if isinstance(i, str):
                itemsfmt.append(i.replace('%', '%%'))
            else:
                ty = type_mapper[str(i.type)]
                itemsfmt.append(ty)
                items.append(i)
        fmt = ' '.join(itemsfmt) + '\n'
        return self.printf(self.constant_string(fmt), *items)

    def var(self, ty, value=None, name=''):
        '''
        Only allocate in the first block
        '''
        with _change_block_temporarily(self.builder, self.declare_block):
            is_cstruct = _is_cstruct(ty)
            if is_cstruct:
                cstruct = ty
                ty = ty.llvm_type()
            ptr = self.builder.alloca(ty, name=name)
        # back to the body
        if value is not None:
            if isinstance(value, CValue):
                value = value.value
            elif not isinstance(value, lc.Value):
                value = self.constant(ty, value).value
            self.builder.store(value, ptr)
        if is_cstruct:
            return cstruct(self, ptr)
        else:
            return CVar(self, ptr)

    def var_copy(self, val, name=''):
        return self.var(val.type, val, name=name)

    def array(self, ty, count, name=''):
        if isinstance(count, int) or isinstance(count, lc.Constant):
            contexthelper = _change_block_temporarily
        else:
            contexthelper = _change_block_temporarily_dummy

        with contexthelper(self.builder, self.declare_block):
            is_cstruct = _is_cstruct(ty)
            if is_cstruct:
                cstruct = ty
                ty = ty.llvm_type()
            if isinstance(count, CValue):
                count = count.value
            elif not isinstance(count, lc.Value):
                count = self.constant(types.int, count).value
            ptr = self.builder.alloca_array(ty, count, name=name)
            return CArray(self, ptr)

    def ret(self, val=None):
        retty = self.function.type.pointee.return_type
        if val is not None:
            if val.type != retty:
                errmsg = "Return type mismatch"
                raise TypeError(errmsg)
            self.builder.ret(val.value)
        else:
            if retty != types.void:
                errmsg = "Cannot return void"
                raise TypeError(errmsg)
            self.builder.ret_void()

    @contextlib.contextmanager
    def ifelse(self, cond):
        cb = _IfElse(self, cond)
        yield cb
        cb.close()

    @contextlib.contextmanager
    def loop(self):
        cb = _Loop(self)
        yield cb
        cb.close()

    @contextlib.contextmanager
    def forever(self):
        with self.loop() as loop:
            with loop.condition() as setcond:
                NULL = self.constant_null(types.int)
                setcond( NULL == NULL )
            with loop.body():
                yield loop

    @contextlib.contextmanager
    def for_range(self, *args):
        def check_arg(x):
            if isinstance(x, int):
                return self.constant(types.int, x)
            if not x.is_int:
                raise TypeError(x, "All args must be of integer type.")
            return x

        if len(args) == 3:
            start, stop, step = map(check_arg, args)
        elif len(args) == 2:
            start, stop = map(check_arg, args)
            step = self.constant(start.type, 1)
        elif len(args) == 1:
            stop = check_arg(args[0])
            start = self.constant(stop.type, 0)
            step = self.constant(stop.type, 1)
        else:
            raise TypeError("Invalid # of arguments: 1, 2 or 3")

        idx = self.var_copy(start)
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( idx < stop )
            with loop.body():
                yield loop, idx
                idx += step

    def position_at_end(self, bb):
        self.basic_block = bb
        self.builder.position_at_end(bb)

    def close(self):
        # Close declaration block
        with _change_block_temporarily(self.builder, self.declare_block):
            self.builder.branch(self.first_body_block)

    def constant(self, ty, val):
        if isinstance(ty, lc.IntegerType):
            res = lc.Constant.int(ty, val)
        elif ty == types.float or ty == types.double:
            res = lc.Constant.real(ty, val)
        else:
            raise TypeError("Cannot auto build constant "
                            "from %s and value %s" % (ty, val))
        return CTemp(self, res)

    def constant_null(self, ty):
        res = lc.Constant.null(ty)
        return CTemp(self, res)

    def constant_string(self, string):
        mod = self.function.module
        # TODO
        # Is this a safe way for uniquing identifying string
        name = '.conststr.%x_%x' % (hash(string), len(string))
        content = lc.Constant.stringz(string)
        try:
            globalstr = mod.get_global_variable_named(name)
        except LLVMException:
            globalstr = mod.add_global_variable(content.type, name=name)
            globalstr.initializer = content
        return CTemp(self, globalstr.bitcast(
                                        types.pointer(content.type.element)))


    def get_intrinsic(self, intrinsic_id, tys):
        lfunc = lc.Function.intrinsic(self.function.module, intrinsic_id, tys)
        return CFunc(self, lfunc)

    def get_function_named(self, name):
        m = self.function.module
        func = m.get_function_named(name)
        return CFunc(self, func)

    def is_terminated(self):
        '''
        Is the current basic-block terminated?
        '''
        return _is_block_terminated(self.builder.basic_block)

    def atomic_cmpxchg(self, ptr, old, val, ordering, crossthread=True):
        res = self.builder.atomic_cmpxchg(ptr.value, old.value, val.value,
                                          ordering, crossthread)
        return CTemp(self, res)

    def atomic_xchg(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_xchg(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_add(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_add(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_sub(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_sub(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_and(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_and(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_nand(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_nand(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_or(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_or(ptr.value, val.value,
                                     ordering, crossthread)
        return CTemp(self, res)

    def atomic_xor(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_xor(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_max(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_max(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_min(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_min(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_umax(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_umax(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_umin(self, ptr, val, ordering, crossthread=True):
        res = self.builder.atomic_umin(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_load(self, ptr, ordering, align=1, crossthread=True):
        res = self.builder.atomic_load(ptr.value, ordering, align, crossthread)
        return CTemp(self, res)

    def atomic_store(self, val, ptr, ordering, align=1, crossthread=True):
        res = self.builder.atomic_store(val.value, ptr.value, ordering,
                                        align, crossthread)
        return CTemp(self, res)

    def fence(self, ordering, crossthread=True):
        res = self.builder.fence(ordering, crossthread)
        return CTemp(self, res)

    def alignment(self, ty):
        return self.target_data.abi_alignment(ty)

    def unreachable(self):
        self.builder.unreachable()

class _DeclareCDef(object):
    def __init__(self, cdef):
        self.cdef = cdef

    def __str__(self):
        return self.cdef._name_

    def __call__(self, module):
        try:
            func = self.cdef.define(module)
        except NameError as e:
            (func,) = e
        return func

class CDeclare(object):
    def __init__(self, name, ty, ptr):
        self._name = name
        self._type = ty
        self._ptr = ptr

    def __call__(self, module):
        fnptr = types.pointer(self._type)
        ptr = lc.Constant.int(types.intp, self._ptr)
        ptr = ptr.inttoptr(fnptr)
        return ptr

    def __str__(self):
        return self._name

class CDefinition(CBuilder):
    '''
    Inherit this class to for defining functions
    '''
    _name_ = ''             # name of the function; should overide in subclass
    _retty_  = types.void # return type; can overide in subclass
    _argtys_ = []       # a list of tuple(name, type); can overide in subclass

    def __new__(cls, *args, **kws):
        try:
            fnsp = getattr(cls, 'specialize')
        except AttributeError:
            pass
        else:
            cls = type('%s_Specialized' % cls.__name__, (cls,), {})
            fnsp(*args, **kws)

        obj = object.__new__(_DeclareCDef)
        obj.__init__(cls)
        return obj

    @classmethod
    def define(cls, module):
        functype = lc.Type.function(cls._retty_, [v for k, v in cls._argtys_])
        name = cls._name_
        if not name:
            raise NameError("Function name cannot be empty.")
        func = module.get_or_insert_function(functype, name=name)
        if not func.is_declaration: # already defined?
            raise NameError(func)

        # Name all arguments
        for i, (name, _) in enumerate(cls._argtys_):
            func.args[i].name = name

        # Create builder and populate body
        cbuilder = object.__new__(cls)
        cbuilder.__init__(func)
        cbuilder.body(*cbuilder.args)
        cbuilder.close()
        return func

    def body(self):
        '''
        Overide this function to define the body.
        '''
        raise NotImplementedError

class CValue(object):
    '''
    = Signess =
    Since LLVM type does not provide signess attribute.  This information
    is provided in the CValue.unsigned attribute.  The default value is
    `None`, meaning that this attribute is not set.

    In casting operation, signess information is passed as an optional arg.

    In binary operation, signess of the left operand is used.
    '''

    # Attribute for for integer values.
    unsigned = None

    _BINOP_MAP = {
        # op-name : (signed int, unsigned int, real)
        'add'    : (lc.Builder.add,  lc.Builder.add,  lc.Builder.fadd),
        'sub'    : (lc.Builder.sub,  lc.Builder.sub,  lc.Builder.fsub),
        'mul'    : (lc.Builder.mul,  lc.Builder.mul,  lc.Builder.fmul),
        'div'    : (lc.Builder.sdiv, lc.Builder.udiv, lc.Builder.fdiv),
        'mod'    : (lc.Builder.srem, lc.Builder.urem, lc.Builder.frem),
    }

    _BITWISE_MAP = {
        # op-name : (signed int, unsigned int)
        'lshift' : (lc.Builder.shl,  lc.Builder.shl),
        'rshift' : (lc.Builder.lshr, lc.Builder.ashr),
        'and'    : (lc.Builder.and_, lc.Builder.and_),
        'or'     : (lc.Builder.or_,  lc.Builder.or_),
        'xor'    : (lc.Builder.xor,  lc.Builder.xor),
    }

    _CMP_MAP = {
        # op-name : (signed int, unsigned int, real)
        'eq' : (lc.ICMP_EQ,  lc.ICMP_EQ,  lc.FCMP_OEQ),
        'ne' : (lc.ICMP_NE,  lc.ICMP_NE,  lc.FCMP_ONE),
        'lt' : (lc.ICMP_SLT, lc.ICMP_ULT, lc.FCMP_OLT),
        'le' : (lc.ICMP_SLE, lc.ICMP_ULE, lc.FCMP_OLE),
        'gt' : (lc.ICMP_SGT, lc.ICMP_UGT, lc.FCMP_OGT),
        'ge' : (lc.ICMP_SGE, lc.ICMP_UGE, lc.FCMP_OGE),
    }

    def __init__(self, parent):
        self.parent = parent

    def _use_binop(self, op):
        def wrapped(rhs):
            self._ensure_same_type(rhs)
            binop = self._BINOP_MAP[op]
            if self.is_int:
                if not self.unsigned:
                    idx = 0
                else:
                    idx = 1
            elif self.is_real:
                idx = 2
            else:
                errmsg = "Binary operation %s does not support type %s"
                raise TypeError(errmsg % (op, self.type))
            res = binop[idx](self.parent.builder, self.value, rhs.value)
            return CTemp(self.parent, res)
        return wrapped

    def _use_bitwise(self, op):
        def wrapped(rhs):
            self._ensure_same_type(rhs)
            if not self.is_int:
                errmsg = "Bitwise operation %s does not support type %s"
                raise TypeError(op, self.type)
            if not self.unsigned:
                idx = 0
            else:
                idx = 1
            res = self._BITWISE_MAP[idx](self.parent.builder,
                                       self.value, rhs.value)
            return CTemp(self.parent, res)
        return wrapped

    def __add__(self, rhs):
        return self._use_binop('add')(rhs)

    def __sub__(self, rhs):
        return self._use_binop('sub')(rhs)

    def __mul__(self, rhs):
        return self._use_binop('mul')(rhs)

    def __div__(self, rhs):
        return self._use_binop('div')(rhs)

    def __truediv__(self, rhs):
        return self.__div__(rhs)

    def __mod__(self, rhs):
        return self._use_binop('mod')(rhs)

    def __lshift__(self, rhs):
        return self._use_bitwise('lshift')(rhs)

    def __rshift__(self, rhs):
        return self._use_bitwise('rshift')(rhs)

    def __and__(self, rhs):
        return self._use_bitwise('and')(rhs)

    def __or__(self, rhs):
        return self._use_bitwise('or')(rhs)

    def __xor__(self, rhs):
        return self._use_bitwise('xor')(rhs)

    def _ensure_same_type(self, val):
        if self.type != val.type:
            errmsg = "Type mismatch: %s != %s"
            raise TypeError(errmsg % (self.type, val.type))

    @property
    def is_int(self):
        return _is_int(self.type)

    @property
    def is_real(self):
        return _is_real(self.type)

    def cast(self, ty, unsigned=False):
        make = lambda X: CTemp(self.parent, X)
        if self.type == ty:
            return self # pass thru
        elif self.is_pointer and _is_pointer(ty):
            builder = self.parent.builder
            return make(builder.bitcast(self.value, ty))
        elif self.is_int:
            if _is_int(ty):
                if self.type.width < ty.width:
                    if not unsigned:
                        return make(self.parent.builder.sext(self.value, ty))
                    else:
                        return make(self.parent.builder.zext(self.value, ty))
                else:
                    return make(self.parent.builder.trunc(self.value, ty))
            elif _is_real(ty):
                if not unsigned:
                    return make(self.parent.builder.sitofp(self.value, ty))
                else:
                    return make(self.parent.builder.uitofp(self.value, ty))
        elif self.is_real:
            if not unsigned:
                return make(self.parent.builder.fptosi(self.value, ty))
            else:
                return make(self.parent.builder.fptoui(self.value, ty))

        errmsg = "Cast from %s to %s is not possible."
        raise TypeError(errmsg % (self.type, ty))

    def _cmp_op(self, name):
        def wrapped(rhs):
            make = lambda X: CTemp(self.parent, X)
            self._ensure_same_type(rhs)

            flag_bag = self._CMP_MAP[name]
            if self.is_int:
                comparator = self.parent.builder.icmp
                if not self.unsigned:
                    flag = flag_bag[0]
                else:
                    flag = flag_bag[1]
            elif self.is_real:
                comparator = self.parent.builder.fcmp
                flag = flag_bag[2]
            else:
                errmsg = "Comparision between %s and %s is not supported."
                raise TypeError(errmsg % (self.type, rhs.type))

            return CTemp(self.parent, comparator(flag, self.value, rhs.value))
        return wrapped

    def __eq__(self, rhs):
        return self._cmp_op('eq')(rhs)

    def __ne__(self, rhs):
        return self._cmp_op('ne')(rhs)

    def __lt__(self, rhs):
        return self._cmp_op('lt')(rhs)

    def __le__(self, rhs):
        return self._cmp_op('le')(rhs)

    def __gt__(self, rhs):
        return self._cmp_op('gt')(rhs)

    def __ge__(self, rhs):
        return self._cmp_op('ge')(rhs)


    @property
    def is_pointer(self):
        return _is_pointer(self.type)

    def _ensure_is_pointer(self):
        if not self.is_pointer:
            raise TypeError("Must be a pointer; got %s" % self.type)

    def __getitem__(self, idx):
        self._ensure_is_pointer()
        if not isinstance(idx, CValue):
            idx = self.parent.constant(types.int, idx)
        bldr = self.parent.builder
        ptr = bldr.gep(self.value, [idx.value])
        return CVar(self.parent, ptr)

    def load(self, volatile=False):
        self._ensure_is_pointer()
        loaded = self.parent.builder.load(self.value, volatile=volatile)
        return CTemp(self.parent, loaded)

    def store(self, val, volatile=False):
        self._ensure_is_pointer()
        self.parent.builder.store(val.value, self.value, volatile=volatile)

    def atomic_load(self, ordering, align=None, crossthread=True):
        self._ensure_is_pointer()
        if align is None:
            align = self.parent.alignment(self.type.pointee)
        inst = self.parent.builder.atomic_load(self.value, ordering, align,
                                               crossthread=crossthread)
        return CTemp(self.parent, inst)

    def atomic_store(self, value, ordering, align=None,  crossthread=True):
        self._ensure_is_pointer()
        if align is None:
            align = self.parent.alignment(self.type.pointee)
        self.parent.builder.atomic_store(value.ptr, self.value, ordering,
                                         align=align, crossthread=crossthread)

    def atomic_cmpxchg(self, old, new, ordering, crossthread=True):
        self._ensure_is_pointer()
        inst = self.parent.builder.atomic_cmpxchg(self.value, old.value,
                                                  new.value, ordering,
                                                  crossthread=crossthread)
        return CTemp(self.parent, inst)


class CFunc(CValue):
    def __init__(self, parent, func):
        super(CFunc, self).__init__(parent)
        self.function = func

    def __call__(self, *args):
        arg_values = _list_values(args)
        ftype = self.function.type.pointee
        for i, (exp, got) in enumerate(zip(ftype.args, arg_values)):
            if exp != got.type:
                raise TypeError("At call to %s, "
                                "argument %d mismatch: %s != %s"
                                % (self.function.name, i, exp, got.type))
        res = self.parent.builder.call(self.function, arg_values)
        return CTemp(self.parent, res)

    @property
    def value(self):
        return self.function

    @property
    def type(self):
        return self.function.type

class CTemp(CValue):
    def __init__(self, parent, value):
        super(CTemp, self).__init__(parent)
        self.value = value

    @property
    def type(self):
        return self.value.type

class CVar(CValue):

    def __init__(self, parent, ptr):
        super(CVar, self).__init__(parent)
        self.ptr = ptr

    def _inplace_binop(self, op):
        def wrapped(rhs):
            res = self._use_binop(op)(rhs)
            self.assign(res)
            return self
        return wrapped

    def __iadd__(self, rhs):
        return self._inplace_binop('add')(rhs)

    def __isub__(self, rhs):
        return self._inplace_binop('sub')(rhs)

    def __imul__(self, rhs):
        return self._inplace_binop('mul')(rhs)

    def __idiv__(self, rhs):
        return self._inplace_binop('div')(rhs)

    def __imod__(self, rhs):
        return self._inplace_binop('mod')(rhs)

    def _inplace_bitwise(self, op):
        def wrapped(rhs):
            res = self._use_bitwise(op)(rhs)
            self.assign(res)
            return self
        return wrapped

    def __ilshift__(self, rhs):
        return self._inplace_bitwise('lshift')(rhs)

    def __irshift__(self, rhs):
        return self._inplace_bitwise('rshift')(rhs)

    def __iand__(self, rhs):
        return self._inplace_bitwise('and')(rhs)

    def __ior__(self, rhs):
        return self._inplace_bitwise('or')(rhs)

    def __ixor__(self, rhs):
        return self._inplace_bitwise('xor')(rhs)

    @property
    def value(self):
        return self.parent.builder.load(self.ptr)

    def assign(self, val):
        self._ensure_same_type(val)
        self.parent.builder.store(val.value, self.ptr)

    @property
    def type(self):
        return self.ptr.type.pointee

    def reference(self):
        return CTemp(self.parent, self.ptr)

    def as_struct(self, cstruct_class, volatile=False):
        if _is_pointer(self.type):
            ptr = self.parent.builder.load(self.ptr, volatile=volatile)
            return cstruct_class(self.parent, ptr)
        else:
            return cstruct_class(self.parent, self.ptr)

class CArray(CValue):
    def __init__(self, parent, base):
        super(CArray, self).__init__(parent)
        self.base_ptr = base

    @property
    def value(self):
        return self.base_ptr

    def reference(self):
        return CTemp(self.parent, self.base_ptr)

    @property
    def type(self):
        return self.base_ptr.type

## Moved to CValue
#    def __getitem__(self, idx):
#        self._ensure_is_pointer()
#        builder = self.parent.builder
#        if isinstance(idx, CValue):
#            idx = idx.value
#        elif not isinstance(idx, lc.Value):
#            idx = self.parent.constant(types.int, idx).value
#        ptr = builder.gep(self.value, [idx])
#        return CVar(self.parent, ptr)

class CStruct(CValue):
    @classmethod
    def llvm_type(cls):
        return lc.Type.struct([v for k, v in cls._fields_])

    def __init__(self, parent, ptr):
        super(CStruct, self).__init__(parent)
        makeind = lambda x: self.parent.constant(types.int, x).value
        self.ptr = ptr
        for i, (fd, _) in enumerate(self._fields_):
            gep = self.parent.builder.gep(ptr, [makeind(0), makeind(i)])
            gep.name = "%s.%s" % (type(self).__name__, fd)
            setattr(self, fd, CVar(self.parent, gep))

    def reference(self):
        return CTemp(self.parent, self.ptr)

class CExternal(object):
    def __init__(self, cbuilder):
        is_func = lambda x: isinstance(x, lc.FunctionType)
        non_magic = lambda s: not ( s.startswith('__') and s.endswith('__') )

        to_declare = []
        for fname in filter(non_magic, vars(type(self))):
            ftype = getattr(self, fname)
            if is_func(ftype):
                to_declare.append((fname, ftype))

        mod = cbuilder.function.module
        for fname, ftype in to_declare:
            func = mod.get_or_insert_function(ftype, name=fname)
            if func.type.pointee != ftype:
                raise NameError("Function has already been declared "
                                "with a different type: %s != %s"
                                % (func.type, ftype) )
            setattr(self, fname, CFunc(cbuilder, func))

