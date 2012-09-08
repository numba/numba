#
# TODO: Add support for vector.
#

import contextlib
import llvm.core as lc
import llvm.ee as le
import llvm.passes as lp
from llvm import LLVMException
from . import shortnames as types

###
#  Utilities
###

class FunctionAlreadyExists(NameError):
    pass

def _is_int(ty):
    return isinstance(ty, lc.IntegerType)

def _is_real(ty):
    tys = [ lc.Type.float(),
            lc.Type.double(),
            lc.Type.x86_fp80(),
            lc.Type.fp128(),
            lc.Type.ppc_fp128() ]
    return any(ty == x for x in tys)

def _is_vector(ty, of=None):
    if isinstance(ty, lc.VectorType):
        if of is not None:
            return of(ty.element)
        return True
    else:
        return False

def _is_pointer(ty):
    return isinstance(ty, lc.PointerType)

def _is_block_terminated(bb):
    instrs = bb.instructions
    return len(instrs) > 0 and instrs[-1].is_terminator

def _is_struct(ty):
    return isinstance(ty, lc.StructType)

def _is_cstruct(ty):
    try:
        return issubclass(ty, CStruct)
    except TypeError:
        return False

def _list_values(iterable):
    return [i.value for i in iterable]

def _auto_coerce_index(cbldr, idx):
    if not isinstance(idx, CValue):
        idx = cbldr.constant(types.int, idx)
    return idx

@contextlib.contextmanager
def _change_block_temporarily(builder, bb):
    origbb = builder.basic_block
    builder.position_at_end(bb)
    yield
    builder.position_at_end(origbb)

@contextlib.contextmanager
def _change_block_temporarily_dummy(*args):
    yield

class CastError(TypeError):
    def __init__(self, orig, to):
        super(CastError, self).__init__("Cannot cast from %s to %s" % (orig, to))

class _IfElse(object):
    '''if-else construct.

    Example
    -------
    with cbuilder.ifelse(cond) as ifelse:
        with ifelse.then():
            # code when cond is true
            # this block is mandatory
        with ifelse.otherwise():
            # code when cond is false
            # this block is optional
    '''

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
        closed_count = 0
        for bb in self._to_close:
            if not _is_block_terminated(bb):
                with _change_block_temporarily(builder, bb):
                    builder.branch(bbend)
                    closed_count += 1
        builder.position_at_end(bbend)
        if not closed_count:
            self.parent.unreachable()

class _Loop(object):
    '''while...do loop.

    Example
    -------
    with cbuilder.loop() as loop:
        with loop.condition() as setcond:
            # Put the condition evaluation here
            setcond( cond )         # set loop condition
            # Do not put code after setcond(...)
        with loop.body():
            # Put the code of the loop body here

    Use loop.break_loop() to break out of the loop.
    Use loop.continue_loop() to jump the condition evaulation.
    '''

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
        # close last block
        if not _is_block_terminated(builder.basic_block):
            builder.branch(self._bbcond)

    def break_loop(self):
        self.parent.builder.branch(self._bbend)

    def continue_loop(self):
        self.parent.builder.branch(self._bbcond)

    def close(self):
        builder = self.parent.builder
        builder.position_at_end(self._bbend)

class CBuilder(object):
    '''
    A wrapper class for features in llvm-py package
    to allow user to use C-like high-level language contruct easily.
    '''

    def __init__(self, function):
        '''constructor

        function : is an empty function to be populating.
        '''
        self.function = function
        self.declare_block = self.function.append_basic_block('decl')
        self.first_body_block = self.function.append_basic_block('body')
        self.builder = lc.Builder.new(self.first_body_block)
        self.target_data = le.TargetData.new(self.function.module.data_layout)
        self._auto_inline_list = []
        # Prepare arguments. Make all function arguments behave like variables.
        self.args = []
        for arg in function.args:
            var = self.var(arg.type, arg, name=arg.name)
            self.args.append(var)

    @staticmethod
    def new_function(mod, name, ret, args):
        '''factory method

        Create a new function in the module and return a CBuilder instance.
        '''
        functype = lc.Type.function(ret, args)
        func = mod.add_function(functype, name=name)
        return CBuilder(func)

    def depends(self, fndecl):
        '''add function dependency

        Returns a CFunc instance and define the function if it is not defined.

        fndecl : is a callable that takes a `llvm.core.Module` and returns
                 a function pointer.
        '''
        return CFunc(self, fndecl(self.function.module))

    def printf(self, fmt, *args):
        '''printf() from libc

        fmt : a character string holding printf format string.
        *args : additional variable arguments.
        '''
        from .libc import LibC
        libc = LibC(self)
        ret = libc.printf(fmt, *args)
        return CTemp(self, ret)

    def debug(self, *args):
        '''debug print

        Use printf to dump the values of all arguments.
        '''
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
            elif isinstance(i.type, lc.PointerType):
                itemsfmt.append("%p")
                items.append(i)
            else:
                tyname = str(i.type)
                if tyname == 'float':
                    # auto convert float to double
                    ty = '%e'
                    i = i.cast(types.double)
                else:
                    ty = type_mapper[tyname]
                itemsfmt.append(ty)
                items.append(i)
        fmt = ' '.join(itemsfmt) + '\n'
        return self.printf(self.constant_string(fmt), *items)

    def sizeof(self, ty):
        bldr = self.builder
        ptrty = types.pointer(ty)
        first = lc.Constant.null(ptrty)
        second = bldr.gep(first, [lc.Constant.int(types.intp, 1)])

        firstint = bldr.ptrtoint(first, types.intp)
        secondint = bldr.ptrtoint(second, types.intp)
        diff = bldr.sub(secondint, firstint)
        return CTemp(self, diff)

    def min(self, x, y):
        z = self.var(x.type)
        with self.ifelse( x < y ) as ifelse:
            with ifelse.then():
                z.assign(x)
            with ifelse.otherwise():
                z.assign(y)
        return z

    def var(self, ty, value=None, name=''):
        '''allocate variable on the stack

        ty : variable type
        value : [optional] initializer value
        name : [optional] name used in LLVM IR
        '''
        with _change_block_temporarily(self.builder, self.declare_block):
            # goto the first block
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
        '''allocate a new variable by copying another value

        The new variable has the same type and value of `val`.
        '''
        return self.var(val.type, val, name=name)

    def array(self, ty, count, name=''):
        '''allocate an array on the stack

        ty : array element type
        count : array size; can be python int, llvm.core.Constant, or CValue
        name : [optional] name used in LLVM IR
        '''
        if isinstance(count, int) or isinstance(count, lc.Constant):
            # Only go to the first block if array size is fixed.
            contexthelper = _change_block_temporarily
        else:
            # Do not go to the first block if the array size is dynamic.
            contexthelper = _change_block_temporarily_dummy

        with contexthelper(self.builder, self.declare_block):
            if _is_cstruct(ty): # array of struct?
                cstruct = ty
                ty = ty.llvm_type()

            if isinstance(count, CValue):
                count = count.value
            elif not isinstance(count, lc.Value):
                count = self.constant(types.int, count).value

            ptr = self.builder.alloca_array(ty, count, name=name)
            return CArray(self, ptr)

    def ret(self, val=None):
        '''insert return statement

        val : if is `None`, insert return-void
              else, return `val`
        '''
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
        '''start a if-else block

        cond : branch condition
        '''
        cb = _IfElse(self, cond)
        yield cb
        cb.close()

    @contextlib.contextmanager
    def loop(self):
        '''start a loop block
        '''
        cb = _Loop(self)
        yield cb
        cb.close()

    @contextlib.contextmanager
    def forever(self):
        '''start a forever loop block
        '''
        with self.loop() as loop:
            with loop.condition() as setcond:
                NULL = self.constant_null(types.int)
                setcond( NULL == NULL )
            with loop.body():
                yield loop

    @contextlib.contextmanager
    def for_range(self, *args):
        '''start a for-range block.

        *args : same as arguments of builtin `range()`
        '''
        def check_arg(x):
            if isinstance(x, int):
                return self.constant(types.int, x)
            if not isinstance(x, IntegerValue):
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
        '''reposition inserter to the end of basic-block

        bb : a basic block
        '''
        self.basic_block = bb
        self.builder.position_at_end(bb)

    def close(self):
        '''end code generation
        '''
        # Close declaration block
        with _change_block_temporarily(self.builder, self.declare_block):
            self.builder.branch(self.first_body_block)

        # Do the auto inlining
        for callinst in self._auto_inline_list:
            lc.inline_function(callinst)

    def constant(self, ty, val):
        '''create a constant

        ty : data type
        val : initializer
        '''
        if isinstance(ty, lc.IntegerType):
            res = lc.Constant.int(ty, val)
        elif ty == types.float or ty == types.double:
            res = lc.Constant.real(ty, val)
        else:
            raise TypeError("Cannot auto build constant "
                            "from %s and value %s" % (ty, val))
        return CTemp(self, res)

    def constant_null(self, ty):
        '''create a zero filled constant

        ty : data type
        '''
        res = lc.Constant.null(ty)
        return CTemp(self, res)

    def constant_string(self, string):
        '''create a constant string

        This will de-duplication string of same content to minimize memory use.
        '''
        mod = self.function.module
        collision = 0
        name_fmt = '.conststr.%x.%x'
        content = lc.Constant.stringz(string)
        while True:
            name = name_fmt % (hash(string), collision)
            try:
                # check if the name already exists
                globalstr = mod.get_global_variable_named(name)
            except LLVMException:
                # new constant string
                globalstr = mod.add_global_variable(content.type, name=name)
                globalstr.initializer = content
                globalstr.global_constant = True
            else:
                # compare existing content
                existed = str(globalstr.initializer)
                if existed != str(content):
                    collision += 1
                    continue # loop until we resolve the name collision

            return CTemp(self, globalstr.bitcast(
                                           types.pointer(content.type.element)))


    def get_intrinsic(self, intrinsic_id, tys):
        '''get intrinsic function

        intrinsic_id : numerical ID of target intrinsic
        tys : type argument for the intrinsic
        '''
        lfunc = lc.Function.intrinsic(self.function.module, intrinsic_id, tys)
        return CFunc(self, lfunc)

    def get_function_named(self, name):
        '''get function by name
        '''
        m = self.function.module
        func = m.get_function_named(name)
        return CFunc(self, func)

    def is_terminated(self):
        '''is the current basic-block terminated?
        '''
        return _is_block_terminated(self.builder.basic_block)

    def atomic_cmpxchg(self, ptr, old, val, ordering, crossthread=True):
        '''atomic compare-exchange

        ptr : pointer to data
        old : old value to compare to
        val : new value
        ordering : memory ordering as a string
        crossthread : set to `False` for single-thread code

        Returns the old value on success.
        '''
        res = self.builder.atomic_cmpxchg(ptr.value, old.value, val.value,
                                          ordering, crossthread)
        return CTemp(self, res)

    def atomic_xchg(self, ptr, val, ordering, crossthread=True):
        '''atomic exchange

        ptr : pointer to data
        val : new value
        ordering : memory ordering as a string
        crossthread : set to `False` for single-thread code

        Returns the old value
        '''

        res = self.builder.atomic_xchg(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_add(self, ptr, val, ordering, crossthread=True):
        '''atomic add

        ptr : pointer to data
        val : new value
        ordering : memory ordering as a string
        crossthread : set to `False` for single-thread code

        Returns the computation result of the operation
        '''

        res = self.builder.atomic_add(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_sub(self, ptr, val, ordering, crossthread=True):
        '''atomic sub

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_sub(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_and(self, ptr, val, ordering, crossthread=True):
        '''atomic bitwise and

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_and(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_nand(self, ptr, val, ordering, crossthread=True):
        '''atomic bitwise nand

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_nand(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_or(self, ptr, val, ordering, crossthread=True):
        '''atomic bitwise or

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_or(ptr.value, val.value,
                                     ordering, crossthread)
        return CTemp(self, res)

    def atomic_xor(self, ptr, val, ordering, crossthread=True):
        '''atomic bitwise xor

        See `atomic_add` for parameters documentation
        '''

        res = self.builder.atomic_xor(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_max(self, ptr, val, ordering, crossthread=True):
        '''atomic signed maximum between value at `ptr` and `val`

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_max(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_min(self, ptr, val, ordering, crossthread=True):
        '''atomic signed minimum between value at `ptr` and `val`

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_min(ptr.value, val.value,
                                      ordering, crossthread)
        return CTemp(self, res)

    def atomic_umax(self, ptr, val, ordering, crossthread=True):
        '''atomic unsigned maximum between value at `ptr` and `val`

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_umax(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_umin(self, ptr, val, ordering, crossthread=True):
        '''atomic unsigned minimum between value at `ptr` and `val`

        See `atomic_add` for parameters documentation
        '''
        res = self.builder.atomic_umin(ptr.value, val.value,
                                       ordering, crossthread)
        return CTemp(self, res)

    def atomic_load(self, ptr, ordering, align=1, crossthread=True):
        '''atomic load

        ptr : pointer to the value to load
        align : memory alignment in bytes
        See `atomic_add` for other documentation of other parameters
        '''
        res = self.builder.atomic_load(ptr.value, ordering, align, crossthread)
        return CTemp(self, res)

    def atomic_store(self, val, ptr, ordering, align=1, crossthread=True):
        '''atomic store

        ptr : pointer to where to store
        val : value to store
        align : memory alignment in bytes
        See `atomic_add` for other documentation of other parameters
        '''

        res = self.builder.atomic_store(val.value, ptr.value, ordering,
                                        align, crossthread)
        return CTemp(self, res)

    def fence(self, ordering, crossthread=True):
        '''insert memory fence
        '''
        res = self.builder.fence(ordering, crossthread)
        return CTemp(self, res)

    def alignment(self, ty):
        '''get minimum alignment of `ty`
        '''
        return self.abi.abi_alignment(ty)

    @property
    def abi(self):
        return self.target_data

    def unreachable(self):
        '''insert instruction that causes segfault some platform (Intel),
        or no-op on others.

        It has no defined semantic.
        '''
        self.builder.unreachable()

    def add_auto_inline(self, callinst):
        self._auto_inline_list.append(callinst)


    def set_memop_non_temporal(self, ldst):
        const_one = self.constant(types.int, 1).value
        md = lc.MetaData.get(self.function.module, [const_one])
        ldst.set_metadata('nontemporal', md)

class _DeclareCDef(object):
    '''create a function a CDefinition to use with `CBuilder.depends`

    An instance of this class is created by the constructor of CDefinition.
    Do not use directly.
    '''
    def __init__(self, cdef):
        self.cdef = cdef

    def __str__(self):
        return self.cdef._name_

    def __call__(self, module):
        try:
            func = self.cdef.define(module)
        except FunctionAlreadyExists as e:
            (func,) = e
        return func

class CFuncRef(object):
    '''create a function reference to use with `CBuilder.depends`

    Either from name, type and pointer,
    Or from  a llvm.core.FunctionType instance
    '''
    def __init__(self, *args, **kwargs):
        def one_arg(fn):
            self._fn = fn
            self._name = fn.name

        def three_arg(name, ty, ptr):
            self._name = name
            self._type = ty
            self._ptr = ptr

        try:
            three_arg(*args, **kwargs)
            self._meth = self._from_pointer
        except TypeError:
            one_arg(*args, **kwargs)
            self._meth = self._from_func

    def __call__(self, module):
        return self._meth()

    def _from_func(self):
        return self._fn

    def _from_pointer(self):
        fnptr = types.pointer(self._type)
        ptr = lc.Constant.int(types.intp, self._ptr)
        ptr = ptr.inttoptr(fnptr)
        return ptr

    def __str__(self):
        return self._name

class CDefinition(CBuilder):
    '''represents function definition

    Inherit from this class to create a new function definition.

    Class Members
    -------------
    _name_ : name of the function
    _retty_ : return type
    _argtys_ : argument names and types as list of tuples;
               e.g. [ ( 'myarg', lc.Type.int() ), ... ]
    '''
    _name_ = ''             # name of the function; should overide in subclass
    _retty_  = types.void # return type; can overide in subclass
    _argtys_ = []       # a list of tuple(name, type, [attributes]); can overide in subclass

    def __new__(cls, *args, **kws):
        if cls.is_generic():
            # Call specialize if it is defined.
            cls = type('%s_Specialized' % cls.__name__, (cls,), {})
            cls.specialize(*args, **kws)

        obj = object.__new__(_DeclareCDef)
        obj.__init__(cls)
        return obj

    @classmethod
    def is_generic(cls):
        '''Is this a generic definition?
        '''
        return hasattr(cls, 'specialize')

    @classmethod
    def define(cls, module):
        '''define the function in the module.

        Raises NameError if a function of the same name has already been
        defined.
        '''
        functype = lc.Type.function(cls._retty_, [arg[1] for arg in cls._argtys_])
        name = cls._name_
        if not name:
            raise AttributeError("Function name cannot be empty.")

        func = module.get_or_insert_function(functype, name=name)

        if not func.is_declaration: # already defined?
            raise FunctionAlreadyExists(func)

        # Name all arguments
        for i, arginfo in enumerate(cls._argtys_):
            name = arginfo[0]
            func.args[i].name = name
            if len(arginfo) > 2:
                for attr in arginfo[2]:
                    func.args[i].add_attribute(attr)
        # Create builder and populate body
        cbuilder = object.__new__(cls)
        cbuilder.__init__(func)
        cbuilder.body(*cbuilder.args)
        cbuilder.close()

        # optimize
        fpm = lp.FunctionPassManager.new(module)
        pmb = lp.PassManagerBuilder.new()
        pmb.opt_level = 3
        pmb.vectorize = True
        pmb.populate(fpm)
        fpm.run(func)
        return func

    def body(self):
        '''overide this function to define the body.
        '''
        raise NotImplementedError

class CValue(object):
    def __init__(self, parent, handle):
        self.__parent = parent
        self.__handle = handle

    @property
    def handle(self):
        return self.__handle

    @property
    def parent(self):
        return self.__parent

    def _temp(self, val):
        return CTemp(self.parent, val)

def _get_operator_provider(ty):
    if _is_pointer(ty):
        return PointerValue
    elif _is_int(ty):
        return IntegerValue
    elif _is_real(ty):
        return RealValue
    elif _is_vector(ty):
        inner = _get_operator_provider(ty.element)
        return type(str(ty), (inner, VectorIndexing), {})
    elif _is_struct(ty):
        return StructValue
    else:
        assert False, (str(ty), type(ty))

class CTemp(CValue):
    def __new__(cls, parent, handle):
        meta = _get_operator_provider(handle.type)
        base = type(str('%s_%s' % (cls.__name__, handle.type)), (cls, meta), {})
        return object.__new__(base)

    def __init__(self, *args, **kws):
        super(CTemp, self).__init__(*args, **kws)
        self._init_mixin()

    @property
    def value(self):
        return self.handle

    @property
    def type(self):
        return self.value.type

class CVar(CValue):
    def __new__(cls, parent, ptr):
        meta = _get_operator_provider(ptr.type.pointee)
        base = type(str('%s_%s' % (cls.__name__, ptr.type.pointee)), (cls, meta), {})
        return object.__new__(base)

    def __init__(self, parent, ptr):
        super(CVar, self).__init__(parent, ptr)
        self._init_mixin()
        self.invariant = False

    def reference(self):
        return self._temp(self.handle)

    @property
    def ref(self):
        return self.reference()

    @property
    def value(self):
        return self.parent.builder.load(self.ref.value,
                                        invariant=self.invariant)

    @property
    def type(self):
        return self.ref.type.pointee

    def assign(self, val, **kws):
        if self.invariant:
            raise TypeError("Storing to invariant variable.")
        self.parent.builder.store(val.value, self.ref.value, **kws)
        return self

    def __iadd__(self, rhs):
        return self.assign(self.__add__(rhs))

    def __isub__(self, rhs):
        return self.assign(self.__sub__(rhs))

    def __imul__(self, rhs):
        return self.assign(self.__mul__(rhs))

    def __idiv__(self, rhs):
        return self.assign(self.__div__(rhs))

    def __imod__(self, rhs):
        return self.assign(self.__mod__(rhs))

    def __ilshift__(self, rhs):
        return self.assign(self.__lshift__(rhs))

    def __irshift__(self, rhs):
        return self.assign(self.__rshift__(rhs))

    def __iand__(self, rhs):
        return self.assign(self.__and__(rhs))

    def __ior__(self, rhs):
        return self.assign(self.__or__(rhs))

    def __ixor__(self, rhs):
        return self.assign(self.__xor__(rhs))


class OperatorMixin(object):
    def _init_mixin(self):
        pass

class IntegerValue(OperatorMixin):

    def _init_mixin(self):
        self._unsigned = False

    def _get_unsigned(self):
        return self._unsigned

    def _set_unsigned(self, unsigned):
        self._unsigned = bool(unsigned)

    unsigned = property(_get_unsigned, _set_unsigned)

    def __add__(self, rhs):
        return self._temp(self.parent.builder.add(self.value, rhs.value))

    def __sub__(self, rhs):
        return self._temp(self.parent.builder.sub(self.value, rhs.value))

    def __mul__(self, rhs):
        return self._temp(self.parent.builder.mul(self.value, rhs.value))

    def __div__(self, rhs):
        if self.unsigned:
            return self._temp(self.parent.builder.udiv(self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.sdiv(self.value, rhs.value))

    def __mod__(self, rhs):
        if self.unsigned:
            return self._temp(self.parent.builder.urem(self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.srem(self.value, rhs.value))

    def __ilshift__(self, rhs):
        return self._temp(self.parent.builder.shl(self.value, rhs.value))

    def __irshift__(self, rhs):
        if self.unsigned:
            return self._temp(self.self.parent.builder.lshr(self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.ashr(self.value, rhs.value))

    def __iand__(self, rhs):
        return self._temp(self.parent.builder.and_(self.value, rhs.value))

    def __ior__(self, rhs):
        return self._temp(self.parent.builder.or_(self.value, rhs.value))

    def __ixor__(self, rhs):
        return self._temp(self.parent.builder.xor(self.value, rhs.value))

    def __lt__(self, rhs):
        if self.unsigned:
            return self._temp(self.parent.builder.icmp(lc.ICMP_ULT, self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.icmp(lc.ICMP_SLT, self.value, rhs.value))

    def __le__(self, rhs):
        if self.unsigned:
            return self._temp(self.parent.builder.icmp(lc.ICMP_ULE, self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.icmp(lc.ICMP_SLE, self.value, rhs.value))

    def __eq__(self, rhs):
        return self._temp(self.parent.builder.icmp(lc.ICMP_EQ, self.value, rhs.value))

    def __ne__(self, rhs):
        return self._temp(self.parent.builder.icmp(lc.ICMP_NE, self.value, rhs.value))

    def __gt__(self, rhs):
        if self.unsigned:
            return self._temp(self.parent.builder.icmp(lc.ICMP_UGT, self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.icmp(lc.ICMP_SGT, self.value, rhs.value))

    def __ge__(self, rhs):
        if self.unsigned:
            return self._temp(self.parent.builder.icmp(lc.ICMP_UGE, self.value, rhs.value))
        else:
            return self._temp(self.parent.builder.icmp(lc.ICMP_SGE, self.value, rhs.value))

    def cast(self, ty, unsigned=False):
        if ty == self.type:
            return self._temp(self.value)

        if _is_real(ty):
            if self.unsigned or unsigned:
                return self._temp(self.parent.builder.uitofp(self.value, ty))
            else:
                return self._temp(self.parent.builder.sitofp(self.value, ty))
        elif _is_int(ty):
            if self.parent.abi.size(self.type) < self.parent.abi.size(ty):
                if self.unsigned or unsigned:
                    return self._temp(self.parent.builder.zext(self.value, ty))
                else:
                    return self._temp(self.parent.builder.sext(self.value, ty))
            else:
                return self._temp(self.parent.builder.trunc(self.value, ty))
        raise CastError(self.type, ty)

class RealValue(OperatorMixin):
    def __add__(self, rhs):
        return self._temp(self.parent.builder.fadd(self.value, rhs.value))

    def __sub__(self, rhs):
        return self._temp(self.parent.builder.fsub(self.value, rhs.value))

    def __mul__(self, rhs):
        return self._temp(self.parent.builder.fmul(self.value, rhs.value))

    def __div__(self, rhs):
        return self._temp(self.parent.builder.fdiv(self.value, rhs.value))

    def __mod__(self, rhs):
        return self._temp(self.parent.builder.frem(self.value, rhs.value))

    def __lt__(self, rhs):
        return self._temp(self.parent.builder.fcmp(lc.FCMP_OLT, self.value, rhs.value))

    def __le__(self, rhs):
        return self._temp(self.parent.builder.fcmp(lc.FCMP_OLE, self.value, rhs.value))

    def __eq__(self, rhs):
        return self._temp(self.parent.builder.fcmp(lc.FCMP_OEQ, self.value, rhs.value))

    def __ne__(self, rhs):
        return self._temp(self.parent.builder.fcmp(lc.FCMP_ONE, self.value, rhs.value))

    def __gt__(self, rhs):
        return self._temp(self.parent.builder.fcmp(lc.FCMP_OGT, self.value, rhs.value))

    def __ge__(self, rhs):
        return self._temp(self.parent.builder.fcmp(lc.FCMP_OGE, self.value, rhs.value))

    def cast(self, ty, unsigned=False):
        if ty == self.type:
            return self._temp(self.value)

        if _is_int(ty):
            if unsigned:
                return self._temp(self.parent.builder.fptoui(self.value, ty))
            else:
                return self._temp(self.parent.builder.fptosi(self.value, ty))

        if _is_real(ty):
            if self.parent.abi.size(self.type) > self.parent.abi.size(ty):
                return self._temp(self.parent.builder.fptrunc(self.value, ty))
            else:
                return self._temp(self.parent.builder.fpext(self.value, ty))

        raise CastError(self.type, ty)


class PointerIndexing(OperatorMixin):
    def __getitem__(self, idx):
        '''implement access indexing

        Uses GEP.
        '''
        bldr = self.parent.builder
        if type(idx) is slice:
            # just handle case by case
            # Case #1: A[idx:] get pointer offset by idx
            if not idx.step and not idx.stop:
                idx = _auto_coerce_index(self.parent, idx.start)
                ptr = bldr.gep(self.value, [idx.value])
                return CArray(self.parent, ptr)
        else: # return an variable at idx
            idx = _auto_coerce_index(self.parent, idx)
            ptr = bldr.gep(self.value, [idx.value])
            return CVar(self.parent, ptr)

    def __setitem__(self, idx, val):
        idx = _auto_coerce_index(self.parent, idx)
        bldr = self.parent.builder
        self[idx].assign(val)

class PointerCasting(OperatorMixin):
    def cast(self, ty):
        if ty == self.type:
            return self._temp(self.value)

        if _is_pointer(ty):
            return self._temp(self.parent.builder.bitcast(self.value, ty))
        raise CastError(self.type, ty)



class VectorIndexing(OperatorMixin):
    def __getitem__(self, idx):
        '''implement access indexing

        Uses GEP.
        '''
        bldr = self.parent.builder
        idx = _auto_coerce_index(self.parent, idx)
        val = bldr.extract_element(self.value, idx.value)
        return CTemp(self.parent, val)

    def __setitem__(self, idx, val):
        idx = _auto_coerce_index(self.parent, idx)
        bldr = self.parent.builder
        vec = bldr.insert_element(self.value, val.value, idx.value)
        bldr.store(vec, self.ref.value)

class PointerValue(PointerIndexing, PointerCasting):

    def load(self, **kws):
        return self._temp(self.parent.builder.load(self.value, **kws))

    def store(self, val, nontemporal=False, **kws):
        inst = self.parent.builder.store(val.value, self.value, **kws)
        if nontemporal:
            self.parent.set_memop_non_temporal(inst)


    def atomic_load(self, ordering, align=None, crossthread=True):
        '''atomic load memory for pointer types

        align : overide to control memory alignment; otherwise the default
                alignment of the type is used.

        Other parameters are the same as `CBuilder.atomic_load`
        '''
        if align is None:
            align = self.parent.alignment(self.type.pointee)
        inst = self.parent.builder.atomic_load(self.value, ordering, align,
                                               crossthread=crossthread)
        return self._temp(inst)

    def atomic_store(self, value, ordering, align=None,  crossthread=True):
        '''atomic memory store for pointer types

        align : overide to control memory alignment; otherwise the default
                alignment of the type is used.

        Other parameters are the same as `CBuilder.atomic_store`
        '''
        if align is None:
            align = self.parent.alignment(self.type.pointee)
        self.parent.builder.atomic_store(value.ptr, self.value, ordering,
                                         align=align, crossthread=crossthread)

    def atomic_cmpxchg(self, old, new, ordering, crossthread=True):
        '''atomic compare-exchange for pointer types

        Other parameters are the same as `CBuilder.atomic_cmpxchg`
        '''
        inst = self.parent.builder.atomic_cmpxchg(self.value, old.value,
                                                  new.value, ordering,
                                                  crossthread=crossthread)
        return self._temp(inst)

    def as_struct(self, cstruct_class, volatile=False):
        '''load a pointer to a structure and assume a structure interface
        '''
        ptr = self.parent.builder.load(self.value, volatile=volatile)
        return cstruct_class(self.parent, self.value)

class StructValue(OperatorMixin):

    def as_struct(self, cstruct_class):
        '''assume a structure interface
        '''
        return cstruct_class(self.parent, self.ref.value)


class CFunc(CValue, PointerCasting):
    '''Wraps function pointer
    '''
    def __init__(self, parent, func):
        super(CFunc, self).__init__(parent, func)

    @property
    def function(self):
        return self.handle

    def __call__(self, *args, **opts):
        '''Call the function with the given arguments

        *args : variable arguments of CValue instances
        '''
        arg_values = _list_values(args)
        ftype = self.function.type.pointee
        for i, (exp, got) in enumerate(zip(ftype.args, arg_values)):
            if exp != got.type:
                raise TypeError("At call to %s, "
                                "argument %d mismatch: %s != %s"
                                % (self.function.name, i, exp, got.type))
        res = self.parent.builder.call(self.function, arg_values)

        if hasattr(self.function, 'calling_convention'):
            res.calling_convention = self.function.calling_convention

        if opts.get('inline'):
            self.parent.add_auto_inline(res)

        if ftype.return_type != lc.Type.void():
            return CTemp(self.parent, res)

    @property
    def value(self):
        return self.function

    @property
    def type(self):
        return self.function.type


class CArray(CValue, PointerIndexing, PointerCasting):
    '''wraps a array

    Similar to C arrays
    '''
    def __init__(self, parent, base):
        super(CArray, self).__init__(parent, base)

    @property
    def value(self):
        return self.handle

    def reference(self):
        return self._temp(self.value)

    @property
    def type(self):
        return self.value.type

    def vector_load(self, count, align=0):
        parent = self.parent
        builder = parent.builder
        values = [self[i] for i in range(count)]

        vecty = types.vector(self.type.pointee, count)
        vec = builder.load(builder.bitcast(self.value, types.pointer(vecty)),
                           align=align)
        return self._temp(vec)

    def vector_store(self, vec, align=0):
        if vec.type.element != self.type.pointee:
            raise TypeError("Type mismatch; expect %s but got %s" % \
                            (vec.type.element, self.type.pointee))
        parent = self.parent
        builder = parent.builder
        vecptr = builder.bitcast(self.value, types.pointer(vec.type))
        builder.store(vec.value, vecptr, align=align)
        return self


class CStruct(CValue):
    '''Wraps a structure

    Structure in LLVM can be identified by name of layout.

    Subclass to define a new structure. All fields are defined in the
    `_fields_` class attribute as a list of tuple (name, type).

    Can define new methods which gets inlined to the parent CBuilder.
    '''

    @classmethod
    def llvm_type(cls):
        return lc.Type.struct([v for k, v in cls._fields_])

    def __init__(self, parent, ptr):
        super(CStruct, self).__init__(parent, ptr)
        makeind = lambda x: self.parent.constant(types.int, x).value

        for i, (fd, _) in enumerate(self._fields_):
            gep = self.parent.builder.gep(ptr, [makeind(0), makeind(i)])
            gep.name = "%s.%s" % (type(self).__name__, fd)
            if hasattr(self, fd):
                raise AttributeError("Field name shadows another attribute")
            setattr(self, fd, CVar(self.parent, gep))

    def reference(self):
        return self._temp(self.handle)


class CExternal(object):
    '''subclass to define external interface

    All class attributes that are `llvm.core.FunctionType` are converted
    to `CFunc` instance during instantiation.
    '''

    _calling_convention_ = None # default

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
            if self._calling_convention_:
                func.calling_convention = self._calling_convention_

            if func.type.pointee != ftype:
                raise NameError("Function has already been declared "
                                "with a different type: %s != %s"
                                % (func.type, ftype) )
            setattr(self, fname, CFunc(cbuilder, func))


