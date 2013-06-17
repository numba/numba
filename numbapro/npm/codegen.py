from collections import namedtuple

import llvm.core as lc
from llvm.core import Type, Constant

from . import typing, symbolic
from .errors import CompileError

GlobalVar = namedtuple('GlobalVar', ['type', 'gvar'])

class CodeGenError(CompileError):
    def __init__(self, value, msg):
        super(CodeGenError, self).__init__(value.lineno, msg)

class CodeGen(object):
    def __init__(self, name, blocks, typemap, consts, args, return_type, intp,
                 extended_globals={}, extended_calls={}):
        self.blocks = blocks
        self.typemap = typemap
        self.args = args
        self.return_type = return_type
        self.typesetter = TypeSetter(intp)
        self.consts = consts                     # global values at compile time
        self.extern_globals = {}                 # {name: GlobalVar}
        self.extended_globals = extended_globals # codegen for special globals
        self.extended_calls = extended_calls     # codegen for special functions

        self.bbmap = {}
        self.valmap = {}
        self.pending_phis = {}

        self.lmod = lc.Module.new(repr(self))

        argtys = []
        for ty in args:
            lty = self.to_llvm(ty)
            if ty.is_scalar and ty.is_complex:
                lty = Type.pointer(lty)
            argtys.append(lty)

        if return_type:
            retty = Type.pointer(self.to_llvm(return_type))
            fnty = Type.function(Type.void(), argtys + [retty])
        else:
            fnty = Type.function(Type.void(), argtys)
        self.lfunc = self.lmod.add_function(fnty, name=name)

    def to_llvm(self, ty):
        return self.typesetter.to_llvm(ty)

    def sizeof(self, ty):
        return sizeof(self.builder, self.to_llvm(ty), self.typesetter.llvm_intp)

    def array_getpointer(self, ary, idx):
        aryty = self.typemap[ary]
        ary = self.valmap[ary]
        data = self.builder.load(gep(self.builder, ary, 0, 0))
        shapeptr = gep(self.builder, ary, 0, 1)
        shape = [self.builder.load(gep(self.builder, shapeptr, 0, ax))
                    for ax in range(aryty.ndim)]

        strideptr = gep(self.builder, ary, 0, 2)
        strides = [self.builder.load(gep(self.builder, strideptr, 0, ax))
                    for ax in range(aryty.ndim)]

        order = aryty.order
        indices = [self.do_cast(self.valmap[i.value],
                                self.typemap[i.value],
                                self.typesetter.intp)
                   for i in idx]

        return array_pointer(self.builder, data, shape, strides, order, indices)

    def cast(self, value, destty):
        ty = self.typemap[value]
        lval = self.valmap[value]
        return self.do_cast(lval, ty, destty)

    def do_cast(self, lval, ty, destty):
        if ty == destty:  # same type
            return lval

        if ty.is_int and destty.is_int:
            # int -> int
            if destty.bitwidth > ty.bitwidth:
                op = (self.builder.sext
                        if destty.is_signed
                        else self.builder.zext)
            else:
                op = self.builder.trunc

        elif ty.is_int and destty.is_float:
            # int -> float
            op = (self.builder.sitofp
                    if ty.is_signed
                    else self.builder.uitofp)

        elif ty.is_float and destty.is_float:
            # float -> float
            op = (self.builder.fptrunc
                    if destty.bitwidth < ty.bitwidth
                    else self.builder.fpext)

        elif ty.is_float and destty.is_int:
            # float -> int
            op = (self.builder.fptosi
                    if destty.is_signed
                    else self.builder.fptoui)


        try:
            return op(lval, self.to_llvm(destty))
        except NameError:
            raise NotImplementedError('casting %s -> %s' % (ty, destty))

    def define_const(self, ty, val):
        if ty.is_scalar:
            if ty.kind == 'i':
                lty = self.to_llvm(ty)
                return Constant.int_signextend(lty, val)
            elif ty.kind == 'u':
                lty = self.to_llvm(ty)
                return Constant.int(lty, val)
            elif ty.kind == 'f':
                lty = self.to_llvm(ty)
                return Constant.real(lty, val)
            elif ty.kind == 'c':
                lty = self.to_llvm(ty.complex_element)
                real = self.define_const(lty, val.real)
                imag = self.define_const(lty, val.imag)
                return Constant.struct([real, imag])
        elif ty.is_tuple:
            ev = [self.define_const(ty.element, v) for v in val]
            return Constant.struct(ev)
        else:
            raise NotImplementedError

    def generate(self):
        # generate in the order of basic block offset
        genorder = list(sorted(self.blocks.iteritems()))
        # init basicblock map
        for off, blk in genorder:
            self.bbmap[off] = self.lfunc.append_basic_block('bb%d' % off)

        # add phi first
        for _, blk in genorder:
            bb = self.bbmap[blk.offset]
            self.builder = lc.Builder.new(bb)
            for expr in blk.body:
                if expr.kind == 'Phi':
                    self.generate_expression(expr)

        # generate other expressions and terminator
        for _, blk in genorder:
            bb = self.bbmap[blk.offset]
            self.builder = lc.Builder.new(bb)
            for expr in blk.body:
                if expr.kind != 'Phi':
                    self.generate_expression(expr)
            else:
                self.generate_terminator(blk.terminator)

        # fix PHIs
        for phi, incomings in self.pending_phis.iteritems():
            for blkoff, incval in incomings:
                val = self.valmap[incval.value]
                blk = self.bbmap[blkoff]
                if val is None:
                    val = Constant.undef(phi.type)
                phi.add_incoming(val, blk)

        # verify
        return self.lfunc
        self.lmod.verify()

    def generate_expression(self, expr):
        kind = symbolic.OP_MAP.get(expr.kind, expr.kind)
        fn = getattr(self, 'expr_' + kind, self.generic_expr)
        fn(expr)

    def generic_expr(self, expr):
        raise CodeGenError(expr, "%s not implemented" % expr)

    def generate_terminator(self, term):
        kind = term.kind
        fn = getattr(self, 'term_' + kind, self.generic_term)
        fn(term)

    def generic_term(self, term):
        raise CodeGenError(term, "not implemented")

    # -------- expr ---------

    def expr_Arg(self, expr):
        num = expr.args.num
        argty = self.args[num]
        if argty.is_scalar and argty.is_complex:
            val = self.builder.load(self.lfunc.args[num])
        else:
            val = self.lfunc.args[num]
        self.valmap[expr] = val

    def expr_Undef(self, expr):
        self.valmap[expr] = None

    def expr_Global(self, expr):
        name = expr.args.name
        value = expr.args.value
        if value is not None:
            self.valmap[expr] = self.extended_globals[value](self, expr)
        else:
            # handle regular global value
            if name not in self.extern_globals:
                # first time reference to a global
                ty = self.typemap[expr]
                lty = self.to_llvm(ty)
                gvar = self.lmod.add_global_variable(lty,
                                             name='__npm_global_%s' % name)

                gvar.initializer = Constant.undef(gvar.type.pointee)
                self.extern_globals[name] = GlobalVar(type=ty, gvar=gvar)

            self.valmap[expr] = self.builder.load(
                                                self.extern_globals[name].gvar)


    def expr_Const(self, expr):
        ty = self.typemap[expr]
        self.valmap[expr] = self.define_const(ty, expr.args.value)

    def expr_Phi(self, expr):
        ty = self.typemap[expr]
        phi = self.builder.phi(self.to_llvm(ty))

        for blkoff, valref in expr.args.incomings:
            val = valref.value
            vty = self.typemap[val]
            assert vty == ty, (vty, ty, expr)
        self.pending_phis[phi] = expr.args.incomings
        self.valmap[expr] = phi

    def expr_BinOp(self, expr):
        finalty = self.typemap[expr]
        lhs, rhs = expr.args.lhs.value, expr.args.rhs.value
        lty, rty = map(self.typemap.get, [lhs, rhs])
        if expr.kind == '/':
            opty = finalty
        else:
            opty = lty.coerce(rty)
        lhs, rhs = map(lambda x: self.cast(x, opty), [lhs, rhs])
        if opty.is_int:
            offset = 'iu'.index(opty.kind)
            res = INT_OPMAP[expr.kind][offset](self.builder, lhs, rhs)
        elif opty.is_float:
            res = FLOAT_OPMAP[expr.kind](self.builder, lhs, rhs)
        elif opty.is_complex:
            res = COMPLEX_OPMAP[expr.kind](self.builder, lhs, rhs)
        self.valmap[expr] = self.do_cast(res, opty, finalty)

    def expr_BoolOp(self, expr):
        assert self.typemap[expr] == typing.boolean
        lhs, rhs = expr.args.lhs.value, expr.args.rhs.value
        lty, rty = map(self.typemap.get, [lhs, rhs])
        opty = lty.coerce(rty)
        lhs, rhs = map(lambda x: self.cast(x, opty), [lhs, rhs])
        self.valmap[expr] = self._do_compare(expr.kind, opty, lhs, rhs)

    def expr_UnaryOp(self, expr):
        finalty = self.typemap[expr]
        operand = expr.args.value.value
        ity = self.typemap[operand]
        res = INT_UNARY_OPMAP[expr.kind](self.builder, self.valmap[operand])
        self.valmap[expr] = res

    def expr_For(self, expr):
        ty = self.typemap[expr]
        lty = self.to_llvm(ty)
        index = expr.args.index.value
        stop = expr.args.stop.value
        step = expr.args.step.value

        index_ty = self.typemap[index]
        stop_ty = self.typemap[stop]
        step_ty = self.typemap[step]

        intp = self.typesetter.intp
        index = self.cast(index, intp)

        stop = self.cast(stop, intp)
        step = self.cast(step, intp)

        assert step_ty.is_signed

        positive = self.builder.icmp(lc.ICMP_SGE,
                                     step, self.define_const(intp, 0))
        ok_pos = self.builder.icmp(lc.ICMP_SLT, index, stop)
        ok_neg = self.builder.icmp(lc.ICMP_SGT, index, stop)

        ok = self.builder.select(positive, ok_pos, ok_neg)

        # fixes end of loop iterator value
        index_prev = self.builder.sub(index, step)
        self.valmap[expr.args.index.value] = self.builder.select(ok, index, index_prev)

        self.valmap[expr] = ok

    def expr_GetItem(self, expr):
        elemty = self.typemap[expr.args.obj.value].element
        outty = self.typemap[expr]
        ptr = self.array_getpointer(expr.args.obj.value, expr.args.idx)
        val = self.builder.load(ptr)
        self.valmap[expr] = self.do_cast(val, elemty, outty)

    def expr_SetItem(self, expr):
        elemty = self.typemap[expr.args.obj.value].element
        value = self.cast(expr.args.value.value, elemty)
        ptr = self.array_getpointer(expr.args.obj.value, expr.args.idx)
        self.builder.store(value, ptr)

    def expr_ArrayAttr(self, expr):
        attr = expr.args.attr
        ary = self.valmap[expr.args.obj.value]
        aryty = self.typemap[expr.args.obj.value]
        if attr == 'shape':
            idx = self.valmap[expr.args.idx.value]
            shapeptr = gep(self.builder, ary, 0, 1, idx)
            res = self.builder.load(shapeptr)
        elif attr == 'strides':
            idx = self.valmap[expr.args.idx.value]
            strideptr = gep(self.builder, ary, 0, 2, idx)
            res = self.builder.load(strideptr)
        elif attr == 'ndim':
            res = Constant.int(self.typesetter.llvm_intp, aryty.ndim)
        elif attr == 'size':
            shapeptr = gep(self.builder, ary, 0, 1)
            shape = [self.builder.load(gep(self.builder, shapeptr, 0, ax))
                        for ax in range(aryty.ndim)]
            res = reduce(self.builder.mul, shape)
            
        self.valmap[expr] = self.do_cast(res, self.typesetter.intp,
                                         self.typemap[expr])

    def expr_ComplexGetAttr(self, expr):
        attr = expr.args.attr
        if attr == 'real':
            offset = 0
        elif attr == 'imag':
            offset = 1
        obj = expr.args.obj.value
        cmplx = self.valmap[obj]
        cty = self.typemap[obj]
        res = self.builder.extract_value(cmplx, offset)
        self.valmap[expr] = self.do_cast(res, cty.complex_element,
                                         self.typemap[expr])


    def expr_Call(self, expr):
        func = expr.args.func
        
        if func is complex:
            self.call_complex(expr)
        elif func is int:
            self.call_int(expr)
        elif func is float:
            self.call_int(expr)
        elif func is min:
            self.call_min(expr)
        elif func is max:
            self.call_max(expr)
        elif func is abs:
            self.call_abs(expr)
        elif func in self.extended_calls:
            self.extended_calls[func](self, expr)
        else:
            raise CodeGenError(expr, "function %s not implemented" % func)

    def expr_Unpack(self, expr):
        tuplevalue = expr.args.obj.value
        tuty = self.typemap[tuplevalue]
        package = self.valmap[tuplevalue]
        offset = expr.args.offset
        ty = self.typemap[expr]
        self.valmap[expr] = self.do_cast(package[offset], tuty.element, ty)

    # -------- call ---------
    def call_complex(self, expr):
        cmplxty = self.typemap[expr]
        res = Constant.undef(self.to_llvm(cmplxty))
        args = expr.args.args
        nargs = len(args)
        if nargs == 1:
            (real,) = args
            realval = self.cast(real.value, cmplxty.complex_element)
            imagval = Constant.null(realval.type)
        else:
            assert nargs == 2
            real, imag = args
            realval = self.cast(real.value, cmplxty.complex_element)
            imagval = self.cast(imag.value, cmplxty.complex_element)

        res = self.builder.insert_value(res, realval, 0)
        res = self.builder.insert_value(res, imagval, 1)
        self.valmap[expr] = res

    def call_int(self, expr):
        outty = self.typemap[expr]
        assert len(expr.args.args) == 1
        self.valmap[expr] = self.cast(expr.args.args[0].value, outty)

    def call_float(self, expr):
        outty = self.typemap[expr]
        assert len(expr.args.args) == 1
        self.valmap[expr] = self.cast(expr.args.args[0].value, outty)

    def call_min(self, expr):
        args = [a.value for a in expr.args.args]
        opty = typing.coerce(*map(self.typemap.get, args))
        castargs = [self.cast(a, opty) for a in args]
        do_min = lambda a, b: self._do_min(opty, a, b)
        res = reduce(do_min, castargs)
        self.valmap[expr] = self.do_cast(res, opty, self.typemap[expr])
    
    def call_max(self, expr):
        args = [a.value for a in expr.args.args]
        opty = typing.coerce(*map(self.typemap.get, args))
        castargs = [self.cast(a, opty) for a in args]
        do_max = lambda a, b: self._do_max(opty, a, b)
        res = reduce(do_max, castargs)
        self.valmap[expr] = self.do_cast(res, opty, self.typemap[expr])

    def call_abs(self, expr):
        (arg,) = expr.args.args
        number = arg.value
        opty = self.typemap[number]
        llopty = self.to_llvm(opty)
        zero = Constant.null(llopty)
        llnum = self.valmap[number]
        if opty.is_int:
            assert opty.is_signed
            negone = Constant.int_signextend(llopty, -1)
            mul = self.builder.mul
        else:
            assert opty.is_float
            negone = Constant.real(llopty, -1.)
            mul = self.builder.fmul
        isneg = self._do_compare('<', opty, llnum, zero)
        res = self.builder.select(isneg, mul(llnum, negone), llnum)
        self.valmap[expr] = self.do_cast(res, opty, self.typemap[expr])

    # -------- _do_* ---------

    def _do_compare(self, cmp, opty, lhs, rhs):
        if opty.is_int:
            offset = 'iu'.index(opty.kind)
            flag = INT_BOOL_OP_MAP[cmp][offset]
            res = self.builder.icmp(flag, lhs, rhs)
        elif opty.is_float:
            flag = FLOAT_BOOL_OP_MAP[cmp]
            res = self.builder.fcmp(flag, lhs, rhs)
        return res

    def _do_min(self, opty, a, b):
        pred = self._do_compare('<=', opty, a, b)
        return self.builder.select(pred, a, b)

    def _do_max(self, opty, a, b):
        pred = self._do_compare('>=', opty, a, b)
        return self.builder.select(pred, a, b)

    # -------- term ---------

    def term_Jump(self, term):
        self.builder.branch(self.bbmap[term.args.target])

    def term_Branch(self, term):
        opty = self.typemap[term.args.cmp.value]
        pred = self._do_compare('!=', opty, self.valmap[term.args.cmp.value],
                                Constant.null(self.to_llvm(opty)))
        self.builder.cbranch(pred, self.bbmap[term.args.true],
                             self.bbmap[term.args.false])

    def term_Ret(self, expr):
        ty = self.return_type
        val = self.cast(expr.args.value.value, ty)
        self.builder.store(val, self.lfunc.args[-1])
        self.builder.ret_void()

    def term_RetVoid(self, expr):
        self.builder.ret_void()


class TypeSetter(object):
    def __init__(self, intp):
        self.intp = typing.ScalarType('i%d' % intp)
        self.llvm_intp = self.to_llvm(self.intp)

    def to_llvm(self, ty):
        if isinstance(ty, typing.ScalarType):
            return self.convert_scalar(ty)
        elif isinstance(ty, typing.ArrayType):
            return self.convert_array(ty)
        else:
            raise TypeError('unknown type: %s' % ty)

    def convert_scalar(self, ty):
        if ty.kind in 'iu':
            return Type.int(ty.bitwidth)
        elif ty.kind == 'f':
            if ty.bitwidth == 32:
                return Type.float()
            elif ty.bitwidth == 64:
                return Type.double()
        elif ty.kind == 'c':
            fty = self.convert_scalar(ty.complex_element)
            return Type.struct([fty, fty])

        raise TypeError('unknown scalar: %s' % ty)

    def convert_array(self, ty):
        elemty = self.convert_scalar(ty.element)
        data = Type.pointer(elemty)
        shape = Type.array(self.llvm_intp, ty.ndim)
        strides = Type.array(self.llvm_intp, ty.ndim)
        struct = Type.struct([data, shape, strides])
        return Type.pointer(struct)

def complex_extract(builder, cval):
    return (builder.extract_value(cval, 0),
            builder.extract_value(cval, 1))

def complex_make(builder, complexty, real, imag):
    res = Constant.undef(complexty)
    res = builder.insert_value(res, real, 0)
    res = builder.insert_value(res, imag, 1)
    return res

def complex_add(builder, lhs, rhs):
    lreal, limag = complex_extract(builder, lhs)
    rreal, rimag = complex_extract(builder, rhs)

    real = builder.fadd(lreal, rreal)
    imag = builder.fadd(limag, rimag)

    return complex_make(builder, lhs.type, real, imag)

def complex_sub(builder, lhs, rhs):
    lreal, limag = complex_extract(builder, lhs)
    rreal, rimag = complex_extract(builder, rhs)

    real = builder.fsub(lreal, rreal)
    imag = builder.fsub(limag, rimag)

    return complex_make(builder, lhs.type, real, imag)

def integer_invert(builder, val):
    return builder.xor(val, Constant.int_signextend(val.type, -1))

INT_OPMAP  = {
     '+': (lc.Builder.add, lc.Builder.add),
     '-': (lc.Builder.sub, lc.Builder.sub),
     '*': (lc.Builder.mul, lc.Builder.mul),
     '/': (lc.Builder.sdiv, lc.Builder.udiv),
    '//': (lc.Builder.sdiv, lc.Builder.udiv),
     '%': (lc.Builder.srem, lc.Builder.urem),
     '&': (lc.Builder.and_, lc.Builder.and_),
     '|': (lc.Builder.or_, lc.Builder.or_),
     '^': (lc.Builder.xor, lc.Builder.xor),
}

INT_UNARY_OPMAP = {
    '~': integer_invert,
}

FLOAT_OPMAP = {
     '+': lc.Builder.fadd,
     '-': lc.Builder.fsub,
     '*': lc.Builder.fmul,
     '/': lc.Builder.fdiv,
    '//': lc.Builder.fdiv,
     '%': lc.Builder.frem,
}

COMPLEX_OPMAP = {
    '+': complex_add,
    '-': complex_sub,
}

INT_BOOL_OP_MAP = {
    '>': (lc.ICMP_SGT, lc.ICMP_UGT),
    '>=': (lc.ICMP_SGE, lc.ICMP_UGE),
    '<': (lc.ICMP_SLT, lc.ICMP_ULT),
    '<=': (lc.ICMP_SLE, lc.ICMP_ULE),
    '==': (lc.ICMP_EQ, lc.ICMP_EQ),
    '!=': (lc.ICMP_NE, lc.ICMP_NE),
}

FLOAT_BOOL_OP_MAP = {
    '>':  lc.FCMP_OGT,
    '>=':  lc.FCMP_OGE,
    '<':  lc.FCMP_OLT,
    '<=':  lc.FCMP_OLE,
    '==':  lc.FCMP_OEQ,
    '!=':  lc.FCMP_ONE,
}


def array_pointer(builder, data, shape, strides, order, indices):
    assert order in 'CFA'
    intp = shape[0].type
    if order in 'CF':
        # optimize for C and F contiguous
        steps = []
        if order == 'C':
            for i in range(len(shape)):
                last = Constant.int(intp, 1)
                for j in shape[i + 1:]:
                    last = builder.mul(last, j)
                steps.append(last)
        elif order =='F':
            for i in range(len(shape)):
                last = Constant.int(intp, 1)
                for j in shape[:i]:
                    last = builder.mul(last, j)
                steps.append(last)
        else:
            assert False
        loc = Constant.null(intp)
        for i, s in zip(indices, steps):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        ptr = builder.gep(data, [loc])
    else:
        # any order
        loc = Constant.null(intp)
        for i, s in zip(indices, strides):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        base = builder.ptrtoint(data, intp)
        target = builder.add(base, loc)
        ptr = builder.inttoptr(target, data.type)
    return ptr

def array_setitem(builder, data, shape, strides, order, indices, value):
    ptr = array_pointer(builder, data, shape, strides, order, indices)
    builder.store(value, ptr)

def array_getitem(builder, data, shape, strides, order, indices):
    ptr = array_pointer(builder, data, shape, strides, order, indices)
    val = builder.load(ptr)
    return val

def auto_int(x):
    if isinstance(x, int):
        return Constant.int(Type.int(), x)
    else:
        return x

def gep(builder, ptr, *idx):
    return builder.gep(ptr, [auto_int(x) for x in idx])

def sizeof(builder, ty, intp):
    ptr = Type.pointer(ty)
    null = Constant.null(ptr)
    offset = builder.gep(null, [Constant.int(Type.int(), 1)])
    return builder.ptrtoint(offset, intp)
