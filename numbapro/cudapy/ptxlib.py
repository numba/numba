from __future__ import print_function
import __builtin__
import llvm.core as lc
from numbapro.npm import types, cgutils, aryutils, arylib, intrinsics
from . import ptx, nvvmutils


def SRegImplFactory(func):
    class SRegImpl(object):
        function = func, (), types.uint32

        def generic_implement(self, context, args, argtys, retty):
            builder = context.builder
            sreg = ptx.SREG_MAPPING[self.function[0]]
            lretty = self.function[-1].llvm_as_value()
            func = cgutils.get_function(builder, sreg, lretty, ())
            return builder.call(func, ())
    return SRegImpl


TidX = SRegImplFactory(ptx._ptx_sreg_tidx)
TidY = SRegImplFactory(ptx._ptx_sreg_tidy)
TidZ = SRegImplFactory(ptx._ptx_sreg_tidz)

NTidX = SRegImplFactory(ptx._ptx_sreg_ntidx)
NTidY = SRegImplFactory(ptx._ptx_sreg_ntidy)
NTidZ = SRegImplFactory(ptx._ptx_sreg_ntidz)

CTAidX = SRegImplFactory(ptx._ptx_sreg_ctaidx)
CTAidY = SRegImplFactory(ptx._ptx_sreg_ctaidy)

NCTAidX = SRegImplFactory(ptx._ptx_sreg_nctaidx)
NCTAidY = SRegImplFactory(ptx._ptx_sreg_nctaidy)


#-------------------------------------------------------------------------------
#  Grid

class Grid1D(object):
    function = ptx._ptx_grid1d, (types.uint32,), types.intp

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder

        fname_tidx = ptx.SREG_MAPPING[ptx._ptx_sreg_tidx]
        fname_ntidx = ptx.SREG_MAPPING[ptx._ptx_sreg_ntidx]
        fname_ctaidx = ptx.SREG_MAPPING[ptx._ptx_sreg_ctaidx]

        li32 = types.uint32.llvm_as_value()
        fn_tidx = cgutils.get_function(builder, fname_tidx, li32, ())
        fn_ntidx = cgutils.get_function(builder, fname_ntidx, li32, ())
        fn_ctaidx = cgutils.get_function(builder, fname_ctaidx, li32, ())

        tidx = builder.call(fn_tidx, ())
        ntidx = builder.call(fn_ntidx, ())
        ctaidx = builder.call(fn_ctaidx, ())

        tidx = context.cast(tidx, types.uint32, retty)
        ntidx = context.cast(ntidx, types.uint32, retty)
        ctaidx = context.cast(ctaidx, types.uint32, retty)

        return builder.add(tidx, builder.mul(ntidx, ctaidx))


class Grid2D(object):
    return_type = types.tupletype(types.intp, types.intp)
    function = ptx._ptx_grid2d, (types.uint32,), return_type

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder

        fname_tidx = ptx.SREG_MAPPING[ptx._ptx_sreg_tidx]
        fname_tidy = ptx.SREG_MAPPING[ptx._ptx_sreg_tidy]
        fname_ntidx = ptx.SREG_MAPPING[ptx._ptx_sreg_ntidx]
        fname_ntidy = ptx.SREG_MAPPING[ptx._ptx_sreg_ntidy]
        fname_ctaidx = ptx.SREG_MAPPING[ptx._ptx_sreg_ctaidx]
        fname_ctaidy = ptx.SREG_MAPPING[ptx._ptx_sreg_ctaidy]

        li32 = types.uint32.llvm_as_value()
        fn_tidx = cgutils.get_function(builder, fname_tidx, li32, ())
        fn_tidy = cgutils.get_function(builder, fname_tidy, li32, ())
        fn_ntidx = cgutils.get_function(builder, fname_ntidx, li32, ())
        fn_ntidy = cgutils.get_function(builder, fname_ntidy, li32, ())
        fn_ctaidx = cgutils.get_function(builder, fname_ctaidx, li32, ())
        fn_ctaidy = cgutils.get_function(builder, fname_ctaidy, li32, ())

        tidx = builder.call(fn_tidx, ())
        tidy = builder.call(fn_tidy, ())
        ntidx = builder.call(fn_ntidx, ())
        ntidy = builder.call(fn_ntidy, ())
        ctaidx = builder.call(fn_ctaidx, ())
        ctaidy = builder.call(fn_ctaidy, ())

        tidx = context.cast(tidx, types.uint32, types.intp)
        tidy = context.cast(tidy, types.uint32, types.intp)
        ntidx = context.cast(ntidx, types.uint32, types.intp)
        ntidy = context.cast(ntidy, types.uint32, types.intp)
        ctaidx = context.cast(ctaidx, types.uint32, types.intp)
        ctaidy = context.cast(ctaidy, types.uint32, types.intp)

        x = builder.add(tidx, builder.mul(ntidx, ctaidx))
        y = builder.add(tidy, builder.mul(ntidy, ctaidy))

        return retty.desc.llvm_pack(builder, (x, y))

#-------------------------------------------------------------------------------
#  Syncthreads


class Syncthreads(object):
    function = ptx.syncthreads, (), types.void

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        fname = 'llvm.nvvm.barrier0'
        sync = cgutils.get_function(builder, fname, lc.Type.void(), ())
        return builder.call(sync, ())

#-------------------------------------------------------------------------------
#  Atomics


def atomic_value_template(args):
    return args[0].desc.element


class AtomicAdd(object):
    function = ptx.atomic.add, (types.ArrayKind, types.intp,
                                atomic_value_template), atomic_value_template

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ary, idx, val = args
        aryty, idxty, valty = argtys
        data = aryutils.getdata(builder, ary)
        shape = aryutils.getshape(builder, ary)
        strides = aryutils.getstrides(builder, ary)
        order = aryty.desc.order
        indices = arylib.wraparound(context, ary, [idx])

        ptr = aryutils.getpointer(builder, data, shape, strides, order, indices)


        castedval = context.cast(val, valty, retty)
        res = builder.atomic_rmw('add', ptr, castedval, 'monotonic')

        return res


def atomic_fixed_array(args):
    ary, idx, val = args
    if (isinstance(idx.desc, types.FixedArray) and
            ary.desc.ndim == idx.desc.length):
        return idx


class AtomicAddFixedArray(object):
    function = ptx.atomic.add, (types.ArrayKind, atomic_fixed_array,
                                atomic_value_template), atomic_value_template

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ary, idx, val = args
        aryty, idxty, valty = argtys
        data = aryutils.getdata(builder, ary)
        shape = aryutils.getshape(builder, ary)
        strides = aryutils.getstrides(builder, ary)
        order = aryty.desc.order

        indices = idxty.llvm_unpack(builder, idx)
        indices = [idxty.desc.element.llvm_cast(builder, i, types.intp)
                   for i in indices]
        indices = arylib.wraparound(context, ary, indices)

        ptr = aryutils.getpointer(builder, data, shape, strides, order, indices)

        castedval = context.cast(val, valty, retty)
        res = builder.atomic_rmw('add', ptr, castedval, 'monotonic')

        return res


def atomic_tuple(args):
    ary, idx, val = args
    if (isinstance(idx.desc, types.Tuple) and
            ary.desc.ndim == len(idx.desc.elements)):
        return idx


class AtomicAddTuple(object):
    function = ptx.atomic.add, (types.ArrayKind, atomic_tuple,
                                atomic_value_template), atomic_value_template

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ary, idx, val = args
        aryty, idxty, valty = argtys
        data = aryutils.getdata(builder, ary)
        shape = aryutils.getshape(builder, ary)
        strides = aryutils.getstrides(builder, ary)
        order = aryty.desc.order

        indices = idxty.llvm_unpack(builder, idx)
        indices = [t.llvm_cast(builder, i, types.intp)
                   for t, i in zip(idxty.desc.elements, indices)]
        indices = arylib.wraparound(context, ary, indices)

        ptr = aryutils.getpointer(builder, data, shape, strides, order, indices)

        castedval = context.cast(val, valty, retty)
        res = builder.atomic_rmw('add', ptr, castedval, 'monotonic')

        return res

#-------------------------------------------------------------------------------
#  Print

class FormatString(object):

    def __init__(self, context, args, argtys):
        self.context = context
        self.builder = context.builder
        self.lmod = context.builder.basic_block.function.module
        self.args = args
        self.argtys = argtys

    def call_vprintf(self, context):
        vprintf = nvvmutils.declare_vprint(self.lmod)
        self.builder.call(vprintf, (self.get_format_string(),
                                    self.get_va_args()))

    def get_format_string(self):
        fmtstr = []
        for aty in self.argtys:
            fs = self._determine_format(aty)
            fmtstr.append(fs)
        fmt = ' '.join(fmtstr) + '\n'
        pfmt = nvvmutils.declare_string(self.builder, fmt)
        return to_void_ptr(self.context, pfmt)

    def get_va_args(self):
        vaargs = []
        for val, typ in zip(self.args, self.argtys):
            values, align = self._determine_arg(val, typ)
            vaargs.append((values, align))

        # Calculate offset and total length for the required vaargs buffer
        offset = 0
        storeloc = []
        for vals, align in vaargs:
            aligndiff = offset % align
            if aligndiff != 0:         # is aligned?
                offset += align - aligndiff
            for v in vals:
                storeloc.append((offset, v))
                offset += 4

        # Note: Need space for 4 byte of zeros at the end on 4 byte alignment.
        #       We will over allocate to fix misalignment.
        vamemlen = offset + 8
        end = offset + (offset - offset % align)        # start of the end
        vamem = self.builder.alloca_array(lc.Type.int(),
                                          lc.Constant.int(lc.Type.int(),
                                                          vamemlen))

        # Store values to vaargs buffer
        for offset, val in storeloc:
            assert offset % 4 == 0
            constoffset = lc.Constant.int(lc.Type.int(), offset // 4)
            ptr = self.builder.gep(vamem, [constoffset], inbounds=True)
            self.builder.store(val, ptr)

        # Store trailing zero bytes
        ptr = self.builder.gep(vamem, [lc.Constant.int(lc.Type.int(),
                                                        end // 4)])
        self.builder.store(lc.Constant.null(ptr.type.pointee), ptr)

        return to_void_ptr(self.context, vamem)

    def _determine_format(self, typ):
        if isinstance(typ.desc, types.ConstString):
            return "%s"
        elif typ == types.int64:
            return "%lld"
        elif isinstance(typ.desc, types.Signed) and typ.desc.bitwidth <= 32:
            return "%d"
        elif isinstance(typ.desc, types.Unsigned) and typ.desc.bitwidth <= 32:
            return "%u"
        elif typ == types.boolean:
            return "%s"
        elif typ == types.float32 or typ == types.float64:
            return "%e"
        elif typ == types.complex128 or typ == types.complex64:
            return "(%e %+ej)"
        else:
            raise TypeError("%s cannot be printed" % typ)

    def _determine_arg(self, val, typ):
        if isinstance(typ.desc, types.ConstString):
            pstr = nvvmutils.declare_string(self.builder, typ.desc.text)
            return self._prepare_string(pstr)
        elif typ == types.int64:
            return split_lower_upper(self.builder, val), 8
        elif isinstance(typ.desc, types.Signed) and typ.desc.bitwidth <= 32:
            ext = self.builder.sext(val, lc.Type.int())
            return [ext], 4
        elif isinstance(typ.desc, types.Unsigned) and typ.desc.bitwidth <= 32:
            ext = self.builder.zext(val, lc.Type.int())
            return [ext], 4
        elif typ == types.boolean:
            ptrue = nvvmutils.declare_string(self.builder, "True")
            pfalse = nvvmutils.declare_string(self.builder, "False")
            pstr = self.builder.select(val, ptrue, pfalse)
            return self._prepare_string(pstr)
        elif typ == types.float32 or typ == types.float64:
            if typ == types.float32:
                val = self.builder.fpext(val, lc.Type.double())
            intval = self.builder.bitcast(val, lc.Type.int(64))
            lo, hi = split_lower_upper(self.builder, intval)
            return [lo, hi], 8
        elif typ == types.float64:
            intval = self.builder.bitcast(val, lc.Type.int(64))
            lo, hi = split_lower_upper(self.builder, intval)
            return [lo, hi], 8
        elif typ == types.complex128 or typ == types.complex64:
            real, imag = typ.desc.llvm_unpack(self.builder, val)
            if typ == types.complex64:
                real = self.builder.fpext(real, lc.Type.double())
                imag = self.builder.fpext(imag, lc.Type.double())
            intreal = self.builder.bitcast(real, lc.Type.int(64))
            intimag = self.builder.bitcast(imag, lc.Type.int(64))
            rlo, rhi = split_lower_upper(self.builder, intreal)
            ilo, ihi = split_lower_upper(self.builder, intimag)
            return [rlo, rhi, ilo, ihi], 8
        else:
            assert False

    def _prepare_string(self, strptr):
        ptrsize = tuple.__itemsize__ * 8
        addr = self.builder.ptrtoint(strptr, lc.Type.int(ptrsize))
        if ptrsize == 64:
            lo, hi = split_lower_upper(self.builder, addr)
            return [lo, hi], 8
        else:
            return [addr], 4


class PrintMany(object):
    function = __builtin__.print, None, types.void

    def generic_implement(self, context, args, argtys, retty):
        fs = FormatString(context, args, argtys)
        fs.call_vprintf(context)


def split_lower_upper(builder, intval):
    vec = builder.bitcast(intval, lc.Type.vector(lc.Type.int(), 2))
    lo = builder.extract_element(vec, lc.Constant.int(lc.Type.int(), 0))
    hi = builder.extract_element(vec, lc.Constant.int(lc.Type.int(), 1))
    return lo, hi


def to_void_ptr(context, ptr):
    voidptrty = lc.Type.pointer(lc.Type.int(8))
    return context.builder.bitcast(ptr, voidptrty)


NULL_PTR = lc.Constant.null(lc.Type.pointer(lc.Type.int(8)))


def print_string(context, text):
    builder = context.builder
    lmod = context.builder.basic_block.function.module
    ptr = nvvmutils.declare_string(builder, text)
    vprintf = nvvmutils.declare_vprint(lmod)
    builder.call(vprintf, (to_void_ptr(context, ptr), NULL_PTR))


def push_to_local(context, val):
    p = context.builder.alloca(val.type)
    context.builder.store(val, p)
    return to_void_ptr(context, p)


def print_format(context, fmt, val):
    builder = context.builder
    lmod = context.builder.basic_block.function.module
    vprintf = nvvmutils.declare_vprint(lmod)
    fmtptr = to_void_ptr(context, nvvmutils.declare_string(builder, fmt))

    callargs = (fmtptr, push_to_local(context, val))
    builder.call(vprintf, callargs)


def param_conststring(args):
    [astr] = args
    if isinstance(astr.desc, types.ConstString):
        return astr


def param_int64(args):
    [aint] = args
    if isinstance(aint.desc, types.Integer):
        return aint


class PrintInlineString(object):
    function = intrinsics.print_inline, (param_conststring,), types.void

    def generic_implement(self, context, args, argtys, retty):
        [txtty] = argtys
        print_string(context, txtty.desc.text)
        print_string(context, ' ')


class PrintInlineInt64(object):
    function = intrinsics.print_inline, (param_int64,), types.void

    def generic_implement(self, context, args, argtys, retty):
        [val] = args
        print_format(context, '%lld ', val)


class PrintNewLine(object):
    function = intrinsics.print_newline, (), types.void

    def generic_implement(self, context, args, argtys, retty):
        if args:
            raise TypeError("expecting no arguments")
        print_string(context, '\n')

extensions = [
    # SReg
    TidX, TidY, TidZ,
    NTidX, NTidY, NTidZ,
    CTAidX, CTAidY,
    NCTAidX, NCTAidY,
    # Grid
    Grid1D, Grid2D,
    # Syncthreads,
    Syncthreads,
    # Atomic
    AtomicAdd, AtomicAddFixedArray, AtomicAddTuple,
    # Print
    PrintMany,
    PrintInlineString,
    PrintInlineInt64,
    PrintNewLine,
]

