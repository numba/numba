from __future__ import print_function, division, absolute_import
from llvm.core import (Type, Builder, inline_function, LINKAGE_INTERNAL,
                       ICMP_EQ, Constant)
from numba import types, cgutils, config


def _build_ufunc_loop_body(load, store, context, func, builder, arrays, out,
                           offsets, store_offset, signature):
    elems = load()

    # Compute
    status, retval = context.call_function(builder, func,
                                           signature.return_type,
                                           signature.args, elems)

    # Ignoring error status and store result
    # Store
    if out.byref:
        retval = builder.load(retval)

    store(retval)

    # increment indices
    for off, ary in zip(offsets, arrays):
        builder.store(builder.add(builder.load(off), ary.step), off)

    builder.store(builder.add(builder.load(store_offset), out.step),
                  store_offset)

    return status.code


def build_slow_loop_body(context, func, builder, arrays, out, offsets,
                            store_offset, signature):
    def load():
        elems = [ary.load_direct(builder.load(off))
                 for off, ary in zip(offsets, arrays)]
        return elems

    def store(retval):
        out.store_direct(retval, builder.load(store_offset))

    return _build_ufunc_loop_body(load, store, context, func, builder, arrays,
                                  out, offsets, store_offset, signature)


def build_fast_loop_body(context, func, builder, arrays, out, offsets,
                            store_offset, signature, ind):
    def load():
        elems = [ary.load_aligned(ind)
                 for ary in arrays]
        return elems

    def store(retval):
        out.store_aligned(retval, ind)

    return _build_ufunc_loop_body(load, store, context, func, builder, arrays,
                                  out, offsets, store_offset, signature)


def build_ufunc_wrapper(context, func, signature):
    """
    Wrap the scalar function with a loop that iterates over the arguments
    """
    module = func.module

    byte_t = Type.int(8)
    byte_ptr_t = Type.pointer(byte_t)
    byte_ptr_ptr_t = Type.pointer(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    intp_ptr_t = Type.pointer(intp_t)

    fnty = Type.function(Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                       intp_ptr_t, byte_ptr_t])

    wrapper = module.add_function(fnty, "__ufunc__." + func.name)
    arg_args, arg_dims, arg_steps, arg_data = wrapper.args
    arg_args.name = "args"
    arg_dims.name = "dims"
    arg_steps.name = "steps"
    arg_data.name = "data"

    builder = Builder.new(wrapper.append_basic_block("entry"))

    loopcount = builder.load(arg_dims, name="loopcount")

    actual_args = context.get_arguments(func)

    # Prepare inputs
    arrays = []
    for i, typ in enumerate(signature.args):
        arrays.append(UArrayArg(context, builder, arg_args, arg_steps, i,
                                context.get_argument_type(typ)))

    # Prepare output
    valty = context.get_data_type(signature.return_type)
    out = UArrayArg(context, builder, arg_args, arg_steps, len(actual_args), valty)

    # Setup indices
    offsets = []
    zero = context.get_constant(types.intp, 0)
    for _ in arrays:
        p = cgutils.alloca_once(builder, intp_t)
        offsets.append(p)
        builder.store(zero, p)

    store_offset = cgutils.alloca_once(builder, intp_t)
    builder.store(zero, store_offset)

    unit_strided = cgutils.true_bit
    for ary in arrays:
        unit_strided = builder.and_(unit_strided, ary.is_unit_strided)

    with cgutils.ifelse(builder, unit_strided) as (is_unit_strided,
                                                   is_strided):

        with is_unit_strided:
            with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
                fastloop = build_fast_loop_body(context, func, builder,
                                                arrays, out, offsets,
                                                store_offset, signature, ind)
            builder.ret_void()

        with is_strided:
            # General loop
            with cgutils.for_range(builder, loopcount, intp=intp_t):
                slowloop = build_slow_loop_body(context, func, builder,
                                                arrays, out, offsets,
                                                store_offset, signature)

            builder.ret_void()

    builder.ret_void()
    del builder

    # Set core function to internal so that it is not generated
    func.linkage = LINKAGE_INTERNAL
    # Force inline of code function
    inline_function(slowloop)
    inline_function(fastloop)
    # Run optimizer
    context.optimize(module)

    if config.DUMP_OPTIMIZED:
        print(module)

    return wrapper


class UArrayArg(object):
    def __init__(self, context, builder, args, steps, i, argtype):
        # Get data
        p = builder.gep(args, [context.get_constant(types.intp, i)])
        if cgutils.is_struct_ptr(argtype):
            self.byref = True
            self.data = builder.bitcast(builder.load(p), argtype)
        else:
            self.byref = False
            self.data = builder.bitcast(builder.load(p), Type.pointer(argtype))
            # Get step
        p = builder.gep(steps, [context.get_constant(types.intp, i)])
        abisize = context.get_constant(types.intp,
                                       context.get_abi_sizeof(argtype))
        self.step = builder.load(p)
        self.is_unit_strided = builder.icmp(ICMP_EQ, abisize, self.step)
        self.builder = builder

    def load(self, ind):
        offset = self.builder.mul(self.step, ind)
        return self.load_direct(offset)

    def load_direct(self, offset):
        intp_t = offset.type
        addr = self.builder.ptrtoint(self.data, intp_t)
        addr_off = self.builder.add(addr, offset)
        ptr = self.builder.inttoptr(addr_off, self.data.type)
        if self.byref:
            return ptr
        else:
            return self.builder.load(ptr)

    def load_aligned(self, ind):
        ptr = self.builder.gep(self.data, [ind])
        return self.builder.load(ptr)

    def store(self, value, ind):
        offset = self.builder.mul(self.step, ind)
        self.store_direct(value, offset)

    def store_direct(self, value, offset):
        addr = self.builder.ptrtoint(self.data, offset.type)
        addr_off = self.builder.add(addr, offset)
        ptr = self.builder.inttoptr(addr_off, self.data.type)
        assert ptr.type.pointee == value.type
        self.builder.store(value, ptr)

    def store_aligned(self, value, ind):
        ptr = self.builder.gep(self.data, [ind])
        self.builder.store(value, ptr)


def build_gufunc_wrapper(context, func, signature, sin, sout):
    module = func.module

    byte_t = Type.int(8)
    byte_ptr_t = Type.pointer(byte_t)
    byte_ptr_ptr_t = Type.pointer(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    intp_ptr_t = Type.pointer(intp_t)

    fnty = Type.function(Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                       intp_ptr_t, byte_ptr_t])

    wrapper = module.add_function(fnty, "__gufunc__." + func.name)
    arg_args, arg_dims, arg_steps, arg_data = wrapper.args
    arg_args.name = "args"
    arg_dims.name = "dims"
    arg_steps.name = "steps"
    arg_data.name = "data"

    builder = Builder.new(wrapper.append_basic_block("entry"))
    loopcount = builder.load(arg_dims, name="loopcount")

    # Unpack shapes
    unique_syms = set()
    for grp in (sin, sout):
        for syms in grp:
            unique_syms |= set(syms)

    sym_map = {}
    for grp in (sin, sout):
        for syms in sin:
            for s in syms:
                if s not in sym_map:
                    sym_map[s] = len(sym_map)

    sym_dim = {}
    for s, i in sym_map.items():
        sym_dim[s] = builder.load(builder.gep(arg_dims,
                                              [context.get_constant(types.intp,
                                                                    i + 1)]))

    # Prepare inputs
    arrays = []
    step_offset = len(sin) + len(sout)
    for i, (typ, sym) in enumerate(zip(signature.args, sin + sout)):
        ary = GUArrayArg(context, builder, arg_args, arg_dims, arg_steps, i,
                         step_offset, typ, sym, sym_dim)
        if not ary.as_scalar:
            step_offset += ary.ndim
        arrays.append(ary)

    # Loop
    with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
        args = [a.array_value for a in arrays]

        status, retval = context.call_function(builder, func,
                                               signature.return_type,
                                               signature.args, args)
        # ignore status
        # ignore retval

        for a in arrays:
            a.next(ind)

    builder.ret_void()

    # Set core function to internal so that it is not generated
    func.linkage = LINKAGE_INTERNAL
    # Force inline of code function
    inline_function(status.code)
    # Run optimizer
    context.optimize(module)

    if config.DUMP_OPTIMIZED:
        print(module)

    wrapper.verify()
    return wrapper


class GUArrayArg(object):
    def __init__(self, context, builder, args, dims, steps, i, step_offset,
                 typ, syms, sym_dim):

        self.context = context
        self.builder = builder

        if isinstance(typ, types.Array):
            self.dtype = typ.dtype
        else:
            self.dtype = typ

        self.syms = syms
        self.as_scalar = not syms

        if self.as_scalar:
            self.ndim = 1
        else:
            self.ndim = len(syms)

        core_step_ptr = builder.gep(steps,
                                    [context.get_constant(types.intp, i)],
                                    name="core.step.ptr")

        self.core_step = builder.load(core_step_ptr)
        self.strides = []
        for j in range(self.ndim):
            step = builder.gep(steps, [context.get_constant(types.intp,
                                                            step_offset + j)],
                               name="step.ptr")

            self.strides.append(builder.load(step))

        self.shape = []
        for s in syms:
            self.shape.append(sym_dim[s])

        data = builder.load(builder.gep(args,
                                        [context.get_constant(types.intp,
                                                              i)],
                                        name="data.ptr"),
                            name="data")

        self.data = data

        arytyp = types.Array(dtype=self.dtype, ndim=self.ndim, layout="A")
        arycls = context.make_array(arytyp)

        self.array = arycls(context, builder)
        self.array.data = builder.bitcast(self.data, self.array.data.type)
        if not self.as_scalar:
            self.array.shape = cgutils.pack_array(builder, self.shape)
            self.array.strides = cgutils.pack_array(builder, self.strides)
        else:
            one = context.get_constant(types.intp, 1)
            zero = context.get_constant(types.intp, 0)
            self.array.shape = cgutils.pack_array(builder, [one])
            self.array.strides = cgutils.pack_array(builder, [zero])
        self.array_value = self.array._getpointer()

    def next(self, i):
        intp_t = i.type
        array_data = self.array.data
        addr = self.builder.ptrtoint(array_data, intp_t)
        addr_new = self.builder.add(addr, self.core_step)
        self.array.data = self.builder.inttoptr(addr_new, array_data.type)

