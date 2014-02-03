from __future__ import print_function, division, absolute_import
from llvm.core import Type, Builder, TYPE_POINTER
from numba import types, cgutils


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
    out = UArrayArg(context, builder, arg_args, arg_steps, len(actual_args),
                    context.get_value_type(signature.return_type))

    # Loop
    with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
        # Load
        elems = [ary.load(ind) for ary in arrays]

        # Compute
        status, retval = context.call_function(builder, func, signature.args,
                                               elems)
        # Ignoring error status and store result

        # Store
        if out.byref:
            retval = builder.load(retval)

        out.store(retval, ind)

    builder.ret_void()
    return wrapper


class UArrayArg(object):
    def __init__(self, context, builder, args, steps, i, argtype):
        # Get data
        p = builder.gep(args, [context.get_constant(types.intp, i)])
        if argtype.kind == TYPE_POINTER:
            self.byref = True
            self.data = builder.bitcast(builder.load(p), argtype)
        else:
            self.byref = False
            self.data = builder.bitcast(builder.load(p), Type.pointer(argtype))
        # Get step
        p = builder.gep(steps, [context.get_constant(types.intp, i)])
        self.step = builder.load(p)

        self.builder = builder

    def load(self, ind):
        intp_t = ind.type
        addr = self.builder.ptrtoint(self.data, intp_t)
        addr_off = self.builder.add(addr, self.builder.mul(self.step, ind))
        ptr = self.builder.inttoptr(addr_off, self.data.type)
        if self.byref:
            return ptr
        else:
            return self.builder.load(ptr)

    def store(self, value, ind):
        addr = self.builder.ptrtoint(self.data, ind.type)
        addr_off = self.builder.add(addr, self.builder.mul(self.step, ind))
        ptr = self.builder.inttoptr(addr_off, self.data.type)
        assert ptr.type.pointee == value.type
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
        step_offset += ary.ndim
        arrays.append(ary)

    # Loop
    with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
        args = [a.array_value for a in arrays]
        status, retval = context.call_function(builder, func, signature.args,
                                               args)
        # ignore status
        # ignore retval

        for a in arrays:
            a.next(ind)

    builder.ret_void()

    wrapper.verify()
    return wrapper


class GUArrayArg(object):
    def __init__(self, context, builder, args, dims, steps, i, step_offset,
                 typ, syms, sym_dim):
        if isinstance(typ, types.Array):
            self.dtype = typ.dtype
        else:
            self.dtype = typ

        self.syms = syms

        self.ndim = len(syms)

        core_step_ptr = builder.gep(steps, [context.get_constant(types.intp,
                                                                 i)],
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
        self.array.shape = cgutils.pack_array(builder, self.shape)
        self.array.strides = cgutils.pack_array(builder, self.strides)
        self.array_value = self.array._getvalue()

        self.builder = builder

    def next(self, i):
        intp_t = i.type
        array_data = self.array.data
        addr = self.builder.ptrtoint(array_data, intp_t)
        addr_new = self.builder.add(addr, self.core_step)
        self.array.data = self.builder.inttoptr(addr_new, array_data.type)