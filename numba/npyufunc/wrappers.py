from __future__ import print_function, division, absolute_import
import numpy as np
from llvmlite.llvmpy.core import (Type, Builder, LINKAGE_INTERNAL,
                                  ICMP_EQ, Constant)
import llvmlite.llvmpy.core as lc
from llvmlite import binding as ll

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

    if not cgutils.is_struct(retval.type):
        retval = context.get_value_as_argument(builder, signature.return_type,
                                               retval)

    store(retval)

    # increment indices
    for off, ary in zip(offsets, arrays):
        builder.store(builder.add(builder.load(off), ary.step), off)

    builder.store(builder.add(builder.load(store_offset), out.step),
                  store_offset)

    return status.code


def _build_ufunc_loop_body_objmode(load, store, context, func, builder,
                                   arrays, out, offsets, store_offset,
                                   signature, env):
    elems = load()

    # Compute
    _objargs = [types.pyobject] * len(signature.args)
    status, retval = context.call_function(builder, func, types.pyobject,
                                           _objargs, elems, env=env)

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


def build_obj_loop_body(context, func, builder, arrays, out, offsets,
                        store_offset, signature, pyapi, env):
    def load():
        # Load
        elems = [ary.load_direct(builder.load(off))
                 for off, ary in zip(offsets, arrays)]
        # Box
        elems = [pyapi.from_native_value(v, t)
                 for v, t in zip(elems, signature.args)]
        return elems

    def store(retval):
        is_error = cgutils.is_null(builder, retval)
        with cgutils.ifelse(builder, is_error) as (if_error, if_ok):
            with if_error:
                msg = context.insert_const_string(pyapi.module,
                                                  "object mode ufunc")
                msgobj = pyapi.string_from_string(msg)
                pyapi.err_write_unraisable(msgobj)
                pyapi.decref(msgobj)
            with if_ok:
                # Unbox
                retval = pyapi.to_native_value(retval, signature.return_type)
                # Store
                out.store_direct(retval, builder.load(store_offset))

    return _build_ufunc_loop_body_objmode(load, store, context, func, builder,
                                          arrays, out, offsets, store_offset,
                                          signature, env)


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


def build_ufunc_wrapper(library, context, func, signature, objmode, env):
    """
    Wrap the scalar function with a loop that iterates over the arguments
    """
    byte_t = Type.int(8)
    byte_ptr_t = Type.pointer(byte_t)
    byte_ptr_ptr_t = Type.pointer(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    intp_ptr_t = Type.pointer(intp_t)

    fnty = Type.function(Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                       intp_ptr_t, byte_ptr_t])

    wrapper_module = library.create_ir_module('')
    if objmode:
        func_type = context.get_function_type2(
            types.pyobject, [types.pyobject] * len(signature.args))
    else:
        func_type = context.get_function_type2(signature.return_type,
                                               signature.args)
    oldfunc = func
    func = wrapper_module.add_function(func_type,
                                       name=func.name)
    func.attributes.add("alwaysinline")

    wrapper = wrapper_module.add_function(fnty, "__ufunc__." + func.name)
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
    out = UArrayArg(context, builder, arg_args, arg_steps, len(actual_args),
                    valty)

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

    if objmode:
    # General loop
        pyapi = context.get_python_api(builder)
        gil = pyapi.gil_ensure()
        with cgutils.for_range(builder, loopcount, intp=intp_t):
            slowloop = build_obj_loop_body(context, func, builder,
                                           arrays, out, offsets,
                                           store_offset, signature,
                                           pyapi, env)
        pyapi.gil_release(gil)
        builder.ret_void()

    else:

        with cgutils.ifelse(builder, unit_strided) as (is_unit_strided,
                                                       is_strided):

            with is_unit_strided:
                with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
                    fastloop = build_fast_loop_body(context, func, builder,
                                                    arrays, out, offsets,
                                                    store_offset, signature,
                                                    ind)
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


    # Run optimizer
    library.add_ir_module(wrapper_module)
    wrapper = library.get_function(wrapper.name)
    oldfunc.linkage = LINKAGE_INTERNAL

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
        ptr = cgutils.pointer_add(self.builder, self.data, offset)
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
        ptr = cgutils.pointer_add(self.builder, self.data, offset)
        assert ptr.type.pointee == value.type, (ptr.type, value.type)
        self.builder.store(value, ptr)

    def store_aligned(self, value, ind):
        ptr = self.builder.gep(self.data, [ind])
        self.builder.store(value, ptr)


class _GufuncWrapper(object):
    def __init__(self, library, context, func, signature, sin, sout, fndesc,
                 env):
        self.library = library
        self.context = context
        self.func = func
        self.signature = signature
        self.sin = sin
        self.sout = sout
        self.fndesc = fndesc
        self.is_objectmode = self.signature.return_type == types.pyobject
        self.env = env

    def build(self):
        byte_t = Type.int(8)
        byte_ptr_t = Type.pointer(byte_t)
        byte_ptr_ptr_t = Type.pointer(byte_ptr_t)
        intp_t = self.context.get_value_type(types.intp)
        intp_ptr_t = Type.pointer(intp_t)

        fnty = Type.function(Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                           intp_ptr_t, byte_ptr_t])

        wrapper_module = self.library.create_ir_module('')
        func_type = self.context.get_function_type(self.fndesc)
        func = wrapper_module.add_function(func_type, name=self.func.name)
        func.attributes.add("alwaysinline")
        wrapper = wrapper_module.add_function(fnty,
                                              "__gufunc__." + self.func.name)
        arg_args, arg_dims, arg_steps, arg_data = wrapper.args
        arg_args.name = "args"
        arg_dims.name = "dims"
        arg_steps.name = "steps"
        arg_data.name = "data"

        builder = Builder.new(wrapper.append_basic_block("entry"))
        loopcount = builder.load(arg_dims, name="loopcount")

        # Unpack shapes
        unique_syms = set()
        for grp in (self.sin, self.sout):
            for syms in grp:
                unique_syms |= set(syms)

        sym_map = {}
        for syms in self.sin:
            for s in syms:
                if s not in sym_map:
                    sym_map[s] = len(sym_map)

        sym_dim = {}
        for s, i in sym_map.items():
            sym_dim[s] = builder.load(builder.gep(arg_dims,
                                                  [self.context.get_constant(
                                                      types.intp,
                                                      i + 1)]))

        # Prepare inputs
        arrays = []
        step_offset = len(self.sin) + len(self.sout)
        for i, (typ, sym) in enumerate(zip(self.signature.args,
                                           self.sin + self.sout)):
            ary = GUArrayArg(self.context, builder, arg_args, arg_dims,
                             arg_steps, i, step_offset, typ, sym, sym_dim)
            if not ary.as_scalar:
                step_offset += ary.ndim
            arrays.append(ary)

        bbreturn = cgutils.get_function(builder).append_basic_block('.return')

        # Prologue
        self.gen_prologue(builder)

        # Loop
        with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
            args = [a.array_value for a in arrays]
            innercall, error = self.gen_loop_body(builder, func, args)
            # If error, escape
            cgutils.cbranch_or_continue(builder, error, bbreturn)

            for a in arrays:
                a.next(ind)

        builder.branch(bbreturn)
        builder.position_at_end(bbreturn)

        # Epilogue
        self.gen_epilogue(builder)

        builder.ret_void()

        self.library.add_ir_module(wrapper_module)
        wrapper = self.library.get_function(wrapper.name)

        # Set core function to internal so that it is not generated
        self.func.linkage = LINKAGE_INTERNAL

        return wrapper, self.env

    def gen_loop_body(self, builder, func, args):
        status, retval = self.context.call_function(builder, func,
                                                    self.signature.return_type,
                                                    self.signature.args, args)

        innercall = status.code
        error = status.err
        return innercall, error

    def gen_prologue(self, builder):
        pass        # Do nothing

    def gen_epilogue(self, builder):
        pass        # Do nothing


class _GufuncObjectWrapper(_GufuncWrapper):
    def gen_loop_body(self, builder, func, args):
        innercall, error = _prepare_call_to_object_mode(self.context,
                                                        builder, func,
                                                        self.signature,
                                                        args, env=self.envptr)
        return innercall, error

    def gen_prologue(self, builder):
        #  Get an environment object for the function
        ll_intp = self.context.get_value_type(types.intp)
        ll_pyobj = self.context.get_value_type(types.pyobject)
        self.envptr = Constant.int(ll_intp, id(self.env)).inttoptr(ll_pyobj)

        # Acquire the GIL
        self.pyapi = self.context.get_python_api(builder)
        self.gil = self.pyapi.gil_ensure()

    def gen_epilogue(self, builder):
        # Release GIL
        self.pyapi.gil_release(self.gil)


def build_gufunc_wrapper(library, context, func, signature, sin, sout, fndesc,
                         env):
    wrapcls = (_GufuncObjectWrapper
               if signature.return_type == types.pyobject
               else _GufuncWrapper)
    return wrapcls(library, context, func, signature, sin, sout, fndesc,
                   env).build()


def _prepare_call_to_object_mode(context, builder, func, signature, args,
                                 env):
    mod = cgutils.get_module(builder)

    thisfunc = cgutils.get_function(builder)
    bb_core_return = thisfunc.append_basic_block('ufunc.core.return')

    pyapi = context.get_python_api(builder)

    # Call to
    # PyObject* ndarray_new(int nd,
    #       npy_intp *dims,   /* shape */
    #       npy_intp *strides,
    #       void* data,
    #       int type_num,
    #       int itemsize)

    ll_int = context.get_value_type(types.int32)
    ll_intp = context.get_value_type(types.intp)
    ll_intp_ptr = Type.pointer(ll_intp)
    ll_voidptr = context.get_value_type(types.voidptr)
    ll_pyobj = context.get_value_type(types.pyobject)
    fnty = Type.function(ll_pyobj, [ll_int, ll_intp_ptr,
                                    ll_intp_ptr, ll_voidptr,
                                    ll_int, ll_int])

    fn_array_new = mod.get_or_insert_function(fnty, name="numba_ndarray_new")

    # Convert each llarray into pyobject
    error_pointer = cgutils.alloca_once(builder, Type.int(1), name='error')
    builder.store(cgutils.true_bit, error_pointer)
    ndarray_pointers = []
    ndarray_objects = []
    for i, (arr, arrtype) in enumerate(zip(args, signature.args)):
        ptr = cgutils.alloca_once(builder, ll_pyobj)
        ndarray_pointers.append(ptr)

        builder.store(Constant.null(ll_pyobj), ptr)   # initialize to NULL

        arycls = context.make_array(arrtype)
        array = arycls(context, builder, ref=arr)

        zero = Constant.int(ll_int, 0)

        # Extract members of the llarray
        nd = Constant.int(ll_int, arrtype.ndim)
        dims = builder.gep(array._get_ptr_by_name('shape'), [zero, zero])
        strides = builder.gep(array._get_ptr_by_name('strides'), [zero, zero])
        data = builder.bitcast(array.data, ll_voidptr)
        dtype = np.dtype(str(arrtype.dtype))

        # Prepare other info for reconstruction of the PyArray
        type_num = Constant.int(ll_int, dtype.num)
        itemsize = Constant.int(ll_int, dtype.itemsize)

        # Call helper to reconstruct PyArray objects
        obj = builder.call(fn_array_new, [nd, dims, strides, data,
                                          type_num, itemsize])
        builder.store(obj, ptr)
        ndarray_objects.append(obj)

        obj_is_null = cgutils.is_null(builder, obj)
        builder.store(obj_is_null, error_pointer)
        cgutils.cbranch_or_continue(builder, obj_is_null, bb_core_return)

    # Call ufunc core function
    object_sig = [types.pyobject] * len(ndarray_objects)

    status, retval = context.call_function(builder, func, ll_pyobj, object_sig,
                                           ndarray_objects, env=env)
    builder.store(status.err, error_pointer)

    # Release returned object
    pyapi.decref(retval)

    builder.branch(bb_core_return)
    # At return block
    builder.position_at_end(bb_core_return)

    # Release argument object
    for ndary_ptr in ndarray_pointers:
        pyapi.decref(builder.load(ndary_ptr))

    innercall = status.code
    return innercall, builder.load(error_pointer)


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
        self.array.data = cgutils.pointer_add(self.builder,
                                              self.array.data, self.core_step)

