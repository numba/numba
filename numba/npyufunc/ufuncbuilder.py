# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
from llvm.core import Type, Builder, TYPE_POINTER
from numba.decorators import jit, target_registry
from numba.dispatcher import read_flags, GlobalContext, normalize_signature
from numba import types, utils, cgutils, compiler
from . import _internal


class UFuncDispatcher(object):
    def __init__(self, py_func):
        self.py_func = py_func
        self.overloads = utils.UniqueDict()

    def compile(self, sig, **kws):
        flags = compiler.Flags()
        read_flags(flags, kws)

        glctx = GlobalContext()
        typingctx = glctx.typing_context
        targetctx = glctx.target_context

        args, return_type = normalize_signature(sig)

        cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                      args=args, return_type=return_type,
                                      flags=flags)

        self.overloads[cres.signature] = cres


target_registry['npyufunc'] = UFuncDispatcher


class UFuncBuilder(object):
    def __init__(self, py_func):
        self.py_func = py_func
        self.nb_func = jit(target='npyufunc')(py_func)
        self.overloads = []

    def add(self, sig):
        self.nb_func.compile(sig, nocompile=True)

    def build_ufunc(self):
        dtypelist = []
        ptrlist = []
        if not self.nb_func:
            raise TypeError("No definition")
        for sig, cres in self.nb_func.overloads.items():
            dtypenums, ptr = self.build(cres)
            dtypelist.append(dtypenums)
            ptrlist.append(utils.longint(ptr))
        datlist = [None] * len(ptrlist)

        inct = len(cres.signature.args)
        outct = 1

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        ufunc = _internal.fromfunc(ptrlist, dtypelist, inct, outct, datlist)

        return ufunc

    def build(self, cres):
        # Buider wrapper for ufunc entry point
        ctx = cres.target_context
        signature = cres.signature
        wrapper = build_ufunc_wrapper(ctx, cres.llvm_func, signature)
        ctx.engine.add_module(wrapper.module)
        ptr = ctx.engine.get_pointer_to_function(wrapper)
        # Get dtypes
        dtypenums = [np.dtype(a.name).num for a in signature.args]
        dtypenums.append(np.dtype(signature.return_type.name).num)
        return dtypenums, ptr


def build_ufunc_wrapper(context, func, signature):
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

    arrays = []
    byrefs = []
    steps = []
    for i, (arg, argty) in enumerate(zip(actual_args, signature.args)):
        p = builder.gep(arg_args, [context.get_constant(types.intp, i)])
        actual_argtype = context.get_argument_type(argty)
        if actual_argtype.kind == TYPE_POINTER:
            byrefs.append(True)
            v = builder.bitcast(builder.load(p), actual_argtype)
        else:
            byrefs.append(False)
            v = builder.bitcast(builder.load(p),
                                Type.pointer(actual_argtype))
        arrays.append(v)

        p = builder.gep(arg_steps, [context.get_constant(types.intp, i)])
        v = builder.load(p)
        steps.append(v)

    outp = builder.gep(arg_args, [context.get_constant(types.intp,
                                                    len(actual_args))])

    if context.is_struct_type(signature.return_type):
        out = builder.bitcast(builder.load(outp),
                              context.get_value_type(signature.return_type))
        outstruct = True
    else:
        out = builder.bitcast(builder.load(outp),
                              Type.pointer(context.get_value_type(
                                                        signature.return_type)))
        outstruct = False

    outstepp = builder.gep(arg_steps,
                           [context.get_constant(types.intp,
                                              len(actual_args))])
    outstep = builder.load(outstepp)

    with cgutils.for_range(builder, loopcount, intp=intp_t) as ind:
        elems = []
        for ary, step, byref in zip(arrays, steps, byrefs):
            addr = builder.ptrtoint(ary, intp_t)
            addr_off = builder.add(addr, builder.mul(step, ind))
            ptr = builder.inttoptr(addr_off, ary.type)
            if byref:
                value = ptr
            else:
                value = builder.load(ptr)
            elems.append(value)

        status, retval = context.call_function(builder, func, elems)
        # ignoring error status and store result

        addr = builder.ptrtoint(out, intp_t)
        addr_off = builder.add(addr, builder.mul(outstep, ind))
        ptr = builder.inttoptr(addr_off, out.type)

        if outstruct:
            # return value is of structure type
            retval = builder.load(retval)

        assert ptr.type.pointee == retval.type
        builder.store(retval, ptr)

    builder.ret_void()
    return wrapper


