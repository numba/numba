from __future__ import print_function, division, absolute_import

import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys

import numpy as np

from .. import compiler, ir, types, rewrites, six, cgutils, sigutils
from numba.ir_utils import *
from ..typing import npydecl, signature
from ..targets import npyimpl, imputils
from .dufunc import DUFunc
from .array_exprs import _is_ufunc, _unaryops, _binops, _cmpops
from numba import config
import llvmlite.llvmpy.core as lc
from numba.parfor2 import LoopNest
import numba
from numba import parfor2
import copy

# Lowerer that converts parfor2 instr to LLVM code.
def _lower_parfor2_parallel(lowerer, parfor):
    # 
    typingctx = lowerer.context.typing_context
    targetctx = lowerer.context
    typemap = lowerer.fndesc.typemap

    # produce instructions for init_block
    if config.DEBUG_ARRAY_OPT:
        print("init_block = ", parfor.init_block, " ", type(parfor.init_block))
    for instr in parfor.init_block.body:
        if config.DEBUG_ARRAY_OPT:
            print("lower init_block instr = ", instr)
        lowerer.lower_inst(instr)

    # compile parfor body as a separate function to be used with GUFuncWrapper
    flags = compiler.Flags()
    flags.set('error_model', 'numpy')
    func, func_args, func_sig = _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, {})

    # get the shape signature
    array_shape_classes = parfor.array_analysis.array_shape_classes
    func_args = ['sched'] + func_args
    num_inputs = len(func_args) - len(parfor2.get_parfor_outputs(parfor))
    print("num_inputs = ", num_inputs)
    print("parfor_outputs = ", parfor2.get_parfor_outputs(parfor))
    gu_signature = _create_shape_signature(array_shape_classes, num_inputs, func_args, func_sig)
    loop_ranges = [l.range_variable.name for l in parfor.loop_nests]
    # call the func in parallel by wrapping it with ParallelGUFuncWrapper
    array_size_vars = parfor.array_analysis.array_size_vars
    call_parallel_gufunc(lowerer, func, gu_signature, func_sig, func_args, loop_ranges, array_size_vars)

'''Create shape signature for GUFunc
'''
def _create_shape_signature(classes, num_inputs, args, func_sig):
    max_shape_num = max(sum([list(x) for x in classes.values()], []))
    gu_sin = []
    gu_sout = []
    count = 0
    for var, typ in zip(args, func_sig.args):
        print("create_shape_signature: var = ", var, " typ = ", typ)
        count = count + 1
        if isinstance(typ, types.Array):
            if var in classes:
                var_shape = classes[var]
                assert len(var_shape) == typ.ndim
            else:
                var_shape = []
                for i in range(typ.ndim):
                    max_shape_num = max_shape_num + 1
                    var_shape.append(max_shape_num)
            dim_syms = tuple([ chr(97 + i) for i in var_shape ]) # chr(97) = 'a'
        else:
            dim_syms = ()
        if (count > num_inputs):
            gu_sout.append(dim_syms)
        else:
            gu_sin.append(dim_syms)
    return (gu_sin, gu_sout)

numba.parfor2.lower_parfor2_parallel = _lower_parfor2_parallel

def _print_body(body_dict):
    for label, block in body_dict.items():
        print("label: ", label)
        for i, inst in enumerate(block.body):
            print("    ", i, " ", inst)

def _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, locals):
    loop_body = copy.deepcopy(parfor.loop_body)

    parfor_dim = len(parfor.loop_nests)
    assert parfor_dim==1
    loop_indices = [l.index_variable.name for l in parfor.loop_nests]

    # Get all the parfor params.
    parfor_params = parfor2.get_parfor_params(parfor)
    # Get just the outputs of the parfor.
    parfor_outputs = parfor2.get_parfor_outputs(parfor)
    # Compute just the parfor inputs as a set difference.
    parfor_inputs = list(set(parfor_params) - set(parfor_outputs))
    # Reorder all the params so that inputs go first then outputs.
    parfor_params = parfor_inputs + parfor_outputs

    if config.DEBUG_ARRAY_OPT==1:
        print("parfor_params = ", parfor_params, " ", type(parfor_params))
        #print("loop_ranges = ", loop_ranges, " ", type(loop_ranges))
        print("loop_indices = ", loop_indices, " ", type(loop_indices))
        print("loop_body = ", loop_body, " ", type(loop_body))
        _print_body(loop_body)

    param_dict = legalize_names(parfor_params)
    if config.DEBUG_ARRAY_OPT==1:
        print("param_dict = ", param_dict, " ", type(param_dict))

    ind_dict = legalize_names(loop_indices)
    legal_loop_indices = [ ind_dict[v] for v in loop_indices]
    if config.DEBUG_ARRAY_OPT==1:
        print("ind_dict = ", ind_dict, " ", type(ind_dict))
        print("legal_loop_indices = ", legal_loop_indices, " ", type(legal_loop_indices))

    for pd in parfor_params:
        print("pd = ", pd)
        print("pd type = ", typemap[pd], " ", type(typemap[pd]))

    param_types = [ typemap[v] for v in parfor_params ]
    if config.DEBUG_ARRAY_OPT==1:
        param_types_dict = { v:typemap[v] for v in parfor_params }
        print("param_types_dict = ", param_types_dict, " ", type(param_types_dict))
        print("param_types = ", param_types, " ", type(param_types))

    replace_var_names(loop_body, param_dict)
    parfor_args = parfor_params # remember the name before legalizing as the actual arguments
    parfor_params = [ param_dict[v] for v in parfor_params ]
    replace_var_names(loop_body, ind_dict)

    if config.DEBUG_ARRAY_OPT==1:
        print("legal parfor_params = ", parfor_params, " ", type(parfor_params))


    
    #loop_ranges_dict = legalize_names(loop_ranges)
    #loop_ranges = [ loop_ranges_dict[v] for v in loop_ranges ]

    #if config.DEBUG_ARRAY_OPT==1:
    #    print("legal loop_ranges ", type(loop_ranges), loop_ranges)

    # Determine the unique names of the scheduling and gufunc functions.
    # sched_func_name = "__numba_parfor_sched_%s" % (hex(hash(parfor)).replace("-", "_"))
    gufunc_name = "__numba_parfor_gufunc_%s" % (hex(hash(parfor)).replace("-", "_"))
    if config.DEBUG_ARRAY_OPT:
        # print("sched_func_name ", type(sched_func_name), " ", sched_func_name)
        print("gufunc_name ", type(gufunc_name), " ", gufunc_name)

    # Create the gufunc function.
    gufunc_txt = "def " + gufunc_name + "(sched, " + (", ".join(parfor_params)) + "):\n"
    for eachdim in range(parfor_dim):
        for indent in range(eachdim+1):
            gufunc_txt += "    "
        gufunc_txt += ( "for " + legal_loop_indices[eachdim] + " in range(sched[" + str(eachdim)
        + "], sched[" + str(eachdim + parfor_dim) + "] + 1):\n" )
    for indent in range(parfor_dim+1):
        gufunc_txt += "    "
    gufunc_txt += "__sentinel__ = 0\n"
    gufunc_txt += "    return None\n"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
    exec(gufunc_txt)
    gufunc_func = eval(gufunc_name)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_func = ", type(gufunc_func), "\n", gufunc_func)
    gufunc_ir = compiler.run_frontend(gufunc_func)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir dump ", type(gufunc_ir))
        gufunc_ir.dump()
        print("loop_body dump ", type(loop_body))
        _print_body(loop_body)

    gufunc_param_types = [numba.types.npytypes.Array(numba.int64, 1, "C")] + param_types
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_param_types = ", type(gufunc_param_types), "\n", gufunc_param_types)

    gufunc_stub_last_label = max(gufunc_ir.blocks.keys())

    # Add gufunc stub last label to each parfor.loop_body label to prevent label conflicts.
    loop_body = add_offset_to_labels(loop_body, gufunc_stub_last_label)
    if config.DEBUG_ARRAY_OPT:
        _print_body(loop_body)

    for label, block in gufunc_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name=="__sentinel__":
                loc = inst.loc
                scope = block.scope
                # split block across __sentinel__
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]
                block.body = block.body[i+1:]
                new_label = next_label()
                body_first_label = min(loop_body.keys())
                prev_block.append(ir.Jump(body_first_label, loc))
                for (l, b) in loop_body.items():
                    gufunc_ir.blocks[l] = b
                body_last_label = max(loop_body.keys())
                gufunc_ir.blocks[new_label] = block
                gufunc_ir.blocks[label] = prev_block
                gufunc_ir.blocks[body_last_label].append(ir.Jump(new_label, loc))
                break
        else:
            continue
        break
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir last dump")
        gufunc_ir.dump()

    kernel_func = compiler.compile_ir(typingctx, targetctx, gufunc_ir, gufunc_param_types, types.none, flags, locals)

    kernel_sig = signature(types.none, *gufunc_param_types)
    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return kernel_func, parfor_args, kernel_sig

def _prepare_arguments(lowerer, gu_signature, outer_sig, expr_args):
    context = lowerer.context
    builder = lowerer.builder
    sin, sout = gu_signature
    num_inputs = len(sin)
    num_args = len(outer_sig.args)
    arguments = []
    inputs = []
    output = None
    out_ty = None
    input_sig_args = outer_sig.args[:num_inputs]
    for i in range(num_args):
        arg_ty = outer_sig.args[i]
        #print("arg_ty = ", arg_ty)
        if i < num_inputs:
            #print("as input")
            var = lowerer.loadvar(expr_args[i])
            arg = npyimpl._prepare_argument(context, builder, var, arg_ty)
            arguments.append(arg)
            inputs.append(arg)
        else:
            if isinstance(arg_ty, types.ArrayCompatible):
                #print("as output array")
                # output = npyimpl._build_array(context, builder, arg_ty, input_sig_args, inputs)
                var = lowerer.loadvar(expr_args[i])
                output = npyimpl._prepare_argument(context, builder, var, arg_ty)
                out_ty = arg_ty
                arguments.append(output)
            else:
                #print("as output scalar")
                output = npyimpl._prepare_argument(context, builder,
                         lc.Constant.null(context.get_value_type(arg_ty)), arg_ty)
                out_ty = arg_ty
                arguments.append(output)
    return inputs, output, out_ty


def call_parallel_gufunc(lowerer, cres, gu_signature, outer_sig, expr_args, loop_ranges, array_size_vars):
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library

    from .parallel import ParallelGUFuncBuilder, build_gufunc_wrapper, get_thread_count, _launch_threads, _init
    #from .ufuncbuilder import GUFuncBuilder, build_gufunc_wrapper #, _launch_threads, _init

    if config.DEBUG_ARRAY_OPT:
        print("make_parallel_loop")
        print("args = ", expr_args)
        print("outer_sig = ", outer_sig.args, outer_sig.return_type, outer_sig.recvr, outer_sig.pysig)
        #print("inner_sig = ", inner_sig.args, inner_sig.return_type, inner_sig.recvr, inner_sig.pysig)
    # The ufunc takes 4 arguments: args, dims, steps, data
    sin, sout = gu_signature
    
    # build the GUFunc
    ufunc = ParallelGUFuncBuilder(cres.entry_point, gu_signature)
    args, return_type = sigutils.normalize_signature(outer_sig)
    sig = ufunc._finalize_signature(cres, args, return_type)
    ufunc._sigs.append(sig)
    ufunc._cres[sig] = cres

    if config.DEBUG_ARRAY_OPT:
        print("_sigs = ", ufunc._sigs)
    sig = ufunc._sigs[0]
    cres = ufunc._cres[sig]
    #dtypenums, wrapper, env = ufunc.build(cres, sig)
    _launch_threads()
    _init()
    llvm_func = cres.library.get_function(cres.fndesc.llvm_func_name)
    wrapper_ptr, env, wrapper_name = build_gufunc_wrapper(llvm_func, cres, sin, sout, {})
    cres.library._ensure_finalized()

    if config.DEBUG_ARRAY_OPT:
        print("parallel function = ", wrapper_name, cres, sig)

    byte_t = lc.Type.int(8)
    byte_ptr_t = lc.Type.pointer(byte_t)
    byte_ptr_ptr_t = lc.Type.pointer(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    uintp_t = context.get_value_type(types.uintp)
    intp_ptr_t = lc.Type.pointer(intp_t)
    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)
    sizeof_intp = context.get_abi_sizeof(intp_t)

    # prepare sched, first pop it out of expr_args, outer_sig, and gu_signature
    sched_name = expr_args.pop(0)
    sched_typ = outer_sig.args[0]
    _outer_sig = signature(types.none, *(outer_sig.args[1:]))
    sched_sig  = sin.pop(0)
    # prepare input/output arguments
    inputs, output, out_ty = _prepare_arguments(lowerer, gu_signature, _outer_sig, expr_args)

    # call do_scheduling with appropriate arguments
    num_dim = len(output.shape)
    out_dims = cgutils.alloca_once(builder, intp_t, size = context.get_constant(types.intp, num_dim), name = "dims")
    for i in range(num_dim):
        builder.store(output.shape[i], builder.gep(out_dims, [context.get_constant(types.intp, i)]))
    sched_size = get_thread_count() * num_dim * 2
    sched = cgutils.alloca_once(builder, intp_t, size = context.get_constant(types.intp, sched_size), name = "sched")
    scheduling_fnty = lc.Type.function(intp_ptr_t, [intp_t, intp_ptr_t, uintp_t, intp_ptr_t])
    do_scheduling = builder.module.get_or_insert_function(scheduling_fnty, name="do_scheduling")
    builder.call(do_scheduling, [context.get_constant(types.intp, num_dim), out_dims,
                                 context.get_constant(types.uintp, get_thread_count()), sched])

    if config.DEBUG_ARRAY_OPT:
      for i in range(get_thread_count()):
        cgutils.printf(builder, "sched[" + str(i) + "] = ")
        for j in range(num_dim * 2):
            cgutils.printf(builder, "%d ", builder.load(builder.gep(sched, [context.get_constant(types.intp, i * num_dim * 2 + j)])))
        cgutils.printf(builder, "\n")

    # prepare arguments: args, dims, steps, data
    all_args = inputs + [output]
    num_args = len(all_args)
    num_inps = len(inputs)
    args = cgutils.alloca_once(builder, byte_ptr_t, size = context.get_constant(types.intp, 1 + num_args), name = "pargs")
    builder.store(builder.bitcast(sched, byte_ptr_t), args)

    for i in range(num_args):
        arg = all_args[i]
        dst = builder.gep(args, [context.get_constant(types.intp, i + 1)])
        if isinstance(arg, npyimpl._ArrayHelper):
            builder.store(builder.bitcast(arg.data, byte_ptr_t), dst)
        else:
            if i < num_inps:
                # Scalar input, must store the value first
                builder.store(arg.val, arg._ptr)
            builder.store(builder.bitcast(arg._ptr, byte_ptr_t), dst)

    # Next, we prepare the individual dimension info recorded in gu_signature
    sig_dim_dict = {}
    occurances = []
    occurances = [sched_sig[0]]
    sig_dim_dict[sched_sig[0]] = context.get_constant(types.intp, 2 * num_dim)
    for var, gu_sig in zip(all_args, sin + sout):
        for sig in gu_sig:
            i = 0
            for dim_sym in sig:
                sig_dim_dict[dim_sym] = var.shape[i]
                if not (dim_sym in occurances):
                    occurances.append(dim_sym)
                i = i + 1

    # prepare dims, which is only a single number, since N-D arrays is treated as 1D array by ufunc
    ndims = len(sig_dim_dict) + 1
    dims = cgutils.alloca_once(builder, intp_t, size = ndims, name = "pshape")
    # For now, outer loop dimension is two
    builder.store(context.get_constant(types.intp, get_thread_count()), dims)
    # dimension for sorted signature symbols follows
    i = 1
    for dim_sym in occurances:
        builder.store(sig_dim_dict[dim_sym], builder.gep(dims, [ context.get_constant(types.intp, i) ]))
        i = i + 1

    # prepare steps for each argument
    steps = cgutils.alloca_once(builder, intp_t, size = context.get_constant(types.intp, num_args + 1), name = "psteps")
    builder.store(context.get_constant(types.intp, 2 * num_dim * sizeof_intp), steps)
    for i in range(num_args):
        # all steps are 0
        # sizeof = context.get_abi_sizeof(context.get_value_type(arguments[i].base_type))
        # stepsize = context.get_constant(types.intp, sizeof)
        stepsize = zero
        #cgutils.printf(builder, "stepsize = %d\n", stepsize)
        dst = builder.gep(steps, [context.get_constant(types.intp, 1 + i)])
        builder.store(stepsize, dst)
    # steps for output array goes last
    # sizeof = context.get_abi_sizeof(context.get_value_type(output.base_type))
    # stepsize = context.get_constant(types.intp, sizeof)
    # cgutils.printf(builder, "stepsize = %d\n", stepsize)
    # dst = builder.gep(steps, [lc.Constant.int(lc.Type.int(), num_args)])
    # builder.store(stepsize, dst)

    # prepare data
    data = builder.inttoptr(zero, byte_ptr_t)

    #result = context.call_function_pointer(builder, wrapper, [args, dims, steps, data])
    fnty = lc.Type.function(lc.Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                             intp_ptr_t, byte_ptr_t])
    fn = builder.module.get_or_insert_function(fnty, name=wrapper_name)
    #cgutils.printf(builder, "before calling kernel %p\n", fn)
    result = builder.call(fn, [args, dims, steps, data])
    #cgutils.printf(builder, "after calling kernel %p\n", fn)
    if config.DEBUG_ARRAY_OPT:
        print("result = ", result)

    # return builder.bitcast(output.return_val, ret_ty)
    return imputils.impl_ret_new_ref(context, builder, out_ty, output.return_val)

    # cres = context.compile_subroutine_no_cache(builder, wrapper_func, outer_sig, flags=flags)
    # args = [lowerer.loadvar(name) for name in expr_args]
    # result = context.call_internal(builder, cres.fndesc, outer_sig, args)
    # status, res = context.call_conv.call_function(builder, cres.fndesc, outer_sig.return_type,
    #                                              outer_sig.args, expr_args)
    #with cgutils.if_unlikely(builder, status.is_error):
    #        context.call_conv.return_status_propagate(builder, status)
    # return res
