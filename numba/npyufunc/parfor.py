from __future__ import print_function, division, absolute_import

import ast
from collections import defaultdict, OrderedDict
import sys

from .. import compiler, ir, types, six, cgutils, sigutils
from numba.ir_utils import (add_offset_to_labels, replace_var_names,
                            remove_dels, legalize_names, mk_unique_var, 
			    rename_labels, get_name_var_table)
from ..typing import signature
from numba import config
import llvmlite.llvmpy.core as lc
import numba
import copy


def _lower_parfor_parallel(lowerer, parfor):
    """Lowerer that handles LLVM code generation for parfor.
    This function lowers a parfor IR node to LLVM.
    The general approach is as follows:
    1) The code from the parfor's init block is lowered normally
       in the context of the current function.
    2) The body of the parfor is transformed into a gufunc function.
    3) Code is inserted into the main function that calls do_scheduling
       to divide the iteration space for each thread, allocates
       reduction arrays, calls the gufunc function, and then invokes
       the reduction function across the reduction arrays to produce
       the final reduction values.
    """
    typingctx = lowerer.context.typing_context
    targetctx = lowerer.context
    typemap = lowerer.fndesc.typemap

    if config.DEBUG_ARRAY_OPT:
        print("_lower_parfor_parallel")
        parfor.dump()

    # produce instructions for init_block
    if config.DEBUG_ARRAY_OPT:
        print("init_block = ", parfor.init_block, " ", type(parfor.init_block))
    for instr in parfor.init_block.body:
        if config.DEBUG_ARRAY_OPT:
            print("lower init_block instr = ", instr)
        lowerer.lower_inst(instr)

    # run get_parfor_outputs() and get_parfor_reductions() before gufunc creation
    # since Jumps are modified so CFG of loop_body dict will become invalid
    parfor_output_arrays = numba.parfor.get_parfor_outputs(parfor)
    parfor_redvars, parfor_reddict = numba.parfor.get_parfor_reductions(parfor)
    # compile parfor body as a separate function to be used with GUFuncWrapper
    flags = compiler.Flags()
    flags.set('error_model', 'numpy')
    func, func_args, func_sig = _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, {})

    # get the shape signature
    array_shape_classes = parfor.array_analysis.array_shape_classes
    func_args = ['sched'] + func_args
    num_reductions = len(parfor_redvars)
    num_inputs = len(func_args) - len(parfor_output_arrays) - num_reductions
    if config.DEBUG_ARRAY_OPT:
        print("num_inputs = ", num_inputs)
        print("parfor_outputs = ", parfor_output_arrays)
        print("parfor_redvars = ", parfor_redvars)
    gu_signature = _create_shape_signature(array_shape_classes, num_inputs, num_reductions, func_args, func_sig)
    if config.DEBUG_ARRAY_OPT:
        print("gu_signature = ", gu_signature)

    # call the func in parallel by wrapping it with ParallelGUFuncBuilder
    loop_ranges = [(l.start, l.stop, l.step) for l in parfor.loop_nests]
    if config.DEBUG_ARRAY_OPT:
        print("loop_nests = ", parfor.loop_nests)
        print("loop_ranges = ", loop_ranges)
    array_size_vars = parfor.array_analysis.array_size_vars
    if config.DEBUG_ARRAY_OPT:
        print("array_size_vars = ", sorted(array_size_vars.items()))
    call_parallel_gufunc(lowerer, func, gu_signature, func_sig, func_args, loop_ranges, array_size_vars, parfor_redvars, parfor_reddict, parfor.init_block)
    if config.DEBUG_ARRAY_OPT:
        sys.stdout.flush()

# A work-around to prevent circular imports
numba.parfor.lower_parfor_parallel = _lower_parfor_parallel


def _create_shape_signature(classes, num_inputs, num_reductions, args, func_sig):
    '''Create shape signature for GUFunc
    '''
    num_inouts = len(args) - num_reductions
    # maximum class number for array shapes
    max_shape_num = max(sum([list(x) for x in classes.values()], []))
    if config.DEBUG_ARRAY_OPT:
        print("create_shape_signature = ", max_shape_num)
    gu_sin = []
    gu_sout = []
    count = 0
    syms_sin = ()
    for var, typ in zip(args, func_sig.args):
        # print("create_shape_signature: var = ", var, " typ = ", typ)
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
            # TODO: use prefix + class number instead of single char
            dim_syms = tuple([ chr(97 + i) for i in var_shape ]) # chr(97) = 'a'
        else:
            dim_syms = ()
        if (count > num_inouts):
            # assume all reduction vars are scalar
            gu_sout.append(())
        elif count > num_inputs and all([s in syms_sin for s in dim_syms]):
            # only when dim_syms are found in gu_sin, we consider this as output
            gu_sout.append(dim_syms)
        else:
            gu_sin.append(dim_syms)
            syms_sin += dim_syms
    return (gu_sin, gu_sout)

def _print_body(body_dict):
    '''Pretty-print a set of IR blocks.
    '''
    for label, block in body_dict.items():
        print("label: ", label)
        for i, inst in enumerate(block.body):
            print("    ", i, " ", inst)

def _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, locals):
    '''
    Takes a parfor and creates a gufunc function for its body.
    There are two parts to this function.
    1) Code to iterate across the iteration space as defined by the schedule.
    2) The parfor body that does the work for a single point in the iteration space.
    Part 1 is created as Python text for simplicity with a sentinel assignment to mark the point
    in the IR where the parfor body should be added.
    This Python text is 'exec'ed into existence and its IR retrieved with run_frontend.
    The IR is scanned for the sentinel assignment where that basic block is split and the IR
    for the parfor body inserted.
    '''

    # TODO: need copy?
    # The parfor body and the main function body share ir.Var nodes.
    # We have to do some replacements of Var names in the parfor body to make them
    # legal parameter names.  If we don't copy then the Vars in the main function also
    # would incorrectly change their name.
    loop_body = copy.copy(parfor.loop_body)

    parfor_dim = len(parfor.loop_nests)
    loop_indices = [l.index_variable.name for l in parfor.loop_nests]

    # Get all the parfor params.
    parfor_params = numba.parfor.get_parfor_params(parfor)
    # Get just the outputs of the parfor.
    parfor_outputs = numba.parfor.get_parfor_outputs(parfor)
    # Get all parfor reduction vars, and operators.
    parfor_redvars, parfor_reddict = numba.parfor.get_parfor_reductions(parfor)
    # Compute just the parfor inputs as a set difference.
    parfor_inputs = sorted(list(set(parfor_params) - set(parfor_outputs) - set(parfor_redvars)))

    if config.DEBUG_ARRAY_OPT==1:
        print("parfor_params = ", parfor_params, " ", type(parfor_params))
        print("parfor_outputs = ", parfor_outputs, " ", type(parfor_outputs))
        print("parfor_inputs = ", parfor_inputs, " ", type(parfor_inputs))
        print("parfor_redvars = ", parfor_redvars, " ", type(parfor_redvars))

    # Reduction variables are represented as arrays, so they go under different names.
    parfor_redarrs = []
    for var in parfor_redvars:
       arr = var + "_arr"
       parfor_redarrs.append(arr)
       typemap[arr] = types.npytypes.Array(typemap[var], 1, "C")

    # Reorder all the params so that inputs go first then outputs.
    parfor_params = parfor_inputs + parfor_outputs + parfor_redarrs

    if config.DEBUG_ARRAY_OPT==1:
        print("parfor_params = ", parfor_params, " ", type(parfor_params))
        #print("loop_ranges = ", loop_ranges, " ", type(loop_ranges))
        print("loop_indices = ", loop_indices, " ", type(loop_indices))
        print("loop_body = ", loop_body, " ", type(loop_body))
        _print_body(loop_body)

    # Some Var are not legal parameter names so create a dict of potentially illegal
    # param name to guaranteed legal name.
    param_dict = legalize_names(parfor_params + parfor_redvars)
    if config.DEBUG_ARRAY_OPT==1:
        print("param_dict = ", sorted(param_dict.items()), " ", type(param_dict))

    # Some loop_indices are not legal parameter names so create a dict of potentially illegal
    # loop index to guaranteed legal name.
    ind_dict = legalize_names(loop_indices)
    # Compute a new list of legal loop index names.
    legal_loop_indices = [ ind_dict[v] for v in loop_indices]
    if config.DEBUG_ARRAY_OPT==1:
        print("ind_dict = ", sorted(ind_dict.items()), " ", type(ind_dict))
        print("legal_loop_indices = ", legal_loop_indices, " ", type(legal_loop_indices))
        for pd in parfor_params:
            print("pd = ", pd)
            print("pd type = ", typemap[pd], " ", type(typemap[pd]))

    # Get the types of each parameter.
    param_types = [ typemap[v] for v in parfor_params ]
    #if config.DEBUG_ARRAY_OPT==1:
    #    param_types_dict = { v:typemap[v] for v in parfor_params }
    #    print("param_types_dict = ", param_types_dict, " ", type(param_types_dict))
    #    print("param_types = ", param_types, " ", type(param_types))

    # Replace illegal parameter names in the loop body with legal ones.
    replace_var_names(loop_body, param_dict)
    parfor_args = parfor_params # remember the name before legalizing as the actual arguments
    # Change parfor_params to be legal names.
    parfor_params = [ param_dict[v] for v in parfor_params ]
    # Change parfor body to replace illegal loop index vars with legal ones.
    replace_var_names(loop_body, ind_dict)

    if config.DEBUG_ARRAY_OPT==1:
        print("legal parfor_params = ", parfor_params, " ", type(parfor_params))

    # Determine the unique names of the scheduling and gufunc functions.
    # sched_func_name = "__numba_parfor_sched_%s" % (hex(hash(parfor)).replace("-", "_"))
    gufunc_name = "__numba_parfor_gufunc_%s" % (hex(hash(parfor)).replace("-", "_"))
    if config.DEBUG_ARRAY_OPT:
        # print("sched_func_name ", type(sched_func_name), " ", sched_func_name)
        print("gufunc_name ", type(gufunc_name), " ", gufunc_name)

    # Create the gufunc function.
    gufunc_txt = "def " + gufunc_name + "(sched, " + (", ".join(parfor_params)) + "):\n"
    # Add initialization of reduction variables
    for arr, var in zip(parfor_redarrs, parfor_redvars):
        gufunc_txt += "    " + param_dict[var] + "=" + param_dict[arr] + "[0]\n"
    # For each dimension of the parfor, create a for loop in the generated gufunc function.
    # Iterate across the proper values extracted from the schedule.
    # The form of the schedule is start_dim0, start_dim1, ..., start_dimN, end_dim0,
    # end_dim1, ..., end_dimN
    for eachdim in range(parfor_dim):
        for indent in range(eachdim+1):
            gufunc_txt += "    "
        sched_dim = eachdim
        gufunc_txt += ( "for " + legal_loop_indices[eachdim] + " in range(sched[" + str(sched_dim)
                      + "], sched[" + str(sched_dim + parfor_dim) + "] + 1):\n" )
    # Add the sentinel assignment so that we can find the loop body position in the IR.
    for indent in range(parfor_dim+1):
        gufunc_txt += "    "
    gufunc_txt += "__sentinel__ = 0\n"
    # Add assignments of reduction variables (for returning the value)
    for arr, var in zip(parfor_redarrs, parfor_redvars):
        gufunc_txt += "    " + param_dict[arr] + "[0] = " + param_dict[var] + "\n"
    gufunc_txt += "    return None\n"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
    # Force gufunc outline into existence.
    exec(gufunc_txt)
    gufunc_func = eval(gufunc_name)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_func = ", type(gufunc_func), "\n", gufunc_func)
    # Get the IR for the gufunc outline.
    gufunc_ir = compiler.run_frontend(gufunc_func)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir dump ", type(gufunc_ir))
        gufunc_ir.dump()
        print("loop_body dump ", type(loop_body))
        _print_body(loop_body)

    # rename all variables in gufunc_ir afresh
    var_table = get_name_var_table(gufunc_ir.blocks)
    new_var_dict = {}
    reserved_names = ["__sentinel__"] + list(param_dict.values()) + legal_loop_indices
    for name, var in var_table.items():
        if not (name in reserved_names):
            new_var_dict[name] = mk_unique_var(name)
    replace_var_names(gufunc_ir.blocks, new_var_dict)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir dump after renaming ")
        gufunc_ir.dump()

    gufunc_param_types = [numba.types.npytypes.Array(numba.intp, 1, "C")] + param_types
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_param_types = ", type(gufunc_param_types), "\n", gufunc_param_types)

    gufunc_stub_last_label = max(gufunc_ir.blocks.keys())

    # Add gufunc stub last label to each parfor.loop_body label to prevent label conflicts.
    loop_body = add_offset_to_labels(loop_body, gufunc_stub_last_label)
    # new label for splitting sentinel block
    new_label = max(loop_body.keys())+1
    if config.DEBUG_ARRAY_OPT:
        _print_body(loop_body)

    # Search all the block in the gufunc outline for the sentinel assignment.
    for label, block in gufunc_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name=="__sentinel__":
                # We found the sentinel assignment.
                loc = inst.loc
                scope = block.scope
                # split block across __sentinel__
                # A new block is allocated for the statements prior to the sentinel
                # but the new block maintains the current block label.
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]
                # The current block is used for statements after the sentinel.
                block.body = block.body[i+1:]
                # But the current block gets a new label.
                body_first_label = min(loop_body.keys())
                # The previous block jumps to the minimum labelled block of the
                # parfor body.
                prev_block.append(ir.Jump(body_first_label, loc))
                # Add all the parfor loop body blocks to the gufunc function's IR.
                for (l, b) in loop_body.items():
                    gufunc_ir.blocks[l] = b
                body_last_label = max(loop_body.keys())
                gufunc_ir.blocks[new_label] = block
                gufunc_ir.blocks[label] = prev_block
                # Add a jump from the last parfor body block to the block containing
                # statements after the sentinel.
                gufunc_ir.blocks[body_last_label].append(ir.Jump(new_label, loc))
                break
        else:
            continue
        break

    gufunc_ir.blocks = rename_labels(gufunc_ir.blocks)
    remove_dels(gufunc_ir.blocks)

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir last dump")
        gufunc_ir.dump()

    kernel_func = compiler.compile_ir(typingctx, targetctx, gufunc_ir,
                                gufunc_param_types, types.none, flags, locals)

    kernel_sig = signature(types.none, *gufunc_param_types)
    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return kernel_func, parfor_args, kernel_sig


def call_parallel_gufunc(lowerer, cres, gu_signature, outer_sig, expr_args,
                    loop_ranges, array_size_vars, redvars, reddict, init_block):
    '''
    Adds the call to the gufunc function from the main function.
    '''
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library

    from .parallel import (ParallelGUFuncBuilder, build_gufunc_wrapper,
                                get_thread_count, _launch_threads, _init)

    if config.DEBUG_ARRAY_OPT:
        print("make_parallel_loop")
        print("args = ", expr_args)
        print("outer_sig = ", outer_sig.args, outer_sig.return_type,
                                outer_sig.recvr, outer_sig.pysig)
        print("loop_ranges = ", loop_ranges)

    # Build the wrapper for GUFunc
    args, return_type = sigutils.normalize_signature(outer_sig)
    llvm_func = cres.library.get_function(cres.fndesc.llvm_func_name)
    sin, sout = gu_signature

    # These are necessary for build_gufunc_wrapper to find external symbols
    _launch_threads()
    _init()

    wrapper_ptr, env, wrapper_name = build_gufunc_wrapper(llvm_func, cres, sin,
                                                            sout, {})
    cres.library._ensure_finalized()

    if config.DEBUG_ARRAY_OPT:
        print("parallel function = ", wrapper_name, cres)

    # loadvars for loop_ranges
    def load_range(v):
        if isinstance(v, ir.Var):
            return lowerer.loadvar(v.name)
        else:
            return context.get_constant(types.intp, v)

    num_dim = len(loop_ranges)
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        start = load_range(start)
        stop = load_range(stop)
        assert(step == 1) # We do not support loop steps other than 1
        step = load_range(step)
        loop_ranges[i] = (start, stop, step)

        if config.DEBUG_ARRAY_OPT:
            print("call_parallel_gufunc loop_ranges[{}] = ".format(i), start,
                stop, step)
            cgutils.printf(builder, "loop range[{}]: %d %d (%d)\n".format(i),
                start, stop, step)


    # Commonly used LLVM types and constants
    byte_t = lc.Type.int(8)
    byte_ptr_t = lc.Type.pointer(byte_t)
    byte_ptr_ptr_t = lc.Type.pointer(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    uintp_t = context.get_value_type(types.uintp)
    intp_ptr_t = lc.Type.pointer(intp_t)
    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)
    sizeof_intp = context.get_abi_sizeof(intp_t)

    # Prepare sched, first pop it out of expr_args, outer_sig, and gu_signature
    sched_name = expr_args.pop(0)
    sched_typ = outer_sig.args[0]
    sched_sig = sin.pop(0)

    # Call do_scheduling with appropriate arguments
    dim_starts = cgutils.alloca_once(builder, intp_t,
            size = context.get_constant(types.intp, num_dim), name = "dims")
    dim_stops = cgutils.alloca_once(builder, intp_t,
            size = context.get_constant(types.intp, num_dim), name = "dims")
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        # substract 1 because do-scheduling takes inclusive ranges
        stop = builder.sub(stop, one)
        builder.store(start, builder.gep(dim_starts,
            [context.get_constant(types.intp, i)]))
        builder.store(stop, builder.gep(dim_stops,
            [context.get_constant(types.intp, i)]))
    sched_size = get_thread_count() * num_dim * 2
    sched = cgutils.alloca_once(builder, intp_t,
            size = context.get_constant(types.intp, sched_size), name = "sched")
    debug_flag = 1 if config.DEBUG_ARRAY_OPT else 0
    scheduling_fnty = lc.Type.function(intp_ptr_t,
        [intp_t, intp_ptr_t, intp_ptr_t, uintp_t, intp_ptr_t, intp_t])
    do_scheduling = builder.module.get_or_insert_function(scheduling_fnty,
                                                        name="do_scheduling")
    builder.call(do_scheduling, [context.get_constant(types.intp, num_dim),
        dim_starts, dim_stops,
        context.get_constant(types.uintp, get_thread_count()), sched,
        context.get_constant(types.intp, debug_flag)])

    # init reduction array allocation here.
    nredvars = len(redvars)
    ninouts = len(expr_args) - nredvars
    redarrs = []
    for i in range(nredvars):
        # arr = expr_args[-(nredvars - i)]
        val = lowerer.loadvar(redvars[i])
        # cgutils.printf(builder, "nredvar(" + redvars[i] + ") = %d\n", val)
        typ = context.get_value_type(lowerer.fndesc.typemap[redvars[i]])
        size = get_thread_count()
        arr = cgutils.alloca_once(builder, typ,
                                size = context.get_constant(types.intp, size))
        redarrs.append(arr)
        for j in range(size):
            dst = builder.gep(arr, [ context.get_constant(types.intp, j) ])
            builder.store(val, dst)

    if config.DEBUG_ARRAY_OPT:
      for i in range(get_thread_count()):
        cgutils.printf(builder, "sched[" + str(i) + "] = ")
        for j in range(num_dim * 2):
            cgutils.printf(builder, "%d ", builder.load(builder.gep(sched,
                    [context.get_constant(types.intp, i * num_dim * 2 + j)])))
        cgutils.printf(builder, "\n")

    # Prepare arguments: args, shapes, steps, data
    all_args = [ lowerer.loadvar(x) for x in expr_args[:ninouts] ] + redarrs
    num_args = len(all_args)
    num_inps = len(sin) + 1
    args = cgutils.alloca_once(builder, byte_ptr_t,
        size = context.get_constant(types.intp, 1 + num_args), name = "pargs")
    array_strides = []
    # sched goes first
    builder.store(builder.bitcast(sched, byte_ptr_t), args)
    array_strides.append(context.get_constant(types.intp, sizeof_intp))
    # followed by other arguments
    for i in range(num_args):
        arg = all_args[i]
        aty = outer_sig.args[i + 1] # skip first argument sched
        dst = builder.gep(args, [context.get_constant(types.intp, i + 1)])
        if i >= ninouts: # reduction variables
            builder.store(builder.bitcast(arg, byte_ptr_t), dst)
        elif isinstance(aty, types.ArrayCompatible):
            ary = context.make_array(aty)(context, builder, arg)
            strides = cgutils.unpack_tuple(builder, ary.strides, aty.ndim)
            for j in range(len(strides)):
                array_strides.append(strides[j])
            builder.store(builder.bitcast(ary.data, byte_ptr_t), dst)
        else:
            if i < num_inps:
                # Scalar input, need to store the value in an array of size 1
                typ = context.get_data_type(aty) if aty != types.boolean else lc.Type.int(1)
                ptr = cgutils.alloca_once(builder, typ)
                builder.store(arg, ptr)
            else:
                # Scalar output, must allocate
                typ = context.get_data_type(aty) if aty != types.boolean else lc.Type.int(1)
                ptr = cgutils.alloca_once(builder, typ)
            builder.store(builder.bitcast(ptr, byte_ptr_t), dst)

    # Next, we prepare the individual dimension info recorded in gu_signature
    sig_dim_dict = {}
    occurances = []
    occurances = [sched_sig[0]]
    sig_dim_dict[sched_sig[0]] = context.get_constant(types.intp, 2 * num_dim)
    for var, arg, aty, gu_sig in zip(expr_args[:ninouts], all_args[:ninouts],
                                        outer_sig.args[1:], sin + sout):
        if config.DEBUG_ARRAY_OPT:
            print("var = ", var, " gu_sig = ", gu_sig)
        i = 0
        for dim_sym in gu_sig:
            dim = array_size_vars[var][i]
            if isinstance(dim, ir.Var):
                sig_dim_dict[dim_sym] = lowerer.loadvar(dim.name)
            elif isinstance(dim, int):
                sig_dim_dict[dim_sym] = context.get_constant(types.intp, dim)
            else:
                # raise NotImplementedError("wrong dimension value encoutered: ", dim)
                if config.DEBUG_ARRAY_OPT:
                    print("var = ", var, " type = ", aty)
                ary = context.make_array(aty)(context, builder, arg)
                shapes = cgutils.unpack_tuple(builder, ary.strides, aty.ndim)
                sig_dim_dict[dim_sym] = shapes[i]
            if not (dim_sym in occurances):
                if config.DEBUG_ARRAY_OPT:
                    print("dim_sym = ", dim_sym, ", size = ", array_size_vars[var][i])
                occurances.append(dim_sym)
            i = i + 1

    # Prepare shapes, which is a single number (outer loop size), followed by the size of individual shape variables.
    nshapes = len(sig_dim_dict) + 1
    shapes = cgutils.alloca_once(builder, intp_t, size = nshapes, name = "pshape")
    # For now, outer loop size is the same as number of threads
    builder.store(context.get_constant(types.intp, get_thread_count()), shapes)
    # Individual shape variables go next
    i = 1
    for dim_sym in occurances:
        if config.DEBUG_ARRAY_OPT:
            cgutils.printf(builder, dim_sym + " = %d\n", sig_dim_dict[dim_sym])
        builder.store(sig_dim_dict[dim_sym], builder.gep(shapes,
                                    [ context.get_constant(types.intp, i) ]))
        i = i + 1

    # Prepare steps for each argument. Note that all steps are counted in bytes.
    num_steps = num_args + 1 + len(array_strides)
    steps = cgutils.alloca_once(builder, intp_t,
            size = context.get_constant(types.intp, num_steps), name = "psteps")
    # First goes the step size for sched, which is 2 * num_dim
    builder.store(context.get_constant(types.intp, 2 * num_dim * sizeof_intp),
                                                                        steps)
    # The steps for all others are 0. (TODO: except reduction results)
    for i in range(num_args):
        if i >= ninouts: # steps for reduction vars are abi_sizeof(typ)
            j = i - ninouts
            typ = context.get_value_type(lowerer.fndesc.typemap[redvars[j]])
            sizeof = context.get_abi_sizeof(typ)
            stepsize = context.get_constant(types.intp, sizeof)
        else:
            # steps are strides
            stepsize = zero
        dst = builder.gep(steps, [context.get_constant(types.intp, 1 + i)])
        builder.store(stepsize, dst)
    for j in range(len(array_strides)):
        dst = builder.gep(steps, [context.get_constant(types.intp, 1 + num_args + j)])
        builder.store(array_strides[j], dst)

    # prepare data
    data = builder.inttoptr(zero, byte_ptr_t)

    fnty = lc.Type.function(lc.Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                             intp_ptr_t, byte_ptr_t])
    fn = builder.module.get_or_insert_function(fnty, name=wrapper_name)
    if config.DEBUG_ARRAY_OPT:
        cgutils.printf(builder, "before calling kernel %p\n", fn)
    result = builder.call(fn, [args, shapes, steps, data])
    if config.DEBUG_ARRAY_OPT:
        cgutils.printf(builder, "after calling kernel %p\n", fn)

    scope = init_block.scope
    loc = init_block.loc
    calltypes = lowerer.fndesc.calltypes
    # Accumulate all reduction arrays back to a single value
    for i in range(get_thread_count()):
        for name, arr in zip(redvars, redarrs):
            tmpname = mk_unique_var(name)
            op, imop = reddict[name]
            src = builder.gep(arr, [context.get_constant(types.intp, i)])
            val = builder.load(src)
            vty = lowerer.fndesc.typemap[name]
            lowerer.fndesc.typemap[tmpname] = vty
            lowerer.storevar(val, tmpname)
            accvar = ir.Var(scope, name, loc)
            tmpvar = ir.Var(scope, tmpname, loc)
            acc_call = ir.Expr.inplace_binop(op, imop, accvar, tmpvar, loc)
            calltypes[acc_call] = signature(vty, vty, vty)
            inst = ir.Assign(acc_call, accvar, loc)
            lowerer.lower_inst(inst)

    # TODO: scalar output must be assigned back to corresponding output variables
    return
