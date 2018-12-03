from __future__ import print_function, division, absolute_import

import ast
import copy
from collections import OrderedDict
import linecache
import os
import sys
import numpy as np

import llvmlite.llvmpy.core as lc
import llvmlite.ir.values as liv

import numba
from .. import compiler, ir, types, six, cgutils, sigutils, lowering, parfor
from numba.ir_utils import (add_offset_to_labels, replace_var_names,
                            remove_dels, legalize_names, mk_unique_var,
                            rename_labels, get_name_var_table, visit_vars_inner,
                            get_definition, guard, find_callname,
                            get_call_table, is_pure, get_np_ufunc_typ,
                            get_unused_var_name, find_potential_aliases)
from numba.analysis import (compute_use_defs, compute_live_map,
                            compute_dead_maps, compute_cfg_from_blocks)
from ..typing import signature
from numba import config
from numba.targets.cpu import ParallelOptions
from numba.six import exec_
from numba.parfor import print_wrapped

import warnings
from ..errors import ParallelSafetyWarning


def _lower_parfor_parallel(lowerer, parfor):
    from .parallel import get_thread_count

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
    # We copy the typemap here because for race condition variable we'll
    # update their type to array so they can be updated by the gufunc.
    orig_typemap = lowerer.fndesc.typemap
    # replace original typemap with copy and restore the original at the end.
    lowerer.fndesc.typemap = copy.copy(orig_typemap)
    typemap = lowerer.fndesc.typemap
    varmap = lowerer.varmap

    if config.DEBUG_ARRAY_OPT:
        print("_lower_parfor_parallel")
        parfor.dump()

    loc = parfor.init_block.loc
    scope = parfor.init_block.scope

    # produce instructions for init_block
    if config.DEBUG_ARRAY_OPT:
        print("init_block = ", parfor.init_block, " ", type(parfor.init_block))
    for instr in parfor.init_block.body:
        if config.DEBUG_ARRAY_OPT:
            print("lower init_block instr = ", instr)
        lowerer.lower_inst(instr)

    for racevar in parfor.races:
        if racevar not in varmap:
            rvtyp = typemap[racevar]
            rv = ir.Var(scope, racevar, loc)
            lowerer._alloca_var(rv.name, rvtyp)

    alias_map = {}
    arg_aliases = {}
    numba.parfor.find_potential_aliases_parfor(parfor, parfor.params, typemap,
                                        lowerer.func_ir, alias_map, arg_aliases)
    if config.DEBUG_ARRAY_OPT:
        print("alias_map", alias_map)
        print("arg_aliases", arg_aliases)

    # run get_parfor_outputs() and get_parfor_reductions() before gufunc creation
    # since Jumps are modified so CFG of loop_body dict will become invalid
    assert parfor.params != None

    parfor_output_arrays = numba.parfor.get_parfor_outputs(
        parfor, parfor.params)
    parfor_redvars, parfor_reddict = numba.parfor.get_parfor_reductions(
        parfor, parfor.params, lowerer.fndesc.calltypes)

    # init reduction array allocation here.
    nredvars = len(parfor_redvars)
    redarrs = {}
    if nredvars > 0:
        # reduction arrays outer dimension equal to thread count
        thread_count = get_thread_count()
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc

        # For each reduction variable...
        for i in range(nredvars):
            redvar_typ = lowerer.fndesc.typemap[parfor_redvars[i]]
            redvar = ir.Var(scope, parfor_redvars[i], loc)
            redarrvar_typ = redtyp_to_redarraytype(redvar_typ)
            reddtype = redarrvar_typ.dtype
            if config.DEBUG_ARRAY_OPT:
                print("redvar_typ", redvar_typ, redarrvar_typ, reddtype, types.DType(reddtype))

            # If this is reduction over an array,
            # the reduction array has just one added per-worker dimension.
            if isinstance(redvar_typ, types.npytypes.Array):
                redarrdim = redvar_typ.ndim + 1
            else:
                redarrdim = 1

            # Reduction array is created and initialized to the initial reduction value.

            # First create a var for the numpy empty ufunc.
            empty_func = ir.Var(scope, mk_unique_var("empty_func"), loc)
            ff_fnty = get_np_ufunc_typ(np.empty)
            ff_sig = typingctx.resolve_function_type(ff_fnty,
                                            (types.UniTuple(types.intp, redarrdim),
                                             types.DType(reddtype)), {})
            typemap[empty_func.name] = ff_fnty
            empty_assign = ir.Assign(ir.Global("empty", np.empty, loc=loc), empty_func, loc)
            lowerer.lower_inst(empty_assign)

            # Create var for outer dimension size of reduction array equal to number of threads.
            num_threads_var = ir.Var(scope, mk_unique_var("num_threads"), loc)
            num_threads_assign = ir.Assign(ir.Const(thread_count, loc), num_threads_var, loc)
            typemap[num_threads_var.name] = types.intp
            lowerer.lower_inst(num_threads_assign)

            # Empty call takes tuple of sizes.  Create here and fill in outer dimension (num threads).
            size_var = ir.Var(scope, mk_unique_var("tuple_size_var"), loc)
            typemap[size_var.name] = types.UniTuple(types.intp, redarrdim)
            size_var_list = [num_threads_var]

            # If this is a reduction over an array...
            if isinstance(redvar_typ, types.npytypes.Array):
                # Add code to get the shape of the array being reduced over.
                redshape_var = ir.Var(scope, mk_unique_var("redarr_shape"), loc)
                typemap[redshape_var.name] = types.UniTuple(types.intp, redvar_typ.ndim)
                redshape_getattr = ir.Expr.getattr(redvar, "shape", loc)
                redshape_assign = ir.Assign(redshape_getattr, redshape_var, loc)
                lowerer.lower_inst(redshape_assign)

                # Add the dimension sizes of the array being reduced over to the tuple of sizes pass to empty.
                for j in range(redvar_typ.ndim):
                    onedimvar = ir.Var(scope, mk_unique_var("redshapeonedim"), loc)
                    onedimgetitem = ir.Expr.static_getitem(redshape_var, j, None, loc)
                    typemap[onedimvar.name] = types.intp
                    onedimassign = ir.Assign(onedimgetitem, onedimvar, loc)
                    lowerer.lower_inst(onedimassign)
                    size_var_list += [onedimvar]

            size_call = ir.Expr.build_tuple(size_var_list, loc)
            size_assign = ir.Assign(size_call, size_var, loc)
            lowerer.lower_inst(size_assign)

            # Add call to empty passing the size var tuple.
            empty_call = ir.Expr.call(empty_func, [size_var], {}, loc=loc)
            redarr_var = ir.Var(scope, mk_unique_var("redarr"), loc)
            typemap[redarr_var.name] = redarrvar_typ
            empty_call_assign = ir.Assign(empty_call, redarr_var, loc)
            lowerer.fndesc.calltypes[empty_call] = ff_sig
            lowerer.lower_inst(empty_call_assign)

            # Remember mapping of original reduction array to the newly created per-worker reduction array.
            redarrs[redvar.name] = redarr_var

            init_val = parfor_reddict[parfor_redvars[i]][0]
            if init_val != None:
                if isinstance(redvar_typ, types.npytypes.Array):
                    # Create an array of identity values for the reduction.
                    # First, create a variable for np.full.
                    full_func = ir.Var(scope, mk_unique_var("full_func"), loc)
                    full_fnty = get_np_ufunc_typ(np.full)
                    full_sig = typingctx.resolve_function_type(full_fnty,
                               (types.UniTuple(types.intp, redvar_typ.ndim),
                                reddtype,
                                types.DType(reddtype)), {})
                    typemap[full_func.name] = full_fnty
                    full_assign = ir.Assign(ir.Global("full", np.full, loc=loc), full_func, loc)
                    lowerer.lower_inst(full_assign)

                    # Then create a var with the identify value.
                    init_val_var = ir.Var(scope, mk_unique_var("init_val"), loc)
                    init_val_assign = ir.Assign(ir.Const(init_val, loc), init_val_var, loc)
                    typemap[init_val_var.name] = reddtype
                    lowerer.lower_inst(init_val_assign)

                    # Then, call np.full with the shape of the reduction array and the identity value.
                    full_call = ir.Expr.call(full_func, [redshape_var, init_val_var], {}, loc=loc)
                    lowerer.fndesc.calltypes[full_call] = full_sig
                    redtoset = ir.Var(scope, mk_unique_var("redtoset"), loc)
                    redtoset_assign = ir.Assign(full_call, redtoset, loc)
                    typemap[redtoset.name] = redvar_typ
                    lowerer.lower_inst(redtoset_assign)
                else:
                    redtoset = ir.Var(scope, mk_unique_var("redtoset"), loc)
                    redtoset_assign = ir.Assign(ir.Const(init_val, loc), redtoset, loc)
                    typemap[redtoset.name] = reddtype
                    lowerer.lower_inst(redtoset_assign)
            else:
                redtoset = redvar

            # For each thread, initialize the per-worker reduction array to the current reduction array value.
            for j in range(get_thread_count()):
                index_var = ir.Var(scope, mk_unique_var("index_var"), loc)
                index_var_assign = ir.Assign(ir.Const(j, loc), index_var, loc)
                typemap[index_var.name] = types.uintp
                lowerer.lower_inst(index_var_assign)

                redsetitem = ir.SetItem(redarr_var, index_var, redtoset, loc)
                lowerer.fndesc.calltypes[redsetitem] = signature(types.none,
                        typemap[redarr_var.name], typemap[index_var.name], redvar_typ)
                lowerer.lower_inst(redsetitem)

    # compile parfor body as a separate function to be used with GUFuncWrapper
    flags = copy.copy(parfor.flags)
    flags.set('error_model', 'numpy')
    # Can't get here unless  flags.set('auto_parallel', ParallelOptions(True))
    index_var_typ = typemap[parfor.loop_nests[0].index_variable.name]
    # index variables should have the same type, check rest of indices
    for l in parfor.loop_nests[1:]:
        assert typemap[l.index_variable.name] == index_var_typ
    numba.parfor.sequential_parfor_lowering = True
    func, func_args, func_sig, redargstartdim, func_arg_types = _create_gufunc_for_parfor_body(
        lowerer, parfor, typemap, typingctx, targetctx, flags, {},
        bool(alias_map), index_var_typ, parfor.races)
    numba.parfor.sequential_parfor_lowering = False

    # get the shape signature
    func_args = ['sched'] + func_args
    num_reductions = len(parfor_redvars)
    num_inputs = len(func_args) - len(parfor_output_arrays) - num_reductions
    if config.DEBUG_ARRAY_OPT:
        print("func_args = ", func_args)
        print("num_inputs = ", num_inputs)
        print("parfor_outputs = ", parfor_output_arrays)
        print("parfor_redvars = ", parfor_redvars)
        print("num_reductions = ", num_reductions)
    gu_signature = _create_shape_signature(
        parfor.get_shape_classes,
        num_inputs,
        num_reductions,
        func_args,
        redargstartdim,
        func_sig,
        parfor.races,
        typemap)
    if config.DEBUG_ARRAY_OPT:
        print("gu_signature = ", gu_signature)

    # call the func in parallel by wrapping it with ParallelGUFuncBuilder
    loop_ranges = [(l.start, l.stop, l.step) for l in parfor.loop_nests]
    if config.DEBUG_ARRAY_OPT:
        print("loop_nests = ", parfor.loop_nests)
        print("loop_ranges = ", loop_ranges)
    call_parallel_gufunc(
        lowerer,
        func,
        gu_signature,
        func_sig,
        func_args,
        func_arg_types,
        loop_ranges,
        parfor_redvars,
        parfor_reddict,
        redarrs,
        parfor.init_block,
        index_var_typ,
        parfor.races)
    if config.DEBUG_ARRAY_OPT:
        sys.stdout.flush()

    if nredvars > 0:
        # Perform the final reduction across the reduction array created above.
        thread_count = get_thread_count()
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc

        # For each reduction variable...
        for i in range(nredvars):
            name = parfor_redvars[i]
            redarr = redarrs[name]
            redvar_typ = lowerer.fndesc.typemap[name]

            if config.DEBUG_ARRAY_OPT_RUNTIME:
                res_print_str = "res_print"
                strconsttyp = types.StringLiteral(res_print_str)
                lhs = ir.Var(scope, mk_unique_var("str_const"), loc)
                assign_lhs = ir.Assign(value=ir.Const(value=res_print_str, loc=loc),
                                               target=lhs, loc=loc)
                typemap[lhs.name] = strconsttyp
                lowerer.lower_inst(assign_lhs)

                res_print = ir.Print(args=[lhs, redarr], vararg=None, loc=loc)
                lowerer.fndesc.calltypes[res_print] = signature(types.none,
                                                         typemap[lhs.name],
                                                         typemap[redarr.name])
                print("res_print", res_print)
                lowerer.lower_inst(res_print)

            # For each element in the reduction array created above.
            for j in range(get_thread_count()):
                # Create index var to access that element.
                index_var = ir.Var(scope, mk_unique_var("index_var"), loc)
                index_var_assign = ir.Assign(ir.Const(j, loc), index_var, loc)
                typemap[index_var.name] = types.uintp
                lowerer.lower_inst(index_var_assign)

                # Read that element from the array into oneelem.
                oneelem = ir.Var(scope, mk_unique_var("redelem"), loc)
                oneelemgetitem = ir.Expr.getitem(redarr, index_var, loc)
                typemap[oneelem.name] = redvar_typ
                lowerer.fndesc.calltypes[oneelemgetitem] = signature(redvar_typ,
                        typemap[redarr.name], typemap[index_var.name])
                oneelemassign = ir.Assign(oneelemgetitem, oneelem, loc)
                lowerer.lower_inst(oneelemassign)

                init_var = ir.Var(scope, name+"#init", loc)
                init_assign = ir.Assign(oneelem, init_var, loc)
                if name+"#init" not in typemap:
                    typemap[init_var.name] = redvar_typ
                lowerer.lower_inst(init_assign)

                if config.DEBUG_ARRAY_OPT_RUNTIME:
                    res_print_str = "one_res_print"
                    strconsttyp = types.StringLiteral(res_print_str)
                    lhs = ir.Var(scope, mk_unique_var("str_const"), loc)
                    assign_lhs = ir.Assign(value=ir.Const(value=res_print_str, loc=loc),
                                               target=lhs, loc=loc)
                    typemap[lhs.name] = strconsttyp
                    lowerer.lower_inst(assign_lhs)

                    res_print = ir.Print(args=[lhs, index_var, oneelem, init_var],
                                         vararg=None, loc=loc)
                    lowerer.fndesc.calltypes[res_print] = signature(types.none,
                                                             typemap[lhs.name],
                                                             typemap[index_var.name],
                                                             typemap[oneelem.name],
                                                             typemap[init_var.name])
                    print("res_print", res_print)
                    lowerer.lower_inst(res_print)

                # generate code for combining reduction variable with thread output
                for inst in parfor_reddict[name][1]:
                    # If we have a case where a parfor body has an array reduction like A += B
                    # and A and B have different data types then the reduction in the parallel
                    # region will operate on those differeing types.  However, here, after the
                    # parallel region, we are summing across the reduction array and that is
                    # guaranteed to have the same data type so we need to change the reduction
                    # nodes so that the right-hand sides have a type equal to the reduction-type
                    # and therefore the left-hand side.
                    if isinstance(inst, ir.Assign):
                        rhs = inst.value
                        # We probably need to generalize this since it only does substitutions in
                        # inplace_ginops.
                        if isinstance(rhs, ir.Expr) and rhs.op == 'inplace_binop' and rhs.rhs.name == init_var.name:
                            # Get calltype of rhs.
                            ct = lowerer.fndesc.calltypes[rhs]
                            assert(len(ct.args) == 2)
                            # Create new arg types replace the second arg type with the reduction var type.
                            ctargs = (ct.args[0], redvar_typ)
                            # Update the signature of the call.
                            ct = ct.replace(args=ctargs)
                            # Remove so we can re-insrt since calltypes is unique dict.
                            lowerer.fndesc.calltypes.pop(rhs)
                            # Add calltype back in for the expr with updated signature.
                            lowerer.fndesc.calltypes[rhs] = ct
                    lowerer.lower_inst(inst)

    # Restore the original typemap of the function that was replaced temporarily at the
    # Beginning of this function.
    lowerer.fndesc.typemap = orig_typemap

# A work-around to prevent circular imports
lowering.lower_extensions[parfor.Parfor] = _lower_parfor_parallel


def _create_shape_signature(
        get_shape_classes,
        num_inputs,
        num_reductions,
        args,
        redargstartdim,
        func_sig,
        races,
        typemap):
    '''Create shape signature for GUFunc
    '''
    if config.DEBUG_ARRAY_OPT:
        print("_create_shape_signature", num_inputs, num_reductions, args, redargstartdim)
        for i in args[1:]:
            print("argument", i, type(i), get_shape_classes(i, typemap=typemap))

    num_inouts = len(args) - num_reductions
    # maximum class number for array shapes
    classes = [get_shape_classes(var, typemap=typemap) if var not in races else (-1,) for var in args[1:]]
    class_set = set()
    for _class in classes:
        if _class:
            for i in _class:
                class_set.add(i)
    max_class = max(class_set) + 1 if class_set else 0
    classes.insert(0, (max_class,)) # force set the class of 'sched' argument
    class_set.add(max_class)
    class_map = {}
    # TODO: use prefix + class number instead of single char
    alphabet = ord('a')
    for n in class_set:
       if n >= 0:
           class_map[n] = chr(alphabet)
           alphabet += 1

    alpha_dict = {'latest_alpha' : alphabet}

    def bump_alpha(c, class_map):
        if c >= 0:
            return class_map[c]
        else:
            alpha_dict['latest_alpha'] += 1
            return chr(alpha_dict['latest_alpha'])

    gu_sin = []
    gu_sout = []
    count = 0
    syms_sin = ()
    if config.DEBUG_ARRAY_OPT:
        print("args", args)
        print("classes", classes)
    for cls, arg in zip(classes, args):
        count = count + 1
        if cls:
            dim_syms = tuple(bump_alpha(c, class_map) for c in cls)
        else:
            dim_syms = ()
        if (count > num_inouts):
            # Strip the first symbol corresponding to the number of workers
            # so that guvectorize will parallelize across the reduction.
            gu_sin.append(dim_syms[redargstartdim[arg]:])
        else:
            gu_sin.append(dim_syms)
            syms_sin += dim_syms
    return (gu_sin, gu_sout)

def _print_block(block):
    for i, inst in enumerate(block.body):
        print("    ", i, " ", inst)

def _print_body(body_dict):
    '''Pretty-print a set of IR blocks.
    '''
    for label, block in body_dict.items():
        print("label: ", label)
        _print_block(block)


def wrap_loop_body(loop_body):
    blocks = loop_body.copy()  # shallow copy is enough
    first_label = min(blocks.keys())
    last_label = max(blocks.keys())
    loc = blocks[last_label].loc
    blocks[last_label].body.append(ir.Jump(first_label, loc))
    return blocks

def unwrap_loop_body(loop_body):
    last_label = max(loop_body.keys())
    loop_body[last_label].body = loop_body[last_label].body[:-1]

def compute_def_once_block(block, def_once, def_more, getattr_taken):
    assignments = block.find_insts(ir.Assign)
    for one_assign in assignments:
        a_def = one_assign.target.name
        if a_def in def_more:
            pass
        elif a_def in def_once:
            def_more.add(a_def)
            def_once.remove(a_def)
        else:
            def_once.add(a_def)

        rhs = one_assign.value
        if isinstance(rhs, ir.Expr) and rhs.op == 'getattr' and rhs.value.name in def_once:
            getattr_taken[a_def] = rhs.value.name
        if isinstance(rhs, ir.Expr) and rhs.op == 'call' and rhs.func.name in getattr_taken:
            base_obj = getattr_taken[rhs.func.name]
            def_more.add(base_obj)
            def_once.remove(base_obj)

def compute_def_once_internal(loop_body, def_once, def_more, getattr_taken):
    for label, block in loop_body.items():
        compute_def_once_block(block, def_once, def_more, getattr_taken)
        for inst in block.body:
            if isinstance(inst, parfor.Parfor):
                compute_def_once_block(inst.init_block, def_once, def_more, getattr_taken)
                compute_def_once_internal(inst.loop_body, def_once, def_more, getattr_taken)

def compute_def_once(loop_body):
    def_once = set()
    def_more = set()
    getattr_taken = {}
    compute_def_once_internal(loop_body, def_once, def_more, getattr_taken)
    return def_once

def find_vars(var, varset):
    assert isinstance(var, ir.Var)
    varset.add(var.name)
    return var

def _hoist_internal(inst, dep_on_param, call_table, hoisted, not_hoisted,
                    typemap):
    uses = set()
    visit_vars_inner(inst.value, find_vars, uses)
    diff = uses.difference(dep_on_param)
    if len(diff) == 0 and is_pure(inst.value, None, call_table):
        if config.DEBUG_ARRAY_OPT == 1:
            print("Will hoist instruction", inst)
        hoisted.append(inst)
        if not isinstance(typemap[inst.target.name], types.npytypes.Array):
            dep_on_param += [inst.target.name]
        return True
    else:
        if len(diff) > 0:
            not_hoisted.append((inst, "dependency"))
            if config.DEBUG_ARRAY_OPT == 1:
                print("Instruction", inst, " could not be hoisted because of a dependency.")
        else:
            not_hoisted.append((inst, "not pure"))
            if config.DEBUG_ARRAY_OPT == 1:
                print("Instruction", inst, " could not be hoisted because it isn't pure.")
    return False

def find_setitems_block(setitems, block):
    for inst in block.body:
        if isinstance(inst, ir.StaticSetItem) or isinstance(inst, ir.SetItem):
            setitems.add(inst.target.name)
        elif isinstance(inst, parfor.Parfor):
            find_setitems_block(setitems, inst.init_block)
            find_setitems_body(setitems, inst.loop_body)

def find_setitems_body(setitems, loop_body):
    for label, block in loop_body.items():
        find_setitems_block(setitems, block)

def hoist(parfor_params, loop_body, typemap, wrapped_blocks):
    dep_on_param = copy.copy(parfor_params)
    hoisted = []
    not_hoisted = []

    def_once = compute_def_once(loop_body)
    (call_table, reverse_call_table) = get_call_table(wrapped_blocks)

    setitems = set()
    find_setitems_body(setitems, loop_body)
    dep_on_param = list(set(dep_on_param).difference(setitems))

    for label, block in loop_body.items():
        new_block = []
        for inst in block.body:
            if isinstance(inst, ir.Assign) and inst.target.name in def_once:
                if _hoist_internal(inst, dep_on_param, call_table,
                                   hoisted, not_hoisted, typemap):
                    # don't add this instruction to the block since it is
                    # hoisted
                    continue
            elif isinstance(inst, parfor.Parfor):
                new_init_block = []
                if config.DEBUG_ARRAY_OPT == 1:
                    print("parfor")
                    inst.dump()
                for ib_inst in inst.init_block.body:
                    if (isinstance(ib_inst, ir.Assign) and
                        ib_inst.target.name in def_once):
                        if _hoist_internal(ib_inst, dep_on_param, call_table,
                                           hoisted, not_hoisted, typemap):
                            # don't add this instuction to the block since it is hoisted
                            continue
                    new_init_block.append(ib_inst)
                inst.init_block.body = new_init_block

            new_block.append(inst)
        block.body = new_block
    return hoisted, not_hoisted

def redtyp_is_scalar(redtype):
    return not isinstance(redtype, types.npytypes.Array)

def redtyp_to_redarraytype(redtyp):
    """Go from a reducation variable type to a reduction array type used to hold
       per-worker results.
    """
    redarrdim = 1
    # If the reduction type is an array then allocate reduction array with ndim+1 dimensions.
    if isinstance(redtyp, types.npytypes.Array):
        redarrdim += redtyp.ndim
        # We don't create array of array but multi-dimensional reduciton array with same dtype.
        redtyp = redtyp.dtype
    return types.npytypes.Array(redtyp, redarrdim, "C")

def redarraytype_to_sig(redarraytyp):
    """Given a reduction array type, find the type of the reduction argument to the gufunc.
       Scalar and 1D array reduction both end up with 1D gufunc param type since scalars have to
       be passed as arrays.
    """
    assert isinstance(redarraytyp, types.npytypes.Array)
    return types.npytypes.Array(redarraytyp.dtype, max(1, redarraytyp.ndim - 1), redarraytyp.layout)

def _create_gufunc_for_parfor_body(
        lowerer,
        parfor,
        typemap,
        typingctx,
        targetctx,
        flags,
        locals,
        has_aliases,
        index_var_typ,
        races):
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

    loc = parfor.init_block.loc

    # The parfor body and the main function body share ir.Var nodes.
    # We have to do some replacements of Var names in the parfor body to make them
    # legal parameter names.  If we don't copy then the Vars in the main function also
    # would incorrectly change their name.
    loop_body = copy.copy(parfor.loop_body)
    remove_dels(loop_body)

    parfor_dim = len(parfor.loop_nests)
    loop_indices = [l.index_variable.name for l in parfor.loop_nests]

    # Get all the parfor params.
    parfor_params = parfor.params
    # Get just the outputs of the parfor.
    parfor_outputs = numba.parfor.get_parfor_outputs(parfor, parfor_params)
    # Get all parfor reduction vars, and operators.
    typemap = lowerer.fndesc.typemap
    parfor_redvars, parfor_reddict = numba.parfor.get_parfor_reductions(
        parfor, parfor_params, lowerer.fndesc.calltypes)
    # Compute just the parfor inputs as a set difference.
    parfor_inputs = sorted(
        list(
            set(parfor_params) -
            set(parfor_outputs) -
            set(parfor_redvars)))

    races = races.difference(set(parfor_redvars))
    for race in races:
        warnings.warn_explicit("Variable %s used in parallel loop may be written "
                      "to simultaneously by multiple workers and may result "
                      "in non-deterministic or unintended results." % race,
                      ParallelSafetyWarning,
                      loc.filename,
                      loc.line)
    replace_var_with_array(races, loop_body, typemap, lowerer.fndesc.calltypes)

    if config.DEBUG_ARRAY_OPT == 1:
        print("parfor_params = ", parfor_params, " ", type(parfor_params))
        print("parfor_outputs = ", parfor_outputs, " ", type(parfor_outputs))
        print("parfor_inputs = ", parfor_inputs, " ", type(parfor_inputs))
        print("parfor_redvars = ", parfor_redvars, " ", type(parfor_redvars))

    # Reduction variables are represented as arrays, so they go under
    # different names.
    parfor_redarrs = []
    parfor_red_arg_types = []
    for var in parfor_redvars:
        arr = var + "_arr"
        parfor_redarrs.append(arr)
        redarraytype = redtyp_to_redarraytype(typemap[var])
        parfor_red_arg_types.append(redarraytype)
        redarrsig = redarraytype_to_sig(redarraytype)
        if arr in typemap:
            assert(typemap[arr] == redarrsig)
        else:
            typemap[arr] = redarrsig

    # Reorder all the params so that inputs go first then outputs.
    parfor_params = parfor_inputs + parfor_outputs + parfor_redarrs

    if config.DEBUG_ARRAY_OPT == 1:
        print("parfor_params = ", parfor_params, " ", type(parfor_params))
        print("loop_indices = ", loop_indices, " ", type(loop_indices))
        print("loop_body = ", loop_body, " ", type(loop_body))
        _print_body(loop_body)

    # Some Var are not legal parameter names so create a dict of potentially illegal
    # param name to guaranteed legal name.
    param_dict = legalize_names(parfor_params + parfor_redvars)
    if config.DEBUG_ARRAY_OPT == 1:
        print(
            "param_dict = ",
            sorted(
                param_dict.items()),
            " ",
            type(param_dict))

    # Some loop_indices are not legal parameter names so create a dict of potentially illegal
    # loop index to guaranteed legal name.
    ind_dict = legalize_names(loop_indices)
    # Compute a new list of legal loop index names.
    legal_loop_indices = [ind_dict[v] for v in loop_indices]
    if config.DEBUG_ARRAY_OPT == 1:
        print("ind_dict = ", sorted(ind_dict.items()), " ", type(ind_dict))
        print(
            "legal_loop_indices = ",
            legal_loop_indices,
            " ",
            type(legal_loop_indices))
        for pd in parfor_params:
            print("pd = ", pd)
            print("pd type = ", typemap[pd], " ", type(typemap[pd]))

    # Get the types of each parameter.
    param_types = [typemap[v] for v in parfor_params]
    # Calculate types of args passed to gufunc.
    func_arg_types = [typemap[v] for v in (parfor_inputs + parfor_outputs)] + parfor_red_arg_types

    # Replace illegal parameter names in the loop body with legal ones.
    replace_var_names(loop_body, param_dict)
    # remember the name before legalizing as the actual arguments
    parfor_args = parfor_params
    # Change parfor_params to be legal names.
    parfor_params = [param_dict[v] for v in parfor_params]
    parfor_params_orig = parfor_params

    parfor_params = []
    ascontig = False
    for pindex in range(len(parfor_params_orig)):
        if (ascontig and
            pindex < len(parfor_inputs) and
            isinstance(param_types[pindex], types.npytypes.Array)):
            parfor_params.append(parfor_params_orig[pindex]+"param")
        else:
            parfor_params.append(parfor_params_orig[pindex])

    # Change parfor body to replace illegal loop index vars with legal ones.
    replace_var_names(loop_body, ind_dict)
    loop_body_var_table = get_name_var_table(loop_body)
    sentinel_name = get_unused_var_name("__sentinel__", loop_body_var_table)

    if config.DEBUG_ARRAY_OPT == 1:
        print(
            "legal parfor_params = ",
            parfor_params,
            " ",
            type(parfor_params))

    # Determine the unique names of the scheduling and gufunc functions.
    # sched_func_name = "__numba_parfor_sched_%s" % (hex(hash(parfor)).replace("-", "_"))
    gufunc_name = "__numba_parfor_gufunc_%s" % (
        hex(hash(parfor)).replace("-", "_"))
    if config.DEBUG_ARRAY_OPT:
        # print("sched_func_name ", type(sched_func_name), " ", sched_func_name)
        print("gufunc_name ", type(gufunc_name), " ", gufunc_name)

    gufunc_txt = ""

    # Create the gufunc function.
    gufunc_txt += "def " + gufunc_name + \
        "(sched, " + (", ".join(parfor_params)) + "):\n"

    for pindex in range(len(parfor_inputs)):
        if ascontig and isinstance(param_types[pindex], types.npytypes.Array):
            gufunc_txt += ("    " + parfor_params_orig[pindex]
                + " = np.ascontiguousarray(" + parfor_params[pindex] + ")\n")

    # Add initialization of reduction variables
    for arr, var in zip(parfor_redarrs, parfor_redvars):
        # If reduction variable is a scalar then save current value to
        # temp and accumulate on that temp to prevent false sharing.
        if redtyp_is_scalar(typemap[var]):
            gufunc_txt += "    " + param_dict[var] + \
                 "=" + param_dict[arr] + "[0]\n"
        else:
            # The reduction variable is an array so np.copy it to a temp.
            gufunc_txt += "    " + param_dict[var] + \
                 "=np.copy(" + param_dict[arr] + ")\n"

    # For each dimension of the parfor, create a for loop in the generated gufunc function.
    # Iterate across the proper values extracted from the schedule.
    # The form of the schedule is start_dim0, start_dim1, ..., start_dimN, end_dim0,
    # end_dim1, ..., end_dimN
    for eachdim in range(parfor_dim):
        for indent in range(eachdim + 1):
            gufunc_txt += "    "
        sched_dim = eachdim
        gufunc_txt += ("for " +
                       legal_loop_indices[eachdim] +
                       " in range(sched[" +
                       str(sched_dim) +
                       "], sched[" +
                       str(sched_dim +
                           parfor_dim) +
                       "] + np.uint8(1)):\n")

    if config.DEBUG_ARRAY_OPT_RUNTIME:
        for indent in range(parfor_dim + 1):
            gufunc_txt += "    "
        gufunc_txt += "print("
        for eachdim in range(parfor_dim):
            gufunc_txt += "\"" + legal_loop_indices[eachdim] + "\"," + legal_loop_indices[eachdim] + ","
        gufunc_txt += ")\n"

    # Add the sentinel assignment so that we can find the loop body position
    # in the IR.
    for indent in range(parfor_dim + 1):
        gufunc_txt += "    "
    gufunc_txt += sentinel_name + " = 0\n"
    # Add assignments of reduction variables (for returning the value)
    redargstartdim = {}
    for arr, var in zip(parfor_redarrs, parfor_redvars):
        # After the gufunc loops, copy the accumulated temp value back to reduction array.
        if redtyp_is_scalar(typemap[var]):
            gufunc_txt += "    " + param_dict[arr] + \
                "[0] = " + param_dict[var] + "\n"
            redargstartdim[arr] = 1
        else:
            # After the gufunc loops, copy the accumulated temp array back to reduction array with ":"
            gufunc_txt += "    " + param_dict[arr] + \
                "[:] = " + param_dict[var] + "[:]\n"
            redargstartdim[arr] = 0
    gufunc_txt += "    return None\n"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
    # Force gufunc outline into existence.
    globls = {"np": np}
    locls = {}
    exec_(gufunc_txt, globls, locls)
    gufunc_func = locls[gufunc_name]

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
    reserved_names = [sentinel_name] + \
        list(param_dict.values()) + legal_loop_indices
    for name, var in var_table.items():
        if not (name in reserved_names):
            new_var_dict[name] = mk_unique_var(name)
    replace_var_names(gufunc_ir.blocks, new_var_dict)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir dump after renaming ")
        gufunc_ir.dump()

    gufunc_param_types = [
        numba.types.npytypes.Array(
            index_var_typ, 1, "C")] + param_types
    if config.DEBUG_ARRAY_OPT:
        print(
            "gufunc_param_types = ",
            type(gufunc_param_types),
            "\n",
            gufunc_param_types)

    gufunc_stub_last_label = max(gufunc_ir.blocks.keys()) + 1

    # Add gufunc stub last label to each parfor.loop_body label to prevent
    # label conflicts.
    loop_body = add_offset_to_labels(loop_body, gufunc_stub_last_label)
    # new label for splitting sentinel block
    new_label = max(loop_body.keys()) + 1

    # If enabled, add a print statement after every assignment.
    if config.DEBUG_ARRAY_OPT_RUNTIME:
        for label, block in loop_body.items():
            new_block = block.copy()
            new_block.clear()
            loc = block.loc
            scope = block.scope
            for inst in block.body:
                new_block.append(inst)
                # Append print after assignment
                if isinstance(inst, ir.Assign):
                    # Only apply to numbers
                    if typemap[inst.target.name] not in types.number_domain:
                        continue

                    # Make constant string
                    strval = "{} =".format(inst.target.name)
                    strconsttyp = types.StringLiteral(strval)

                    lhs = ir.Var(scope, mk_unique_var("str_const"), loc)
                    assign_lhs = ir.Assign(value=ir.Const(value=strval, loc=loc),
                                           target=lhs, loc=loc)
                    typemap[lhs.name] = strconsttyp
                    new_block.append(assign_lhs)

                    # Make print node
                    print_node = ir.Print(args=[lhs, inst.target], vararg=None, loc=loc)
                    new_block.append(print_node)
                    sig = numba.typing.signature(types.none,
                                           typemap[lhs.name],
                                           typemap[inst.target.name])
                    lowerer.fndesc.calltypes[print_node] = sig
            loop_body[label] = new_block

    if config.DEBUG_ARRAY_OPT:
        print("parfor loop body")
        _print_body(loop_body)

    wrapped_blocks = wrap_loop_body(loop_body)
    hoisted, not_hoisted = hoist(parfor_params, loop_body, typemap, wrapped_blocks)
    start_block = gufunc_ir.blocks[min(gufunc_ir.blocks.keys())]
    start_block.body = start_block.body[:-1] + hoisted + [start_block.body[-1]]
    unwrap_loop_body(loop_body)

    # store hoisted into diagnostics
    diagnostics = lowerer.metadata['parfor_diagnostics']
    diagnostics.hoist_info[parfor.id] = {'hoisted': hoisted,
                                         'not_hoisted': not_hoisted}

    if config.DEBUG_ARRAY_OPT:
        print("After hoisting")
        _print_body(loop_body)

    # Search all the block in the gufunc outline for the sentinel assignment.
    for label, block in gufunc_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(
                    inst,
                    ir.Assign) and inst.target.name == sentinel_name:
                # We found the sentinel assignment.
                loc = inst.loc
                scope = block.scope
                # split block across __sentinel__
                # A new block is allocated for the statements prior to the sentinel
                # but the new block maintains the current block label.
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]
                # The current block is used for statements after the sentinel.
                block.body = block.body[i + 1:]
                # But the current block gets a new label.
                body_first_label = min(loop_body.keys())

                # The previous block jumps to the minimum labelled block of the
                # parfor body.
                prev_block.append(ir.Jump(body_first_label, loc))
                # Add all the parfor loop body blocks to the gufunc function's
                # IR.
                for (l, b) in loop_body.items():
                    gufunc_ir.blocks[l] = b
                body_last_label = max(loop_body.keys())
                gufunc_ir.blocks[new_label] = block
                gufunc_ir.blocks[label] = prev_block
                # Add a jump from the last parfor body block to the block containing
                # statements after the sentinel.
                gufunc_ir.blocks[body_last_label].append(
                    ir.Jump(new_label, loc))
                break
        else:
            continue
        break

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir last dump before renaming")
        gufunc_ir.dump()

    gufunc_ir.blocks = rename_labels(gufunc_ir.blocks)
    remove_dels(gufunc_ir.blocks)

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir last dump")
        gufunc_ir.dump()
        print("flags", flags)
        print("typemap", typemap)

    old_alias = flags.noalias
    if not has_aliases:
        if config.DEBUG_ARRAY_OPT:
            print("No aliases found so adding noalias flag.")
        flags.noalias = True
    kernel_func = compiler.compile_ir(
        typingctx,
        targetctx,
        gufunc_ir,
        gufunc_param_types,
        types.none,
        flags,
        locals)

    flags.noalias = old_alias

    kernel_sig = signature(types.none, *gufunc_param_types)
    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return kernel_func, parfor_args, kernel_sig, redargstartdim, func_arg_types

def replace_var_with_array_in_block(vars, block, typemap, calltypes):
    new_block = []
    for inst in block.body:
        if isinstance(inst, ir.Assign) and inst.target.name in vars:
            const_node = ir.Const(0, inst.loc)
            const_var = ir.Var(inst.target.scope, mk_unique_var("$const_ind_0"), inst.loc)
            typemap[const_var.name] = types.uintp
            const_assign = ir.Assign(const_node, const_var, inst.loc)
            new_block.append(const_assign)

            setitem_node = ir.SetItem(inst.target, const_var, inst.value, inst.loc)
            calltypes[setitem_node] = signature(
                types.none, types.npytypes.Array(typemap[inst.target.name], 1, "C"), types.intp, typemap[inst.target.name])
            new_block.append(setitem_node)
            continue
        elif isinstance(inst, parfor.Parfor):
            replace_var_with_array_internal(vars, {0: inst.init_block}, typemap, calltypes)
            replace_var_with_array_internal(vars, inst.loop_body, typemap, calltypes)

        new_block.append(inst)
    return new_block

def replace_var_with_array_internal(vars, loop_body, typemap, calltypes):
    for label, block in loop_body.items():
        block.body = replace_var_with_array_in_block(vars, block, typemap, calltypes)

def replace_var_with_array(vars, loop_body, typemap, calltypes):
    replace_var_with_array_internal(vars, loop_body, typemap, calltypes)
    for v in vars:
        el_typ = typemap[v]
        typemap.pop(v, None)
        typemap[v] = types.npytypes.Array(el_typ, 1, "C")

def call_parallel_gufunc(lowerer, cres, gu_signature, outer_sig, expr_args, expr_arg_types,
                         loop_ranges, redvars, reddict, redarrdict, init_block, index_var_typ, races):
    '''
    Adds the call to the gufunc function from the main function.
    '''
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library

    from .parallel import (ParallelGUFuncBuilder, build_gufunc_wrapper,
                           get_thread_count, _launch_threads)

    if config.DEBUG_ARRAY_OPT:
        print("make_parallel_loop")
        print("args = ", expr_args)
        print("outer_sig = ", outer_sig.args, outer_sig.return_type,
              outer_sig.recvr, outer_sig.pysig)
        print("loop_ranges = ", loop_ranges)
        print("expr_args", expr_args)
        print("expr_arg_types", expr_arg_types)
        print("gu_signature", gu_signature)

    # Build the wrapper for GUFunc
    args, return_type = sigutils.normalize_signature(outer_sig)
    llvm_func = cres.library.get_function(cres.fndesc.llvm_func_name)
    sin, sout = gu_signature

    # These are necessary for build_gufunc_wrapper to find external symbols
    _launch_threads()

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
            return context.get_constant(types.uintp, v)

    num_dim = len(loop_ranges)
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        start = load_range(start)
        stop = load_range(stop)
        assert(step == 1)  # We do not support loop steps other than 1
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
    uintp_ptr_t = lc.Type.pointer(uintp_t)
    zero = context.get_constant(types.uintp, 0)
    one = context.get_constant(types.uintp, 1)
    one_type = one.type
    sizeof_intp = context.get_abi_sizeof(intp_t)

    # Prepare sched, first pop it out of expr_args, outer_sig, and gu_signature
    sched_name = expr_args.pop(0)
    sched_typ = outer_sig.args[0]
    sched_sig = sin.pop(0)

    if config.DEBUG_ARRAY_OPT:
        print("Parfor has potentially negative start", index_var_typ.signed)

    if index_var_typ.signed:
        sched_type = intp_t
        sched_ptr_type = intp_ptr_t
    else:
        sched_type = uintp_t
        sched_ptr_type = uintp_ptr_t

    # Call do_scheduling with appropriate arguments
    dim_starts = cgutils.alloca_once(
        builder, sched_type, size=context.get_constant(
            types.uintp, num_dim), name="dims")
    dim_stops = cgutils.alloca_once(
        builder, sched_type, size=context.get_constant(
            types.uintp, num_dim), name="dims")
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        if start.type != one_type:
            start = builder.sext(start, one_type)
        if stop.type != one_type:
            stop = builder.sext(stop, one_type)
        if step.type != one_type:
            step = builder.sext(step, one_type)
        # substract 1 because do-scheduling takes inclusive ranges
        stop = builder.sub(stop, one)
        builder.store(
            start, builder.gep(
                dim_starts, [
                    context.get_constant(
                        types.uintp, i)]))
        builder.store(stop, builder.gep(dim_stops,
                                        [context.get_constant(types.uintp, i)]))

    sched_size = get_thread_count() * num_dim * 2
    sched = cgutils.alloca_once(
        builder, sched_type, size=context.get_constant(
            types.uintp, sched_size), name="sched")
    debug_flag = 1 if config.DEBUG_ARRAY_OPT else 0
    scheduling_fnty = lc.Type.function(
        intp_ptr_t, [uintp_t, sched_ptr_type, sched_ptr_type, uintp_t, sched_ptr_type, intp_t])
    if index_var_typ.signed:
        do_scheduling = builder.module.get_or_insert_function(scheduling_fnty,
                                                          name="do_scheduling_signed")
    else:
        do_scheduling = builder.module.get_or_insert_function(scheduling_fnty,
                                                          name="do_scheduling_unsigned")

    builder.call(
        do_scheduling, [
            context.get_constant(
                types.uintp, num_dim), dim_starts, dim_stops, context.get_constant(
                types.uintp, get_thread_count()), sched, context.get_constant(
                    types.intp, debug_flag)])

    # Get the LLVM vars for the Numba IR reduction array vars.
    redarrs = [lowerer.loadvar(redarrdict[x].name) for x in redvars]

    nredvars = len(redvars)
    ninouts = len(expr_args) - nredvars

    if config.DEBUG_ARRAY_OPT:
        for i in range(get_thread_count()):
            cgutils.printf(builder, "sched[" + str(i) + "] = ")
            for j in range(num_dim * 2):
                cgutils.printf(
                    builder, "%d ", builder.load(
                        builder.gep(
                            sched, [
                                context.get_constant(
                                    types.intp, i * num_dim * 2 + j)])))
            cgutils.printf(builder, "\n")

    # ----------------------------------------------------------------------------
    # Prepare arguments: args, shapes, steps, data
    all_args = [lowerer.loadvar(x) for x in expr_args[:ninouts]] + redarrs
    num_args = len(all_args)
    num_inps = len(sin) + 1
    args = cgutils.alloca_once(
        builder,
        byte_ptr_t,
        size=context.get_constant(
            types.intp,
            1 + num_args),
        name="pargs")
    array_strides = []
    # sched goes first
    builder.store(builder.bitcast(sched, byte_ptr_t), args)
    array_strides.append(context.get_constant(types.intp, sizeof_intp))
    red_shapes = {}
    rv_to_arg_dict = {}
    # followed by other arguments
    for i in range(num_args):
        arg = all_args[i]
        var = expr_args[i]
        aty = expr_arg_types[i]
        dst = builder.gep(args, [context.get_constant(types.intp, i + 1)])
        if i >= ninouts:  # reduction variables
            ary = context.make_array(aty)(context, builder, arg)
            strides = cgutils.unpack_tuple(builder, ary.strides, aty.ndim)
            ary_shapes = cgutils.unpack_tuple(builder, ary.shape, aty.ndim)
            # Start from 1 because we skip the first dimension of length num_threads just like sched.
            for j in range(1, len(strides)):
                array_strides.append(strides[j])
            red_shapes[i] = ary_shapes[1:]
            builder.store(builder.bitcast(ary.data, byte_ptr_t), dst)
        elif isinstance(aty, types.ArrayCompatible):
            if var in races:
                typ = context.get_data_type(
                    aty.dtype) if aty.dtype != types.boolean else lc.Type.int(1)

                rv_arg = cgutils.alloca_once(builder, typ)
                builder.store(arg, rv_arg)
                builder.store(builder.bitcast(rv_arg, byte_ptr_t), dst)
                rv_to_arg_dict[var] = (arg, rv_arg)

                array_strides.append(context.get_constant(types.intp, context.get_abi_sizeof(typ)))
            else:
                ary = context.make_array(aty)(context, builder, arg)
                strides = cgutils.unpack_tuple(builder, ary.strides, aty.ndim)
                for j in range(len(strides)):
                    array_strides.append(strides[j])
                builder.store(builder.bitcast(ary.data, byte_ptr_t), dst)
        else:
            if i < num_inps:
                # Scalar input, need to store the value in an array of size 1
                typ = context.get_data_type(
                    aty) if aty != types.boolean else lc.Type.int(1)
                ptr = cgutils.alloca_once(builder, typ)
                builder.store(arg, ptr)
            else:
                # Scalar output, must allocate
                typ = context.get_data_type(
                    aty) if aty != types.boolean else lc.Type.int(1)
                ptr = cgutils.alloca_once(builder, typ)
            builder.store(builder.bitcast(ptr, byte_ptr_t), dst)

    # ----------------------------------------------------------------------------
    # Next, we prepare the individual dimension info recorded in gu_signature
    sig_dim_dict = {}
    occurances = []
    occurances = [sched_sig[0]]
    sig_dim_dict[sched_sig[0]] = context.get_constant(types.intp, 2 * num_dim)
    assert len(expr_args) == len(all_args)
    assert len(expr_args) == len(expr_arg_types)
    assert len(expr_args) == len(sin + sout)
    assert len(expr_args) == len(outer_sig.args[1:])
    for var, arg, aty, gu_sig in zip(expr_args, all_args,
                                     expr_arg_types, sin + sout):
        if isinstance(aty, types.npytypes.Array):
            i = aty.ndim - len(gu_sig)
        else:
            i = 0
        if config.DEBUG_ARRAY_OPT:
            print("var =", var, "gu_sig =", gu_sig, "type =", aty, "i =", i)

        for dim_sym in gu_sig:
            if config.DEBUG_ARRAY_OPT:
                print("var = ", var, " type = ", aty)
            if var in races:
                sig_dim_dict[dim_sym] = context.get_constant(types.intp, 1)
            else:
                ary = context.make_array(aty)(context, builder, arg)
                shapes = cgutils.unpack_tuple(builder, ary.shape, aty.ndim)
                sig_dim_dict[dim_sym] = shapes[i]

            if not (dim_sym in occurances):
                if config.DEBUG_ARRAY_OPT:
                    print("dim_sym = ", dim_sym, ", i = ", i)
                    cgutils.printf(builder, dim_sym + " = %d\n", sig_dim_dict[dim_sym])
                occurances.append(dim_sym)
            i = i + 1

    # ----------------------------------------------------------------------------
    # Prepare shapes, which is a single number (outer loop size), followed by
    # the size of individual shape variables.
    nshapes = len(sig_dim_dict) + 1
    shapes = cgutils.alloca_once(builder, intp_t, size=nshapes, name="pshape")
    # For now, outer loop size is the same as number of threads
    builder.store(context.get_constant(types.intp, get_thread_count()), shapes)
    # Individual shape variables go next
    i = 1
    for dim_sym in occurances:
        if config.DEBUG_ARRAY_OPT:
            cgutils.printf(builder, dim_sym + " = %d\n", sig_dim_dict[dim_sym])
        builder.store(
            sig_dim_dict[dim_sym], builder.gep(
                shapes, [
                    context.get_constant(
                        types.intp, i)]))
        i = i + 1

    # ----------------------------------------------------------------------------
    # Prepare steps for each argument. Note that all steps are counted in
    # bytes.
    num_steps = num_args + 1 + len(array_strides)
    steps = cgutils.alloca_once(
        builder, intp_t, size=context.get_constant(
            types.intp, num_steps), name="psteps")
    # First goes the step size for sched, which is 2 * num_dim
    builder.store(context.get_constant(types.intp, 2 * num_dim * sizeof_intp),
                  steps)
    # The steps for all others are 0, except for reduction results.
    for i in range(num_args):
        if i >= ninouts:  # steps for reduction vars are abi_sizeof(typ)
            j = i - ninouts
            # Get the base dtype of the reduction array.
            redtyp = lowerer.fndesc.typemap[redvars[j]]
            red_stride = None
            if isinstance(redtyp, types.npytypes.Array):
                redtyp = redtyp.dtype
                red_stride = red_shapes[i]
            typ = context.get_value_type(redtyp)
            sizeof = context.get_abi_sizeof(typ)
            # Set stepsize to the size of that dtype.
            stepsize = context.get_constant(types.intp, sizeof)
            if red_stride != None:
                for rs in red_stride:
                    stepsize = builder.mul(stepsize, rs)
        else:
            # steps are strides
            stepsize = zero
        dst = builder.gep(steps, [context.get_constant(types.intp, 1 + i)])
        builder.store(stepsize, dst)
    for j in range(len(array_strides)):
        dst = builder.gep(
            steps, [
                context.get_constant(
                    types.intp, 1 + num_args + j)])
        builder.store(array_strides[j], dst)

    # ----------------------------------------------------------------------------
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

    for k, v in rv_to_arg_dict.items():
        arg, rv_arg = v
        only_elem_ptr = builder.gep(rv_arg, [context.get_constant(types.intp, 0)])
        builder.store(builder.load(only_elem_ptr), lowerer.getvar(k))

    scope = init_block.scope
    loc = init_block.loc
