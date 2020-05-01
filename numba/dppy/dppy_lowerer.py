from __future__ import print_function, division, absolute_import

import ast
import copy
from collections import OrderedDict
import linecache
import os
import sys
import numpy as np

import numba
from .. import (compiler,
                ir,
                types,
                six,
                sigutils,
                lowering,
                parfor,
                funcdesc)
from numba.ir_utils import (add_offset_to_labels,
                            replace_var_names,
                            remove_dels,
                            legalize_names,
                            mk_unique_var,
                            rename_labels,
                            get_name_var_table,
                            visit_vars_inner,
                            guard,
                            find_callname,
                            remove_dead,
                            get_call_table,
                            is_pure,
                            build_definitions,
                            get_np_ufunc_typ,
                            get_unused_var_name,
                            find_potential_aliases,
                            is_const_call)
from ..typing import signature
from numba import config, dppy
from numba.targets.cpu import ParallelOptions
from numba.six import exec_
import operator

import warnings
from ..errors import NumbaParallelSafetyWarning

from .target import SPIR_GENERIC_ADDRSPACE
from .dufunc_inliner import dufunc_inliner
from . import dppy_host_fn_call_gen as dppy_call_gen
import dppy.core as driver


def _print_block(block):
    for i, inst in enumerate(block.body):
        print("    ", i, inst)

def _print_body(body_dict):
    '''Pretty-print a set of IR blocks.
    '''
    for label, block in body_dict.items():
        print("label: ", label)
        _print_block(block)


# This loop scheduler is pretty basic, there is only
# 3 dimension allowed in OpenCL, so to make the backend
# functional we will schedule the first 3 dimensions
# through OpenCL and generate for loops for the remaining
# dimensions
def _schedule_loop(parfor_dim, legal_loop_indices, loop_ranges, param_dict):
    gufunc_txt    = ""
    global_id_dim = 0
    for_loop_dim  = parfor_dim

    if parfor_dim > 3:
        global_id_dim = 3
    else:
        global_id_dim = parfor_dim

    for eachdim in range(global_id_dim):
        gufunc_txt += ("    " + legal_loop_indices[eachdim] + " = "
                       + "dppy.get_global_id(" + str(eachdim) + ")\n")


    for eachdim in range(global_id_dim, for_loop_dim):
        for indent in range(1 + (eachdim - global_id_dim)):
            gufunc_txt += "    "

        start, stop, step = loop_ranges[eachdim]
        start = param_dict.get(str(start), start)
        stop = param_dict.get(str(stop), stop)
        gufunc_txt += ("for " +
                   legal_loop_indices[eachdim] +
                   " in range(" + str(start) +
                   ", " + str(stop) +
                   " + 1):\n")

    for eachdim in range(global_id_dim, for_loop_dim):
        for indent in range(1 + (eachdim - global_id_dim)):
            gufunc_txt += "    "

    return gufunc_txt


def _dbgprint_after_each_array_assignments(lowerer, loop_body, typemap):
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
                print_node = ir.Print(args=[lhs, inst.target], vararg=None,
                                      loc=loc)
                new_block.append(print_node)
                sig = numba.typing.signature(types.none, typemap[lhs.name],
                                             typemap[inst.target.name])
                lowerer.fndesc.calltypes[print_node] = sig
        loop_body[label] = new_block


def replace_var_with_array_in_block(vars, block, typemap, calltypes):
    new_block = []
    for inst in block.body:
        if isinstance(inst, ir.Assign) and inst.target.name in vars:
            const_node = ir.Const(0, inst.loc)
            const_var = ir.Var(inst.target.scope, mk_unique_var("$const_ind_0"),
                               inst.loc)
            typemap[const_var.name] = types.uintp
            const_assign = ir.Assign(const_node, const_var, inst.loc)
            new_block.append(const_assign)

            setitem_node = ir.SetItem(inst.target, const_var, inst.value,
                                      inst.loc)
            calltypes[setitem_node] = signature(
                types.none, types.npytypes.Array(typemap[inst.target.name], 1,
                                                 "C"), types.intp,
                                                 typemap[inst.target.name])
            new_block.append(setitem_node)
            continue
        elif isinstance(inst, parfor.Parfor):
            replace_var_with_array_internal(vars, {0: inst.init_block},
                                            typemap, calltypes)
            replace_var_with_array_internal(vars, inst.loop_body,
                                            typemap, calltypes)

        new_block.append(inst)
    return new_block

def replace_var_with_array_internal(vars, loop_body, typemap, calltypes):
    for label, block in loop_body.items():
        block.body = replace_var_with_array_in_block(vars, block, typemap,
                                                     calltypes)

def replace_var_with_array(vars, loop_body, typemap, calltypes):
    replace_var_with_array_internal(vars, loop_body, typemap, calltypes)
    for v in vars:
        el_typ = typemap[v]
        typemap.pop(v, None)
        typemap[v] = types.npytypes.Array(el_typ, 1, "C")


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


def legalize_names_with_typemap(names, typemap):
    """ We use ir_utils.legalize_names to replace internal IR variable names
        containing illegal characters (e.g. period) with a legal character
        (underscore) so as to create legal variable names.
        The original variable names are in the typemap so we also
        need to add the legalized name to the typemap as well.
    """
    outdict = legalize_names(names)
    # For each pair in the dict of legalized names...
    for x, y in outdict.items():
        # If the name had some legalization change to it...
        if x != y:
            # Set the type of the new name the same as the type of the old name.
            typemap[y] = typemap[x]
    return outdict


def to_scalar_from_0d(x):
    if isinstance(x, types.ArrayCompatible):
        if x.ndim == 0:
            return x.dtype
    return x

def find_setitems_block(setitems, block, typemap):
    for inst in block.body:
        if isinstance(inst, ir.StaticSetItem) or isinstance(inst, ir.SetItem):
            setitems.add(inst.target.name)
        elif isinstance(inst, parfor.Parfor):
            find_setitems_block(setitems, inst.init_block, typemap)
            find_setitems_body(setitems, inst.loop_body, typemap)

def find_setitems_body(setitems, loop_body, typemap):
    """
      Find the arrays that are written into (goes into setitems)
    """
    for label, block in loop_body.items():
        find_setitems_block(setitems, block, typemap)

def _create_gufunc_for_regular_parfor():
    #TODO
    pass


def _create_gufunc_for_reduction_parfor():
    raise ValueError("Reductions are not yet supported via parfor")


def _create_gufunc_for_parfor_body(
        lowerer,
        parfor,
        typemap,
        typingctx,
        targetctx,
        flags,
        loop_ranges,
        locals,
        has_aliases,
        index_var_typ,
        races):
    '''
    Takes a parfor and creates a gufunc function for its body. There
    are two parts to this function:

        1) Code to iterate across the iteration space as defined by
           the schedule.
        2) The parfor body that does the work for a single point in
           the iteration space.

    Part 1 is created as Python text for simplicity with a sentinel
    assignment to mark the point in the IR where the parfor body
    should be added. This Python text is 'exec'ed into existence and its
    IR retrieved with run_frontend. The IR is scanned for the sentinel
    assignment where that basic block is split and the IR for the parfor
    body inserted.
    '''

    loc = parfor.init_block.loc

    # The parfor body and the main function body share ir.Var nodes.
    # We have to do some replacements of Var names in the parfor body
    # to make them legal parameter names. If we don't copy then the
    # Vars in the main function also would incorrectly change their name.

    loop_body = copy.copy(parfor.loop_body)
    remove_dels(loop_body)

    parfor_dim = len(parfor.loop_nests)
    loop_indices = [l.index_variable.name for l in parfor.loop_nests]

    # Get all the parfor params.
    parfor_params = parfor.params

    for start, stop, step in loop_ranges:
        if isinstance(start, ir.Var):
            parfor_params.add(start.name)
        if isinstance(stop, ir.Var):
            parfor_params.add(stop.name)

    # Get just the outputs of the parfor.
    parfor_outputs = numba.parfor.get_parfor_outputs(parfor, parfor_params)

    # Get all parfor reduction vars, and operators.
    typemap = lowerer.fndesc.typemap

    parfor_redvars, parfor_reddict = numba.parfor.get_parfor_reductions(
                                        lowerer.func_ir,
                                        parfor,
                                        parfor_params,
                                        lowerer.fndesc.calltypes)
    has_reduction = False if len(parfor_redvars) == 0 else True

    if has_reduction:
        _create_gufunc_for_reduction_parfor()

    # Compute just the parfor inputs as a set difference.
    parfor_inputs = sorted(
        list(
            set(parfor_params) -
            set(parfor_outputs)))

    for race in races:
        msg = ("Variable %s used in parallel loop may be written "
               "to simultaneously by multiple workers and may result "
               "in non-deterministic or unintended results." % race)
        warnings.warn(NumbaParallelSafetyWarning(msg, loc))
    replace_var_with_array(races, loop_body, typemap, lowerer.fndesc.calltypes)

    if config.DEBUG_ARRAY_OPT >= 1:
        print("parfor_params = ", parfor_params, type(parfor_params))
        print("parfor_outputs = ", parfor_outputs, type(parfor_outputs))
        print("parfor_inputs = ", parfor_inputs, type(parfor_inputs))

    # Reorder all the params so that inputs go first then outputs.
    parfor_params = parfor_inputs + parfor_outputs

    def addrspace_from(params, def_addr):
        addrspaces = []
        for p in params:
            if isinstance(to_scalar_from_0d(typemap[p]),
                          types.npytypes.Array):
                addrspaces.append(def_addr)
            else:
                addrspaces.append(None)
        return addrspaces

    addrspaces = addrspace_from(parfor_params,
                                numba.dppy.target.SPIR_GLOBAL_ADDRSPACE)

    if config.DEBUG_ARRAY_OPT >= 1:
        print("parfor_params = ", parfor_params, type(parfor_params))
        print("loop_indices = ", loop_indices, type(loop_indices))
        print("loop_body = ", loop_body, type(loop_body))
        _print_body(loop_body)

    # Some Var are not legal parameter names so create a dict of
    # potentially illegal param name to guaranteed legal name.
    param_dict = legalize_names_with_typemap(parfor_params, typemap)
    if config.DEBUG_ARRAY_OPT >= 1:
        print("param_dict = ", sorted(param_dict.items()), type(param_dict))

    # Some loop_indices are not legal parameter names so create a dict
    # of potentially illegal loop index to guaranteed legal name.
    ind_dict = legalize_names_with_typemap(loop_indices, typemap)
    # Compute a new list of legal loop index names.
    legal_loop_indices = [ind_dict[v] for v in loop_indices]

    if config.DEBUG_ARRAY_OPT >= 1:
        print("ind_dict = ", sorted(ind_dict.items()), type(ind_dict))
        print("legal_loop_indices = ",legal_loop_indices,
              type(legal_loop_indices))

        for pd in parfor_params:
            print("pd = ", pd)
            print("pd type = ", typemap[pd], type(typemap[pd]))

    # Get the types of each parameter.
    param_types = [to_scalar_from_0d(typemap[v]) for v in parfor_params]

    param_types_addrspaces = copy.copy(param_types)

    # Calculate types of args passed to gufunc.
    func_arg_types = [typemap[v] for v in (parfor_inputs + parfor_outputs)]
    assert(len(param_types_addrspaces) == len(addrspaces))
    for i in range(len(param_types_addrspaces)):
        if addrspaces[i] is not None:
            #print("before:", id(param_types_addrspaces[i]))
            assert(isinstance(param_types_addrspaces[i], types.npytypes.Array))
            param_types_addrspaces[i] = (param_types_addrspaces[i]
                                        .copy(addrspace=addrspaces[i]))
            #print("setting param type", i, param_types[i], id(param_types[i]),
            #      "to addrspace", param_types_addrspaces[i].addrspace)

    def print_arg_with_addrspaces(args):
        for a in args:
            print(a, type(a))
            if isinstance(a, types.npytypes.Array):
                print("addrspace:", a.addrspace)

    if config.DEBUG_ARRAY_OPT >= 1:
        print_arg_with_addrspaces(param_types)
        print("func_arg_types = ", func_arg_types, type(func_arg_types))

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

    if config.DEBUG_ARRAY_OPT >= 1:
        print(
            "legal parfor_params = ",
            parfor_params,
            type(parfor_params))


    # Determine the unique names of the scheduling and gufunc functions.
    gufunc_name = "__numba_parfor_gufunc_%s" % (
        hex(hash(parfor)).replace("-", "_"))

    if config.DEBUG_ARRAY_OPT:
        # print("sched_func_name ", type(sched_func_name), sched_func_name)
        print("gufunc_name ", type(gufunc_name), gufunc_name)

    gufunc_txt = ""

    # Create the gufunc function.
    gufunc_txt += "def " + gufunc_name
    gufunc_txt += "(" + (", ".join(parfor_params)) + "):\n"


    gufunc_txt += _schedule_loop(parfor_dim, legal_loop_indices, loop_ranges,
                                 param_dict)

    # Add the sentinel assignment so that we can find the loop body position
    # in the IR.
    gufunc_txt += "    "
    gufunc_txt += sentinel_name + " = 0\n"

    # gufunc returns nothing
    gufunc_txt += "    return None\n"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
        sys.stdout.flush()
    # Force gufunc outline into existence.
    globls = {"np": np, "numba": numba, "dppy": dppy}
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

    prs_dict = {}
    pss_dict = {}
    pspmd_dict = {}

    gufunc_param_types = param_types

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
        _dbgprint_after_each_array_assignments(lowerer, loop_body, typemap)

    if config.DEBUG_ARRAY_OPT:
        print("parfor loop body")
        _print_body(loop_body)

    wrapped_blocks = wrap_loop_body(loop_body)
    #hoisted, not_hoisted = hoist(parfor_params, loop_body,
    #                             typemap, wrapped_blocks)
    setitems = set()
    find_setitems_body(setitems, loop_body, typemap)

    hoisted = []
    not_hoisted = []

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
            if (isinstance(inst, ir.Assign) and
                inst.target.name == sentinel_name):
                # We found the sentinel assignment.
                loc = inst.loc
                scope = block.scope
                # split block across __sentinel__
                # A new block is allocated for the statements prior to the
                # sentinel but the new block maintains the current block label.
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
                # Add a jump from the last parfor body block to the block
                # containing statements after the sentinel.
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
        sys.stdout.flush()

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

    remove_dead(gufunc_ir.blocks, gufunc_ir.arg_names, gufunc_ir, typemap)

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir after remove dead")
        gufunc_ir.dump()

    kernel_sig = signature(types.none, *gufunc_param_types)

    if config.DEBUG_ARRAY_OPT:
        sys.stdout.flush()

    if config.DEBUG_ARRAY_OPT:
        print('before DUFunc inlining'.center(80, '-'))
        gufunc_ir.dump()

    # Inlining all DUFuncs
    dufunc_inliner(gufunc_ir, lowerer.fndesc.calltypes, typemap,
                   lowerer.context.typing_context)

    if config.DEBUG_ARRAY_OPT:
        print('after DUFunc inline'.center(80, '-'))
        gufunc_ir.dump()

    # FIXME : We should not always use gpu device, instead select the default
    # device as configured in dppy.
    kernel_func = numba.dppy.compiler.compile_kernel_parfor(
        driver.runtime.get_gpu_device(),
        gufunc_ir,
        gufunc_param_types,
        param_types_addrspaces)

    flags.noalias = old_alias

    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return kernel_func, parfor_args, kernel_sig, func_arg_types, setitems


def _lower_parfor_gufunc(lowerer, parfor):
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
    if config.DEBUG_ARRAY_OPT:
        sys.stdout.flush()

    loc = parfor.init_block.loc
    scope = parfor.init_block.scope

    # produce instructions for init_block
    if config.DEBUG_ARRAY_OPT:
        print("init_block = ", parfor.init_block, type(parfor.init_block))
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

    # run get_parfor_outputs() and get_parfor_reductions() before
    # gufunc creation since Jumps are modified so CFG of loop_body
    # dict will become invalid
    assert parfor.params != None

    parfor_output_arrays = numba.parfor.get_parfor_outputs(
        parfor, parfor.params)


    # compile parfor body as a separate function to be used with GUFuncWrapper
    flags = copy.copy(parfor.flags)
    flags.set('error_model', 'numpy')

    # Can't get here unless flags.set('auto_parallel', ParallelOptions(True))
    index_var_typ = typemap[parfor.loop_nests[0].index_variable.name]

    # index variables should have the same type, check rest of indices
    for l in parfor.loop_nests[1:]:
        assert typemap[l.index_variable.name] == index_var_typ

    numba.parfor.sequential_parfor_lowering = True
    loop_ranges = [(l.start, l.stop, l.step) for l in parfor.loop_nests]

    func, func_args, func_sig, func_arg_types, modified_arrays =(
    _create_gufunc_for_parfor_body(
        lowerer,
        parfor,
        typemap,
        typingctx,
        targetctx,
        flags,
        loop_ranges,
        {},
        bool(alias_map),
        index_var_typ,
        parfor.races))

    numba.parfor.sequential_parfor_lowering = False

    # get the shape signature
    get_shape_classes = parfor.get_shape_classes

    num_inputs = len(func_args) - len(parfor_output_arrays)
    if config.DEBUG_ARRAY_OPT:
        print("func", func, type(func))
        print("func_args", func_args, type(func_args))
        print("func_sig", func_sig, type(func_sig))
        print("num_inputs = ", num_inputs)
        print("parfor_outputs = ", parfor_output_arrays)

    # call the func in parallel by wrapping it with ParallelGUFuncBuilder
    if config.DEBUG_ARRAY_OPT:
        print("loop_nests = ", parfor.loop_nests)
        print("loop_ranges = ", loop_ranges)

    gu_signature = _create_shape_signature(
        parfor.get_shape_classes,
        num_inputs,
        func_args,
        func_sig,
        parfor.races,
        typemap)

    generate_dppy_host_wrapper(
        lowerer,
        func,
        gu_signature,
        func_sig,
        func_args,
        num_inputs,
        func_arg_types,
        loop_ranges,
        parfor.init_block,
        index_var_typ,
        parfor.races,
        modified_arrays)

    if config.DEBUG_ARRAY_OPT:
        sys.stdout.flush()

    # Restore the original typemap of the function that was replaced
    # temporarily at the beginning of this function.
    lowerer.fndesc.typemap = orig_typemap


def _create_shape_signature(
        get_shape_classes,
        num_inputs,
        #num_reductions,
        args,
        func_sig,
        races,
        typemap):
    '''Create shape signature for GUFunc
    '''
    if config.DEBUG_ARRAY_OPT:
        print("_create_shape_signature", num_inputs, args)
        arg_start_print = 0
        for i in args[arg_start_print:]:
            print("argument", i, type(i), get_shape_classes(i, typemap=typemap))

    #num_inouts = len(args) - num_reductions
    num_inouts = len(args)
    # maximum class number for array shapes
    classes = [get_shape_classes(var, typemap=typemap)
               if var not in races else (-1,) for var in args[1:]]
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
        gu_sin.append(dim_syms)
        syms_sin += dim_syms
    return (gu_sin, gu_sout)


# Keep all the dppy kernels and programs created alive indefinitely.
keep_alive_kernels = []

def generate_dppy_host_wrapper(lowerer,
                               cres,
                               gu_signature,
                               outer_sig,
                               expr_args,
                               num_inputs,
                               expr_arg_types,
                               loop_ranges,
                               init_block,
                               index_var_typ,
                               races,
                               modified_arrays):
    '''
    Adds the call to the gufunc function from the main function.
    '''
    context = lowerer.context
    builder = lowerer.builder
    sin, sout = gu_signature
    num_dim = len(loop_ranges)

    if config.DEBUG_ARRAY_OPT:
        print("generate_dppy_host_wrapper")
        print("args = ", expr_args)
        print("outer_sig = ", outer_sig.args, outer_sig.return_type,
              outer_sig.recvr, outer_sig.pysig)
        print("loop_ranges = ", loop_ranges)
        print("expr_args", expr_args)
        print("expr_arg_types", expr_arg_types)
        print("gu_signature", gu_signature)
        print("sin", sin)
        print("sout", sout)
        print("cres", cres, type(cres))
        print("modified_arrays", modified_arrays)
#        print("cres.library", cres.library, type(cres.library))
#        print("cres.fndesc", cres.fndesc, type(cres.fndesc))

    # get dppy_cpu_portion_lowerer object
    dppy_cpu_lowerer = dppy_call_gen.DPPyHostFunctionCallsGenerator(
                           lowerer, cres, num_inputs)

    # Compute number of args ------------------------------------------------
    num_expanded_args = 0

    for arg_type in expr_arg_types:
        if isinstance(arg_type, types.npytypes.Array):
            num_expanded_args += 5 + (2 * arg_type.ndim)
        else:
            num_expanded_args += 1

    if config.DEBUG_ARRAY_OPT:
        print("num_expanded_args = ", num_expanded_args)

    # now that we know the total number of kernel args, lets allocate
    # a kernel_arg array
    dppy_cpu_lowerer.allocate_kenrel_arg_array(num_expanded_args)

    ninouts = len(expr_args)

    def getvar_or_none(lowerer, x):
        try:
            return lowerer.getvar(x)
        except:
            return None

    def loadvar_or_none(lowerer, x):
        try:
            return lowerer.loadvar(x)
        except:
            return None

    def val_type_or_none(context, lowerer, x):
        try:
            return context.get_value_type(lowerer.fndesc.typemap[x])
        except:
            return None

    all_llvm_args = [getvar_or_none(lowerer, x) for x in expr_args[:ninouts]]
    all_val_types = ([val_type_or_none(context, lowerer, x)
                     for x in expr_args[:ninouts]])
    all_args = [loadvar_or_none(lowerer, x) for x in expr_args[:ninouts]]

    keep_alive_kernels.append(cres)

    # -----------------------------------------------------------------------
    # Call clSetKernelArg for each arg and create arg array for
    # the enqueue function. Put each part of each argument into
    # kernel_arg_array.
    for var, llvm_arg, arg_type, gu_sig, val_type, index in zip(
        expr_args, all_llvm_args, expr_arg_types, sin + sout, all_val_types,
        range(len(expr_args))):

        if config.DEBUG_ARRAY_OPT:
            print("var:", var, type(var),
                  "\n\tllvm_arg:", llvm_arg, type(llvm_arg),
                  "\n\targ_type:", arg_type, type(arg_type),
                  "\n\tgu_sig:", gu_sig,
                  "\n\tval_type:", val_type, type(val_type),
                  "\n\tindex:", index)

        dppy_cpu_lowerer.process_kernel_arg(var, llvm_arg, arg_type, gu_sig,
                                            val_type, index, modified_arrays)
    # -----------------------------------------------------------------------

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

    dppy_cpu_lowerer.enqueue_kernel_and_read_back(loop_ranges)


from numba.lowering import Lower

class DPPyLower(Lower):
    def __init__(self, context, library, fndesc, func_ir, metadata=None):
        Lower.__init__(self, context, library, fndesc, func_ir, metadata)
        lowering.lower_extensions[parfor.Parfor] = _lower_parfor_gufunc

def dppy_lower_array_expr(lowerer, expr):
    raise NotImplementedError(expr)
