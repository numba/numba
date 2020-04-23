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
import llvmlite.ir as lir
import llvmlite.binding as lb

import numba
from .. import (compiler,
                ir,
                types,
                six,
                cgutils,
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
                            is_const_call,
                            dead_code_elimination,
                            simplify_CFG)
from ..typing import signature
from numba import config, dppy
from numba.targets.cpu import ParallelOptions
from numba.six import exec_
import types as pytypes
import operator

import warnings
from ..errors import NumbaParallelSafetyWarning

from numba.dppy.target import SPIR_GENERIC_ADDRSPACE
import dppy.core as driver


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

def add_to_def_once_sets(a_def, def_once, def_more):
    '''If the variable is already defined more than once, do nothing.
       Else if defined exactly once previously then transition this
       variable to the defined more than once set (remove it from
       def_once set and add to def_more set).
       Else this must be the first time we've seen this variable defined
       so add to def_once set.
    '''
    if a_def in def_more:
        pass
    elif a_def in def_once:
        def_more.add(a_def)
        def_once.remove(a_def)
    else:
        def_once.add(a_def)

def compute_def_once_block(block, def_once, def_more, getattr_taken, typemap, module_assigns):
    '''Effect changes to the set of variables defined once or more than once
       for a single block.
       block - the block to process
       def_once - set of variable names known to be defined exactly once
       def_more - set of variable names known to be defined more than once
       getattr_taken - dict mapping variable name to tuple of object and attribute taken
       module_assigns - dict mapping variable name to the Global that they came from
    '''
    # The only "defs" occur in assignments, so find such instructions.
    assignments = block.find_insts(ir.Assign)
    # For each assignment...
    for one_assign in assignments:
        # Get the LHS/target of the assignment.
        a_def = one_assign.target.name
        # Add variable to def sets.
        add_to_def_once_sets(a_def, def_once, def_more)

        rhs = one_assign.value
        if isinstance(rhs, ir.Global):
            # Remember assignments of the form "a = Global(...)"
            # Is this a module?
            if isinstance(rhs.value, pytypes.ModuleType):
                module_assigns[a_def] = rhs.value.__name__
        if isinstance(rhs, ir.Expr) and rhs.op == 'getattr' and rhs.value.name in def_once:
            # Remember assignments of the form "a = b.c"
            getattr_taken[a_def] = (rhs.value.name, rhs.attr)
        if isinstance(rhs, ir.Expr) and rhs.op == 'call' and rhs.func.name in getattr_taken:
            # If "a" is being called then lookup the getattr definition of "a"
            # as above, getting the module variable "b" (base_obj)
            # and the attribute "c" (base_attr).
            base_obj, base_attr = getattr_taken[rhs.func.name]
            if base_obj in module_assigns:
                # If we know the definition of the module variable then get the module
                # name from module_assigns.
                base_mod_name = module_assigns[base_obj]
                if not is_const_call(base_mod_name, base_attr):
                    # Calling a method on an object could modify the object and is thus
                    # like a def of that object.  We call is_const_call to see if this module/attribute
                    # combination is known to not modify the module state.  If we don't know that
                    # the combination is safe then we have to assume there could be a modification to
                    # the module and thus add the module variable as defined more than once.
                    add_to_def_once_sets(base_obj, def_once, def_more)
            else:
                # Assume the worst and say that base_obj could be modified by the call.
                add_to_def_once_sets(base_obj, def_once, def_more)
        if isinstance(rhs, ir.Expr) and rhs.op == 'call':
            # If a mutable object is passed to a function, then it may be changed and
            # therefore can't be hoisted.
            # For each argument to the function...
            for argvar in rhs.args:
                # Get the argument's type.
                if isinstance(argvar, ir.Var):
                    argvar = argvar.name
                avtype = typemap[argvar]
                # If that type doesn't have a mutable attribute or it does and it's set to
                # not mutable then this usage is safe for hoisting.
                if getattr(avtype, 'mutable', False):
                    # Here we have a mutable variable passed to a function so add this variable
                    # to the def lists.
                    add_to_def_once_sets(argvar, def_once, def_more)

def compute_def_once_internal(loop_body, def_once, def_more, getattr_taken, typemap, module_assigns):
    '''Compute the set of variables defined exactly once in the given set of blocks
       and use the given sets for storing which variables are defined once, more than
       once and which have had a getattr call on them.
    '''
    # For each block...
    for label, block in loop_body.items():
        # Scan this block and effect changes to def_once, def_more, and getattr_taken
        # based on the instructions in that block.
        compute_def_once_block(block, def_once, def_more, getattr_taken, typemap, module_assigns)
        # Have to recursively process parfors manually here.
        for inst in block.body:
            if isinstance(inst, parfor.Parfor):
                # Recursively compute for the parfor's init block.
                compute_def_once_block(inst.init_block, def_once, def_more, getattr_taken, typemap, module_assigns)
                # Recursively compute for the parfor's loop body.
                compute_def_once_internal(inst.loop_body, def_once, def_more, getattr_taken, typemap, module_assigns)

def compute_def_once(loop_body, typemap):
    '''Compute the set of variables defined exactly once in the given set of blocks.
    '''
    def_once = set()   # set to hold variables defined exactly once
    def_more = set()   # set to hold variables defined more than once
    getattr_taken = {}
    module_assigns = {}
    compute_def_once_internal(loop_body, def_once, def_more, getattr_taken, typemap, module_assigns)
    return def_once

def find_vars(var, varset):
    assert isinstance(var, ir.Var)
    varset.add(var.name)
    return var

def _hoist_internal(inst, dep_on_param, call_table, hoisted, not_hoisted,
                    typemap, stored_arrays):
    if inst.target.name in stored_arrays:
        not_hoisted.append((inst, "stored array"))
        if config.DEBUG_ARRAY_OPT >= 1:
            print("Instruction", inst, " could not be hoisted because the created array is stored.")
        return False

    uses = set()
    visit_vars_inner(inst.value, find_vars, uses)
    diff = uses.difference(dep_on_param)
    if config.DEBUG_ARRAY_OPT >= 1:
        print("_hoist_internal:", inst, "uses:", uses, "diff:", diff)
    if len(diff) == 0 and is_pure(inst.value, None, call_table):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("Will hoist instruction", inst, typemap[inst.target.name])
        hoisted.append(inst)
        if not isinstance(typemap[inst.target.name], types.npytypes.Array):
            dep_on_param += [inst.target.name]
        return True
    else:
        if len(diff) > 0:
            not_hoisted.append((inst, "dependency"))
            if config.DEBUG_ARRAY_OPT >= 1:
                print("Instruction", inst, " could not be hoisted because of a dependency.")
        else:
            not_hoisted.append((inst, "not pure"))
            if config.DEBUG_ARRAY_OPT >= 1:
                print("Instruction", inst, " could not be hoisted because it isn't pure.")
    return False

def find_setitems_block(setitems, itemsset, block, typemap):
    for inst in block.body:
        if isinstance(inst, ir.StaticSetItem) or isinstance(inst, ir.SetItem):
            setitems.add(inst.target.name)
            # If we store a non-mutable object into an array then that is safe to hoist.
            # If the stored object is mutable and you hoist then multiple entries in the
            # outer array could reference the same object and changing one index would then
            # change other indices.
            if getattr(typemap[inst.value.name], "mutable", False):
                itemsset.add(inst.value.name)
        elif isinstance(inst, parfor.Parfor):
            find_setitems_block(setitems, itemsset, inst.init_block, typemap)
            find_setitems_body(setitems, itemsset, inst.loop_body, typemap)

def find_setitems_body(setitems, itemsset, loop_body, typemap):
    """
      Find the arrays that are written into (goes into setitems) and the
      mutable objects (mostly arrays) that are written into other arrays
      (goes into itemsset).
    """
    for label, block in loop_body.items():
        find_setitems_block(setitems, itemsset, block, typemap)

def hoist(parfor_params, loop_body, typemap, wrapped_blocks):
    dep_on_param = copy.copy(parfor_params)
    hoisted = []
    not_hoisted = []

    # Compute the set of variable defined exactly once in the loop body.
    def_once = compute_def_once(loop_body, typemap)
    (call_table, reverse_call_table) = get_call_table(wrapped_blocks)

    setitems = set()
    itemsset = set()
    find_setitems_body(setitems, itemsset, loop_body, typemap)
    dep_on_param = list(set(dep_on_param).difference(setitems))
    if config.DEBUG_ARRAY_OPT >= 1:
        print("hoist - def_once:", def_once, "setitems:",
              setitems, "itemsset:", itemsset, "dep_on_param:",
              dep_on_param, "parfor_params:", parfor_params)

    for label, block in loop_body.items():
        new_block = []
        for inst in block.body:
            if isinstance(inst, ir.Assign) and inst.target.name in def_once:
                if _hoist_internal(inst, dep_on_param, call_table,
                                   hoisted, not_hoisted, typemap, itemsset):
                    # don't add this instruction to the block since it is
                    # hoisted
                    continue
            elif isinstance(inst, parfor.Parfor):
                new_init_block = []
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("parfor")
                    inst.dump()
                for ib_inst in inst.init_block.body:
                    if (isinstance(ib_inst, ir.Assign) and
                        ib_inst.target.name in def_once):
                        if _hoist_internal(ib_inst, dep_on_param, call_table,
                                           hoisted, not_hoisted, typemap, itemsset):
                            # don't add this instuction to the block since it is hoisted
                            continue
                    new_init_block.append(ib_inst)
                inst.init_block.body = new_init_block

            new_block.append(inst)
        block.body = new_block
    return hoisted, not_hoisted


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
            print("before:", id(param_types_addrspaces[i]))
            assert(isinstance(param_types_addrspaces[i], types.npytypes.Array))
            param_types_addrspaces[i] = (param_types_addrspaces[i]
                                        .copy(addrspace=addrspaces[i]))
            print("setting param type", i, param_types[i], id(param_types[i]),
                  "to addrspace", param_types_addrspaces[i].addrspace)

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

#    for pindex in range(len(parfor_inputs)):
#        if ascontig and isinstance(param_types[pindex], types.npytypes.Array):
#            gufunc_txt += ("    " + parfor_params_orig[pindex]
#                + " = np.ascontiguousarray(" + parfor_params[pindex] + ")\n")

    for eachdim in range(parfor_dim):
        gufunc_txt += ("    " + legal_loop_indices[eachdim] + " = "
                       + "dppy.get_global_id(" + str(eachdim) + ")\n")

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
    #hoisted, not_hoisted = hoist(parfor_params, loop_body, typemap, wrapped_blocks)
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
            if isinstance(inst, ir.Assign) and inst.target.name == sentinel_name:
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

    print('before vectorize inline'.center(80, '-'))
    gufunc_ir.dump()

    # Inlining all DUFuncs
    dppy_dufunc_inliner(gufunc_ir, lowerer.fndesc.calltypes, typemap, lowerer.context.typing_context)


    print('after vectorize inline'.center(80, '-'))
    gufunc_ir.dump()

    kernel_func = numba.dppy.compiler.compile_kernel_parfor(
        driver.runtime.get_gpu_device(),
        gufunc_ir,
        gufunc_param_types,
        param_types_addrspaces)

    flags.noalias = old_alias

    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return kernel_func, parfor_args, kernel_sig, func_arg_types


def _lower_parfor_dppy_no_gufunc(lowerer, parfor):
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

    # run get_parfor_outputs() and get_parfor_reductions() before gufunc creation
    # since Jumps are modified so CFG of loop_body dict will become invalid
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

    func, func_args, func_sig, func_arg_types =(
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

    call_dppy(
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
        parfor.races)

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
        gu_sin.append(dim_syms)
        syms_sin += dim_syms
    return (gu_sin, gu_sout)


# Keep all the dppy kernels and programs created alive indefinitely.
keep_alive_kernels = []

def call_dppy(lowerer, cres,
              gu_signature,
              outer_sig,
              expr_args,
              num_inputs,
              expr_arg_types,
              loop_ranges,
              init_block,
              index_var_typ,
              races):
    '''
    Adds the call to the gufunc function from the main function.
    '''
    context = lowerer.context
    builder = lowerer.builder
    sin, sout = gu_signature
    num_dim = len(loop_ranges)

    if config.DEBUG_ARRAY_OPT:
        print("call_dppy")
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
#        print("cres.library", cres.library, type(cres.library))
#        print("cres.fndesc", cres.fndesc, type(cres.fndesc))

    # get dppy_cpu_portion_lowerer object
    dppy_cpu_lowerer = DPPyCPUPortionLowerer(lowerer, cres, num_inputs)

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
                                            val_type, index)
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


class DPPyCPUPortionLowerer(object):
    def __init__(self, lowerer, cres, num_inputs):
        self.lowerer = lowerer
        self.context = self.lowerer.context
        self.builder = self.lowerer.builder

        self.gpu_device = driver.runtime.get_gpu_device()
        self.gpu_device_env = self.gpu_device.get_env_ptr()
        self.gpu_device_int = int(driver.ffi.cast("uintptr_t",
                                                  self.gpu_device_env))

        self.kernel_t_obj = cres.kernel._kernel_t_obj[0]
        self.kernel_int = int(driver.ffi.cast("uintptr_t",
                                              self.kernel_t_obj))

        # Initialize commonly used LLVM types and constant
        self._init_llvm_types_and_constants()
        # Create functions that we need to call
        self._declare_functions()
        # Create a NULL void * pointer for meminfo and parent
        # parts of ndarray type args
        self.null_ptr = self._create_null_ptr()

        self.total_kernel_args = 0
        self.cur_arg           = 0
        self.num_inputs        = num_inputs

        # list of buffer that needs to comeback to host
        self.read_bufs_after_enqueue = []


    def _create_null_ptr(self):
        null_ptr = cgutils.alloca_once(self.builder, self.void_ptr_t,
                size=self.context.get_constant(types.uintp, 1), name="null_ptr")
        self.builder.store(
            self.builder.inttoptr(
                self.context.get_constant(types.uintp, 0), self.void_ptr_t),
                null_ptr)
        return null_ptr


    def _init_llvm_types_and_constants(self):
        self.byte_t          = lc.Type.int(8)
        self.byte_ptr_t      = lc.Type.pointer(self.byte_t)
        self.byte_ptr_ptr_t  = lc.Type.pointer(self.byte_ptr_t)
        self.intp_t          = self.context.get_value_type(types.intp)
        self.uintp_t         = self.context.get_value_type(types.uintp)
        self.intp_ptr_t      = lc.Type.pointer(self.intp_t)
        self.uintp_ptr_t     = lc.Type.pointer(self.uintp_t)
        self.zero            = self.context.get_constant(types.uintp, 0)
        self.one             = self.context.get_constant(types.uintp, 1)
        self.one_type        = self.one.type
        self.sizeof_intp     = self.context.get_abi_sizeof(self.intp_t)
        self.void_ptr_t      = self.context.get_value_type(types.voidptr)
        self.void_ptr_ptr_t  = lc.Type.pointer(self.void_ptr_t)
        self.sizeof_void_ptr = self.context.get_abi_sizeof(self.intp_t)
        self.gpu_device_int_const = self.context.get_constant(
                                        types.uintp, self.gpu_device_int)

    def _declare_functions(self):
        create_dppy_kernel_arg_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_ptr_t, self.intp_t, self.void_ptr_ptr_t])
        self.create_dppy_kernel_arg = self.builder.module.get_or_insert_function(create_dppy_kernel_arg_fnty,
                                                              name="create_dp_kernel_arg")

        create_dppy_kernel_arg_from_buffer_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_ptr_t, self.void_ptr_ptr_t])
        self.create_dppy_kernel_arg_from_buffer = self.builder.module.get_or_insert_function(
                                                   create_dppy_kernel_arg_from_buffer_fnty,
                                                   name="create_dp_kernel_arg_from_buffer")

        create_dppy_rw_mem_buffer_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.intp_t, self.void_ptr_ptr_t])
        self.create_dppy_rw_mem_buffer = self.builder.module.get_or_insert_function(
                                          create_dppy_rw_mem_buffer_fnty,
                                          name="create_dp_rw_mem_buffer")

        write_mem_buffer_to_device_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.void_ptr_t, self.intp_t, self.intp_t, self.intp_t, self.void_ptr_t])
        self.write_mem_buffer_to_device = self.builder.module.get_or_insert_function(
                                          write_mem_buffer_to_device_fnty,
                                          name="write_dp_mem_buffer_to_device")

        read_mem_buffer_from_device_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.void_ptr_t, self.intp_t, self.intp_t, self.intp_t, self.void_ptr_t])
        self.read_mem_buffer_from_device = self.builder.module.get_or_insert_function(
                                        read_mem_buffer_from_device_fnty,
                                        name="read_dp_mem_buffer_from_device")

        enqueue_kernel_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.void_ptr_t, self.intp_t, self.void_ptr_ptr_t,
                     self.intp_t, self.intp_ptr_t, self.intp_ptr_t])
        self.enqueue_kernel = self.builder.module.get_or_insert_function(
                                      enqueue_kernel_fnty,
                                      name="set_args_and_enqueue_dp_kernel_auto_blocking")


    def allocate_kenrel_arg_array(self, num_kernel_args):
        self.total_kernel_args = num_kernel_args

        # we need a kernel arg array to enqueue
        self.kernel_arg_array = cgutils.alloca_once(
            self.builder, self.void_ptr_t, size=self.context.get_constant(
                types.uintp, num_kernel_args), name="kernel_arg_array")


    def _call_dppy_kernel_arg_fn(self, args):
        kernel_arg = cgutils.alloca_once(self.builder, self.void_ptr_t,
                                         size=self.one, name="kernel_arg" + str(self.cur_arg))

        args.append(kernel_arg)
        self.builder.call(self.create_dppy_kernel_arg, args)
        dst = self.builder.gep(self.kernel_arg_array, [self.context.get_constant(types.intp, self.cur_arg)])
        self.cur_arg += 1
        self.builder.store(self.builder.load(kernel_arg), dst)


    def process_kernel_arg(self, var, llvm_arg, arg_type, gu_sig, val_type, index):

        if isinstance(arg_type, types.npytypes.Array):
            if llvm_arg is None:
                raise NotImplementedError(arg_type, var)

            # Handle meminfo.  Not used by kernel so just write a null pointer.
            args = [self.null_ptr, self.context.get_constant(types.uintp, self.sizeof_void_ptr)]
            self._call_dppy_kernel_arg_fn(args)

            # Handle parent.  Not used by kernel so just write a null pointer.
            args = [self.null_ptr, self.context.get_constant(types.uintp, self.sizeof_void_ptr)]
            self._call_dppy_kernel_arg_fn(args)

            # Handle array size
            array_size_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 2)])
            args = [self.builder.bitcast(array_size_member, self.void_ptr_ptr_t),
                    self.context.get_constant(types.uintp, self.sizeof_intp)]
            self._call_dppy_kernel_arg_fn(args)

            # Handle itemsize
            item_size_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 3)])
            args = [self.builder.bitcast(item_size_member, self.void_ptr_ptr_t),
                    self.context.get_constant(types.uintp, self.sizeof_intp)]
            self._call_dppy_kernel_arg_fn(args)

            # Calculate total buffer size
            total_size = cgutils.alloca_once(self.builder, self.intp_t,
                    size=self.one, name="total_size" + str(self.cur_arg))
            self.builder.store(self.builder.sext(self.builder.mul(self.builder.load(array_size_member),
                               self.builder.load(item_size_member)), self.intp_t), total_size)

            # Handle data
            kernel_arg = cgutils.alloca_once(self.builder, self.void_ptr_t,
                    size=self.one, name="kernel_arg" + str(self.cur_arg))
            data_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 4)])

            buffer_name = "buffer_ptr" + str(self.cur_arg)
            buffer_ptr = cgutils.alloca_once(self.builder, self.void_ptr_t,
                                             size=self.one, name=buffer_name)

            # env, buffer_size, buffer_ptr
            args = [self.builder.inttoptr(self.gpu_device_int_const, self.void_ptr_t),
                    self.builder.load(total_size),
                    buffer_ptr]
            self.builder.call(self.create_dppy_rw_mem_buffer, args)

            if index < self.num_inputs:
                args = [self.builder.inttoptr(self.gpu_device_int_const, self.void_ptr_t),
                        self.builder.load(buffer_ptr),
                        self.one,
                        self.zero,
                        self.builder.load(total_size),
                        self.builder.bitcast(self.builder.load(data_member), self.void_ptr_t)]

                self.builder.call(self.write_mem_buffer_to_device, args)
            else:
                self.read_bufs_after_enqueue.append((buffer_ptr, total_size, data_member))

            self.builder.call(self.create_dppy_kernel_arg_from_buffer, [buffer_ptr, kernel_arg])
            dst = self.builder.gep(self.kernel_arg_array, [self.context.get_constant(types.intp, self.cur_arg)])
            self.cur_arg += 1
            self.builder.store(self.builder.load(kernel_arg), dst)

            # Handle shape
            shape_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0),
                     self.context.get_constant(types.int32, 5)])

            for this_dim in range(arg_type.ndim):
                shape_entry = self.builder.gep(shape_member,
                                [self.context.get_constant(types.int32, 0),
                                 self.context.get_constant(types.int32, this_dim)])

                args = [self.builder.bitcast(shape_entry, self.void_ptr_ptr_t),
                        self.context.get_constant(types.uintp, self.sizeof_intp)]
                self._call_dppy_kernel_arg_fn(args)

            # Handle strides
            stride_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0),
                     self.context.get_constant(types.int32, 6)])

            for this_stride in range(arg_type.ndim):
                stride_entry = self.builder.gep(stride_member,
                                [self.context.get_constant(types.int32, 0),
                                 self.context.get_constant(types.int32, this_dim)])

                args = [self.builder.bitcast(stride_entry, self.void_ptr_ptr_t),
                        self.context.get_constant(types.uintp, self.sizeof_intp)]
                self._call_dppy_kernel_arg_fn(args)

        else:
            args = [self.builder.bitcast(llvm_arg, self.void_ptr_ptr_t),
                    self.context.get_constant(types.uintp, self.context.get_abi_sizeof(val_type))]
            self._call_dppy_kernel_arg_fn(args)

    def enqueue_kernel_and_read_back(self, loop_ranges):
        num_dim = len(loop_ranges)
        # the assumption is loop_ranges will always be less than or equal to 3 dimensions

        # Package dim start and stops for auto-blocking enqueue.
        dim_starts = cgutils.alloca_once(
                        self.builder, self.uintp_t,
                        size=self.context.get_constant(types.uintp, num_dim), name="dims")

        dim_stops = cgutils.alloca_once(
                        self.builder, self.uintp_t,
                        size=self.context.get_constant(types.uintp, num_dim), name="dims")

        for i in range(num_dim):
            start, stop, step = loop_ranges[i]
            if start.type != self.one_type:
                start = self.builder.sext(start, self.one_type)
            if stop.type != self.one_type:
                stop = self.builder.sext(stop, self.one_type)
            if step.type != self.one_type:
                step = self.builder.sext(step, self.one_type)

            # substract 1 because do-scheduling takes inclusive ranges
            stop = self.builder.sub(stop, self.one)

            self.builder.store(start,
                               self.builder.gep(dim_starts, [self.context.get_constant(types.uintp, i)]))
            self.builder.store(stop,
                               self.builder.gep(dim_stops, [self.context.get_constant(types.uintp, i)]))

        args = [self.builder.inttoptr(self.gpu_device_int_const, self.void_ptr_t),
                self.builder.inttoptr(self.context.get_constant(types.uintp, self.kernel_int), self.void_ptr_t),
                self.context.get_constant(types.uintp, self.total_kernel_args),
                self.kernel_arg_array,
                self.context.get_constant(types.uintp, num_dim),
                dim_starts,
                dim_stops]

        self.builder.call(self.enqueue_kernel, args)

        # read buffers back to host
        for read_buf in self.read_bufs_after_enqueue:
            buffer_ptr, array_size_member, data_member = read_buf
            args = [self.builder.inttoptr(self.gpu_device_int_const, self.void_ptr_t),
                    self.builder.load(buffer_ptr),
                    self.one,
                    self.zero,
                    self.builder.load(array_size_member),
                    self.builder.bitcast(self.builder.load(data_member), self.void_ptr_t)]
            self.builder.call(self.read_mem_buffer_from_device, args)


def dppy_dufunc_inliner(func_ir, calltypes, typemap, typingctx):
    _DEBUG = False
    modified = False
    work_list = list(func_ir.blocks.items())
    # use a work list, look for call sites via `ir.Expr.op == call` and
    # then pass these to `self._do_work` to make decisions about inlining.
    while work_list:
        label, block = work_list.pop()
        for i, instr in enumerate(block.body):

            #if isinstance(instr, Parfor):
                # work through the loop body
                #for (l, b) in instr.loop_body.items():
                    #for j, inst in enumerate(b.body):
            if isinstance(instr, ir.Assign):
                expr = instr.value
                if isinstance(expr, ir.Expr):
                    if expr.op == 'call':
                        find_assn = block.find_variable_assignment(expr.func.name).value
                        if isinstance(find_assn, ir.Global):
                            # because of circular import, find better solution
                            if (find_assn.value.__class__.__name__ == "DUFunc"):
                                py_func = find_assn.value.py_func
                                workfn = _do_work_call(func_ir, work_list,
                                                       block, i, expr, py_func, typemap, calltypes, typingctx)

                                print("Found call ", str(expr))
                    else:
                        continue

                    #if guard(workfn, state, work_list, b, j, expr):
                    if workfn:
                        modified = True
                        break  # because block structure changed


    if _DEBUG:
        print('after vectorize inline'.center(80, '-'))
        print(func_ir.dump())
        print(''.center(80, '-'))

    if modified:
        # clean up blocks
        dead_code_elimination(func_ir,
                              typemap=typemap)
        # clean up unconditional branches that appear due to inlined
        # functions introducing blocks
        func_ir.blocks = simplify_CFG(func_ir.blocks)

    if _DEBUG:
        print('after vectorize inline DCE'.center(80, '-'))
        print(func_ir.dump())
        print(''.center(80, '-'))

    return True

def _do_work_call(func_ir, work_list, block, i, expr, py_func, typemap, calltypes, typingctx):
    # try and get a definition for the call, this isn't always possible as
    # it might be a eval(str)/part generated awaiting update etc. (parfors)
    to_inline = None
    try:
        to_inline = func_ir.get_definition(expr.func)
    except Exception:
        return False

    # do not handle closure inlining here, another pass deals with that.
    if getattr(to_inline, 'op', False) == 'make_function':
        return False

    # check this is a known and typed function
    try:
        func_ty = typemap[expr.func.name]
    except KeyError:
        # e.g. Calls to CUDA Intrinsic have no mapped type so KeyError
        return False
    if not hasattr(func_ty, 'get_call_type'):
        return False

    sig = calltypes[expr]
    is_method = False

    templates = getattr(func_ty, 'templates', None)
    arg_typs = sig.args

    if templates is None:
        return False

    assert(len(templates) == 1)

    # at this point we know we maybe want to inline something and there's
    # definitely something that could be inlined.
    return _run_inliner(
        func_ir, sig, templates[0], arg_typs, expr, i, py_func, block,
        work_list, typemap, calltypes, typingctx
    )

def _run_inliner(
    func_ir, sig, template, arg_typs, expr, i, py_func, block,
    work_list, typemap, calltypes, typingctx
):
    from numba.inline_closurecall import (inline_closure_call,
                                          callee_ir_validator)

    # pass is typed so use the callee globals
    inline_closure_call(func_ir, py_func.__globals__,
                        block, i, py_func, typingctx=typingctx,
                        arg_typs=arg_typs,
                        typemap=typemap,
                        calltypes=calltypes,
                        work_list=work_list,
                        replace_freevars=False,
                        callee_validator=callee_ir_validator)
    return True