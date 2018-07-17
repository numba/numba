#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#
import numpy

import types as pytypes
import collections

from llvmlite import ir as lir

import numba
from numba.six import exec_
from numba import ir, types, typing, config, analysis, utils, cgutils, rewrites
from numba.typing.templates import signature, infer_global, AbstractTemplate
from numba.targets.imputils import impl_ret_untracked
from numba.analysis import (compute_live_map, compute_use_defs,
                            compute_cfg_from_blocks)
from numba.errors import TypingError
import copy

_unique_var_count = 0


def mk_unique_var(prefix):
    global _unique_var_count
    var = prefix + "." + str(_unique_var_count)
    _unique_var_count = _unique_var_count + 1
    return var


_max_label = 0


def get_unused_var_name(prefix, var_table):
    """ Get a new var name with a given prefix and
        make sure it is unused in the given variable table.
    """
    cur = 0
    while True:
        var = prefix + str(cur)
        if var not in var_table:
            return var
        cur += 1


def next_label():
    global _max_label
    _max_label += 1
    return _max_label


def mk_alloc(typemap, calltypes, lhs, size_var, dtype, scope, loc):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    """
    out = []
    ndims = 1
    size_typ = types.intp
    if isinstance(size_var, tuple):
        if len(size_var) == 1:
            size_var = size_var[0]
            size_var = convert_size_to_var(size_var, typemap, scope, loc, out)
        else:
            # tuple_var = build_tuple([size_var...])
            ndims = len(size_var)
            tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
            if typemap:
                typemap[tuple_var.name] = types.containers.UniTuple(
                    types.intp, ndims)
            # constant sizes need to be assigned to vars
            new_sizes = [convert_size_to_var(s, typemap, scope, loc, out)
                         for s in size_var]
            tuple_call = ir.Expr.build_tuple(new_sizes, loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            out.append(tuple_assign)
            size_var = tuple_var
            size_typ = types.containers.UniTuple(types.intp, ndims)
    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    if typemap:
        typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: empty_attr = getattr(g_np_var, empty)
    empty_attr_call = ir.Expr.getattr(g_np_var, "empty", loc)
    attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
    if typemap:
        typemap[attr_var.name] = get_np_ufunc_typ(numpy.empty)
    attr_assign = ir.Assign(empty_attr_call, attr_var, loc)
    # alloc call: lhs = empty_attr(size_var, typ_var)
    typ_var = ir.Var(scope, mk_unique_var("$np_typ_var"), loc)
    if typemap:
        typemap[typ_var.name] = types.functions.NumberClass(dtype)
    # assuming str(dtype) returns valid np dtype string
    dtype_str = str(dtype)
    if dtype_str=='bool':
        # empty doesn't like 'bool' sometimes (e.g. kmeans example)
        dtype_str = 'bool_'
    np_typ_getattr = ir.Expr.getattr(g_np_var, dtype_str, loc)
    typ_var_assign = ir.Assign(np_typ_getattr, typ_var, loc)
    alloc_call = ir.Expr.call(attr_var, [size_var, typ_var], (), loc)
    if calltypes:
        calltypes[alloc_call] = typemap[attr_var.name].get_call_type(
            typing.Context(), [size_typ, types.functions.NumberClass(dtype)], {})
    # signature(
    #    types.npytypes.Array(dtype, ndims, 'C'), size_typ,
    #    types.functions.NumberClass(dtype))
    alloc_assign = ir.Assign(alloc_call, lhs, loc)

    out.extend([g_np_assign, attr_assign, typ_var_assign, alloc_assign])
    return out


def convert_size_to_var(size_var, typemap, scope, loc, nodes):
    if isinstance(size_var, int):
        new_size = ir.Var(scope, mk_unique_var("$alloc_size"), loc)
        if typemap:
            typemap[new_size.name] = types.intp
        size_assign = ir.Assign(ir.Const(size_var, loc), new_size, loc)
        nodes.append(size_assign)
        return new_size
    assert isinstance(size_var, ir.Var)
    return size_var


def get_np_ufunc_typ(func):
    """get type of the incoming function from builtin registry"""
    for (k, v) in typing.npydecl.registry.globals:
        if k == func:
            return v
    raise RuntimeError("type for func ", func, " not found")


def mk_range_block(typemap, start, stop, step, calltypes, scope, loc):
    """make a block that initializes loop range and iteration variables.
    target label in jump needs to be set.
    """
    # g_range_var = Global(range)
    g_range_var = ir.Var(scope, mk_unique_var("$range_g_var"), loc)
    typemap[g_range_var.name] = get_global_func_typ(range)
    g_range = ir.Global('range', range, loc)
    g_range_assign = ir.Assign(g_range, g_range_var, loc)
    arg_nodes, args = _mk_range_args(typemap, start, stop, step, scope, loc)
    # range_call_var = call g_range_var(start, stop, step)
    range_call = ir.Expr.call(g_range_var, args, (), loc)
    calltypes[range_call] = typemap[g_range_var.name].get_call_type(
        typing.Context(), [types.intp] * len(args), {})
    #signature(types.range_state64_type, types.intp)
    range_call_var = ir.Var(scope, mk_unique_var("$range_c_var"), loc)
    typemap[range_call_var.name] = types.iterators.RangeType(types.intp)
    range_call_assign = ir.Assign(range_call, range_call_var, loc)
    # iter_var = getiter(range_call_var)
    iter_call = ir.Expr.getiter(range_call_var, loc)
    calltypes[iter_call] = signature(types.range_iter64_type,
                                     types.range_state64_type)
    iter_var = ir.Var(scope, mk_unique_var("$iter_var"), loc)
    typemap[iter_var.name] = types.iterators.RangeIteratorType(types.intp)
    iter_call_assign = ir.Assign(iter_call, iter_var, loc)
    # $phi = iter_var
    phi_var = ir.Var(scope, mk_unique_var("$phi"), loc)
    typemap[phi_var.name] = types.iterators.RangeIteratorType(types.intp)
    phi_assign = ir.Assign(iter_var, phi_var, loc)
    # jump to header
    jump_header = ir.Jump(-1, loc)
    range_block = ir.Block(scope, loc)
    range_block.body = arg_nodes + [g_range_assign, range_call_assign,
                                    iter_call_assign, phi_assign, jump_header]
    return range_block


def _mk_range_args(typemap, start, stop, step, scope, loc):
    nodes = []
    if isinstance(stop, ir.Var):
        g_stop_var = stop
    else:
        assert isinstance(stop, int)
        g_stop_var = ir.Var(scope, mk_unique_var("$range_stop"), loc)
        if typemap:
            typemap[g_stop_var.name] = types.intp
        stop_assign = ir.Assign(ir.Const(stop, loc), g_stop_var, loc)
        nodes.append(stop_assign)
    if start == 0 and step == 1:
        return nodes, [g_stop_var]

    if isinstance(start, ir.Var):
        g_start_var = start
    else:
        assert isinstance(start, int)
        g_start_var = ir.Var(scope, mk_unique_var("$range_start"), loc)
        if typemap:
            typemap[g_start_var.name] = types.intp
        start_assign = ir.Assign(ir.Const(start, loc), g_start_var, loc)
        nodes.append(start_assign)
    if step == 1:
        return nodes, [g_start_var, g_stop_var]

    if isinstance(step, ir.Var):
        g_step_var = step
    else:
        assert isinstance(step, int)
        g_step_var = ir.Var(scope, mk_unique_var("$range_step"), loc)
        if typemap:
            typemap[g_step_var.name] = types.intp
        step_assign = ir.Assign(ir.Const(step, loc), g_step_var, loc)
        nodes.append(step_assign)

    return nodes, [g_start_var, g_stop_var, g_step_var]


def get_global_func_typ(func):
    """get type variable for func() from builtin registry"""
    for (k, v) in typing.templates.builtin_registry.globals:
        if k == func:
            return v
    raise RuntimeError("func type not found {}".format(func))


def mk_loop_header(typemap, phi_var, calltypes, scope, loc):
    """make a block that is a loop header updating iteration variables.
    target labels in branch need to be set.
    """
    # iternext_var = iternext(phi_var)
    iternext_var = ir.Var(scope, mk_unique_var("$iternext_var"), loc)
    typemap[iternext_var.name] = types.containers.Pair(
        types.intp, types.boolean)
    iternext_call = ir.Expr.iternext(phi_var, loc)
    calltypes[iternext_call] = signature(
        types.containers.Pair(
            types.intp,
            types.boolean),
        types.range_iter64_type)
    iternext_assign = ir.Assign(iternext_call, iternext_var, loc)
    # pair_first_var = pair_first(iternext_var)
    pair_first_var = ir.Var(scope, mk_unique_var("$pair_first_var"), loc)
    typemap[pair_first_var.name] = types.intp
    pair_first_call = ir.Expr.pair_first(iternext_var, loc)
    pair_first_assign = ir.Assign(pair_first_call, pair_first_var, loc)
    # pair_second_var = pair_second(iternext_var)
    pair_second_var = ir.Var(scope, mk_unique_var("$pair_second_var"), loc)
    typemap[pair_second_var.name] = types.boolean
    pair_second_call = ir.Expr.pair_second(iternext_var, loc)
    pair_second_assign = ir.Assign(pair_second_call, pair_second_var, loc)
    # phi_b_var = pair_first_var
    phi_b_var = ir.Var(scope, mk_unique_var("$phi"), loc)
    typemap[phi_b_var.name] = types.intp
    phi_b_assign = ir.Assign(pair_first_var, phi_b_var, loc)
    # branch pair_second_var body_block out_block
    branch = ir.Branch(pair_second_var, -1, -1, loc)
    header_block = ir.Block(scope, loc)
    header_block.body = [iternext_assign, pair_first_assign,
                         pair_second_assign, phi_b_assign, branch]
    return header_block


def find_op_typ(op, arg_typs):
    for ft in typing.templates.builtin_registry.functions:
        if ft.key == op:
            try:
                func_typ = types.Function(ft).get_call_type(typing.Context(),
                                                            arg_typs, {})
            except TypingError:
                func_typ = None
            if func_typ is not None:
                return func_typ
    raise RuntimeError("unknown array operation")


def legalize_names(varnames):
    """returns a dictionary for conversion of variable names to legal
    parameter names.
    """
    var_map = {}
    for var in varnames:
        new_name = var.replace("_", "__").replace("$", "_").replace(".", "_")
        assert new_name not in var_map
        var_map[var] = new_name
    return var_map


def get_name_var_table(blocks):
    """create a mapping from variable names to their ir.Var objects"""
    def get_name_var_visit(var, namevar):
        namevar[var.name] = var
        return var
    namevar = {}
    visit_vars(blocks, get_name_var_visit, namevar)
    return namevar


def replace_var_names(blocks, namedict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> name)"""
    # remove identity values to avoid infinite loop
    new_namedict = {}
    for l, r in namedict.items():
        if l != r:
            new_namedict[l] = r

    def replace_name(var, namedict):
        assert isinstance(var, ir.Var)
        while var.name in namedict:
            var = ir.Var(var.scope, namedict[var.name], var.loc)
        return var
    visit_vars(blocks, replace_name, new_namedict)


def replace_var_callback(var, vardict):
    assert isinstance(var, ir.Var)
    while var.name in vardict.keys():
        new_var = vardict[var.name]
        var = ir.Var(new_var.scope, new_var.name, new_var.loc)
    return var


def replace_vars(blocks, vardict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> ir.Var)"""
    # remove identity values to avoid infinite loop
    new_vardict = {}
    for l, r in vardict.items():
        if l != r.name:
            new_vardict[l] = r
    visit_vars(blocks, replace_var_callback, new_vardict)


def replace_vars_stmt(stmt, vardict):
    visit_vars_stmt(stmt, replace_var_callback, vardict)


def replace_vars_inner(node, vardict):
    return visit_vars_inner(node, replace_var_callback, vardict)


# other packages that define new nodes add calls to visit variables in them
# format: {type:function}
visit_vars_extensions = {}


def visit_vars(blocks, callback, cbdata):
    """go over statements of block bodies and replace variable names with
    dictionary.
    """
    for block in blocks.values():
        for stmt in block.body:
            visit_vars_stmt(stmt, callback, cbdata)
    return


def visit_vars_stmt(stmt, callback, cbdata):
    # let external calls handle stmt if type matches
    for t, f in visit_vars_extensions.items():
        if isinstance(stmt, t):
            f(stmt, callback, cbdata)
            return
    if isinstance(stmt, ir.Assign):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Arg):
        stmt.name = visit_vars_inner(stmt.name, callback, cbdata)
    elif isinstance(stmt, ir.Return):
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Raise):
        stmt.exception = visit_vars_inner(stmt.exception, callback, cbdata)
    elif isinstance(stmt, ir.Branch):
        stmt.cond = visit_vars_inner(stmt.cond, callback, cbdata)
    elif isinstance(stmt, ir.Jump):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
    elif isinstance(stmt, ir.Del):
        # Because Del takes only a var name, we make up by
        # constructing a temporary variable.
        var = ir.Var(None, stmt.value, stmt.loc)
        var = visit_vars_inner(var, callback, cbdata)
        stmt.value = var.name
    elif isinstance(stmt, ir.DelAttr):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.attr = visit_vars_inner(stmt.attr, callback, cbdata)
    elif isinstance(stmt, ir.SetAttr):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.attr = visit_vars_inner(stmt.attr, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.DelItem):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.index = visit_vars_inner(stmt.index, callback, cbdata)
    elif isinstance(stmt, ir.StaticSetItem):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.index_var = visit_vars_inner(stmt.index_var, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.SetItem):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.index = visit_vars_inner(stmt.index, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Print):
        stmt.args = [visit_vars_inner(x, callback, cbdata) for x in stmt.args]
    else:
        # TODO: raise NotImplementedError("no replacement for IR node: ", stmt)
        pass
    return


def visit_vars_inner(node, callback, cbdata):
    if isinstance(node, ir.Var):
        return callback(node, cbdata)
    elif isinstance(node, list):
        return [visit_vars_inner(n, callback, cbdata) for n in node]
    elif isinstance(node, tuple):
        return tuple([visit_vars_inner(n, callback, cbdata) for n in node])
    elif isinstance(node, ir.Expr):
        # if node.op in ['binop', 'inplace_binop']:
        #     lhs = node.lhs.name
        #     rhs = node.rhs.name
        #     node.lhs.name = callback, cbdata.get(lhs, lhs)
        #     node.rhs.name = callback, cbdata.get(rhs, rhs)
        for arg in node._kws.keys():
            node._kws[arg] = visit_vars_inner(node._kws[arg], callback, cbdata)
    return node


add_offset_to_labels_extensions = {}


def add_offset_to_labels(blocks, offset):
    """add an offset to all block labels and jump/branch targets
    """
    new_blocks = {}
    for l, b in blocks.items():
        # some parfor last blocks might be empty
        term = None
        if b.body:
            term = b.body[-1]
            for inst in b.body:
                for T, f in add_offset_to_labels_extensions.items():
                    if isinstance(inst, T):
                        f_max = f(inst, offset)
        if isinstance(term, ir.Jump):
            b.body[-1] = ir.Jump(term.target + offset, term.loc)
        if isinstance(term, ir.Branch):
            b.body[-1] = ir.Branch(term.cond, term.truebr + offset,
                                   term.falsebr + offset, term.loc)
        new_blocks[l + offset] = b
    return new_blocks


find_max_label_extensions = {}


def find_max_label(blocks):
    max_label = 0
    for l, b in blocks.items():
        term = None
        if b.body:
            term = b.body[-1]
            for inst in b.body:
                for T, f in find_max_label_extensions.items():
                    if isinstance(inst, T):
                        f_max = f(inst)
                        if f_max > max_label:
                            max_label = f_max
        if l > max_label:
            max_label = l
    return max_label


def remove_dels(blocks):
    """remove ir.Del nodes"""
    for block in blocks.values():
        new_body = []
        for stmt in block.body:
            if not isinstance(stmt, ir.Del):
                new_body.append(stmt)
        block.body = new_body
    return


def remove_args(blocks):
    """remove ir.Arg nodes"""
    for block in blocks.values():
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                continue
            new_body.append(stmt)
        block.body = new_body
    return


def remove_dead(blocks, args, func_ir, typemap=None, alias_map=None, arg_aliases=None):
    """dead code elimination using liveness and CFG info.
    Returns True if something has been removed, or False if nothing is removed.
    """
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    call_table, _ = get_call_table(blocks)
    if alias_map is None or arg_aliases is None:
        alias_map, arg_aliases = find_potential_aliases(blocks, args, typemap,
                                                        func_ir)
    if config.DEBUG_ARRAY_OPT == 1:
        print("alias map:", alias_map)
    # keep set for easier search
    alias_set = set(alias_map.keys())

    removed = False
    for label, block in blocks.items():
        # find live variables at each statement to delete dead assignment
        lives = {v.name for v in block.terminator.list_vars()}
        # find live variables at the end of block
        for out_blk, _data in cfg.successors(label):
            lives |= live_map[out_blk]
        removed |= remove_dead_block(block, lives, call_table, arg_aliases,
                                     alias_map, alias_set, func_ir, typemap)
    return removed


# other packages that define new nodes add calls to remove dead code in them
# format: {type:function}
remove_dead_extensions = {}


def remove_dead_block(block, lives, call_table, arg_aliases, alias_map,
                                                  alias_set, func_ir, typemap):
    """remove dead code using liveness info.
    Mutable arguments (e.g. arrays) that are not definitely assigned are live
    after return of function.
    """
    # TODO: find mutable args that are not definitely assigned instead of
    # assuming all args are live after return
    removed = False

    # add statements in reverse order
    new_body = [block.terminator]
    # for each statement in reverse order, excluding terminator
    for stmt in reversed(block.body[:-1]):
        # aliases of lives are also live
        alias_lives = set()
        init_alias_lives = lives & alias_set
        for v in init_alias_lives:
            alias_lives |= alias_map[v]
        lives_n_aliases = lives | alias_lives | arg_aliases
        # let external calls handle stmt if type matches
        if type(stmt) in remove_dead_extensions:
            f = remove_dead_extensions[type(stmt)]
            stmt = f(stmt, lives, arg_aliases, alias_map, func_ir, typemap)
            if stmt is None:
                removed = True
                continue
        # ignore assignments that their lhs is not live or lhs==rhs
        if isinstance(stmt, ir.Assign):
            lhs = stmt.target
            rhs = stmt.value
            if lhs.name not in lives and has_no_side_effect(
                    rhs, lives_n_aliases, call_table):
                removed = True
                continue
            if isinstance(rhs, ir.Var) and lhs.name == rhs.name:
                removed = True
                continue
            # TODO: remove other nodes like SetItem etc.
        if isinstance(stmt, ir.SetItem):
            name = stmt.target.name
            if name not in lives_n_aliases:
                continue

        if type(stmt) in analysis.ir_extension_usedefs:
            def_func = analysis.ir_extension_usedefs[type(stmt)]
            uses, defs = def_func(stmt)
            lives -= defs
            lives |= uses
        else:
            lives |= {v.name for v in stmt.list_vars()}
            if isinstance(stmt, ir.Assign):
                lives.remove(lhs.name)

        new_body.append(stmt)
    new_body.reverse()
    block.body = new_body
    return removed

# list of functions
remove_call_handlers = []

def remove_dead_random_call(rhs, lives, call_list):
    if len(call_list) == 3 and call_list[1:] == ['random', numpy]:
        return call_list[0] != 'seed'
    return False

remove_call_handlers.append(remove_dead_random_call)

def has_no_side_effect(rhs, lives, call_table):
    """ Returns True if this expression has no side effects that
        would prevent re-ordering.
    """
    if isinstance(rhs, ir.Expr) and rhs.op == 'call':
        func_name = rhs.func.name
        if func_name not in call_table or call_table[func_name] == []:
            return False
        call_list = call_table[func_name]
        if (call_list == ['empty', numpy] or
            call_list == [slice] or
            call_list == ['stencil', numba] or
            call_list == ['log', numpy] or
            call_list == [numba.array_analysis.wrap_index]):
            return True
        elif (isinstance(call_list[0], numba.extending._Intrinsic) and
              (call_list[0]._name == 'empty_inferred' or
               call_list[0]._name == 'unsafe_empty_inferred')):
            return True
        from numba.targets.registry import CPUDispatcher
        from numba.targets.linalg import dot_3_mv_check_args
        if isinstance(call_list[0], CPUDispatcher):
            py_func = call_list[0].py_func
            if py_func == dot_3_mv_check_args:
                return True
        for f in remove_call_handlers:
            if f(rhs, lives, call_list):
                return True
        return False
    if isinstance(rhs, ir.Expr) and rhs.op == 'inplace_binop':
        return rhs.lhs.name not in lives
    if isinstance(rhs, ir.Yield):
        return False
    if isinstance(rhs, ir.Expr) and rhs.op == 'pair_first':
        # don't remove pair_first since prange looks for it
        return False
    return True

is_pure_extensions = []

def is_pure(rhs, lives, call_table):
    """ Returns True if every time this expression is evaluated it
        returns the same result.  This is not the case for things
        like calls to numpy.random.
    """
    if isinstance(rhs, ir.Expr) and rhs.op == 'call':
        func_name = rhs.func.name
        if func_name not in call_table or call_table[func_name] == []:
            return False
        call_list = call_table[func_name]
        if (call_list == [slice] or
            call_list == ['log', numpy] or
            call_list == ['empty', numpy]):
            return True
        for f in is_pure_extensions:
            if f(rhs, lives, call_list):
                return True
        return False
    if isinstance(rhs, ir.Yield):
        return False
    return True

alias_analysis_extensions = {}

def find_potential_aliases(blocks, args, typemap, func_ir, alias_map=None,
                                                            arg_aliases=None):
    "find all array aliases and argument aliases to avoid remove as dead"
    if alias_map is None:
        alias_map = {}
    if arg_aliases is None:
        arg_aliases = set(a for a in args if not is_immutable_type(a, typemap))

    # update definitions since they are not guaranteed to be up-to-date
    # FIXME keep definitions up-to-date to avoid the need for rebuilding
    func_ir._definitions = build_definitions(func_ir.blocks)
    np_alias_funcs = ['ravel', 'transpose', 'reshape']

    for bl in blocks.values():
        for instr in bl.body:
            if type(instr) in alias_analysis_extensions:
                f = alias_analysis_extensions[type(instr)]
                f(instr, args, typemap, func_ir, alias_map, arg_aliases)
            if isinstance(instr, ir.Assign):
                expr = instr.value
                lhs = instr.target.name
                # only mutable types can alias
                if is_immutable_type(lhs, typemap):
                    continue
                if isinstance(expr, ir.Var) and lhs!=expr.name:
                    _add_alias(lhs, expr.name, alias_map, arg_aliases)
                # subarrays like A = B[0] for 2D B
                if (isinstance(expr, ir.Expr) and (expr.op == 'cast' or
                    expr.op in ['getitem', 'static_getitem'])):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # array attributes like A.T
                if (isinstance(expr, ir.Expr) and expr.op == 'getattr'
                        and expr.attr in ['T', 'ctypes', 'flat']):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # calls that can create aliases such as B = A.ravel()
                if isinstance(expr, ir.Expr) and expr.op == 'call':
                    fdef = guard(find_callname, func_ir, expr, typemap)
                    # TODO: sometimes gufunc backend creates duplicate code
                    # causing find_callname to fail. Example: test_argmax
                    # ignored here since those cases don't create aliases
                    # but should be fixed in general
                    if fdef is None:
                        continue
                    fname, fmod = fdef
                    if fmod == 'numpy' and fname in np_alias_funcs:
                        _add_alias(lhs, expr.args[0].name, alias_map, arg_aliases)
                    if isinstance(fmod, ir.Var) and fname in np_alias_funcs:
                        _add_alias(lhs, fmod.name, alias_map, arg_aliases)

    # copy to avoid changing size during iteration
    old_alias_map = copy.deepcopy(alias_map)
    # combine all aliases transitively
    for v in old_alias_map:
        for w in old_alias_map[v]:
            alias_map[v] |= alias_map[w]
        for w in old_alias_map[v]:
            alias_map[w] = alias_map[v]

    return alias_map, arg_aliases

def _add_alias(lhs, rhs, alias_map, arg_aliases):
    if rhs in arg_aliases:
        arg_aliases.add(lhs)
    else:
        if rhs not in alias_map:
            alias_map[rhs] = set()
        if lhs not in alias_map:
            alias_map[lhs] = set()
        alias_map[rhs].add(lhs)
        alias_map[lhs].add(rhs)
    return

def is_immutable_type(var, typemap):
    # Conservatively, assume mutable if type not available
    if typemap is None or var not in typemap:
        return False
    typ = typemap[var]
    # TODO: add more immutable types
    if isinstance(typ, (types.Number, types.scalars._NPDatetimeBase,
                        types.containers.BaseTuple,
                        types.iterators.RangeType)):
        return True
    if typ==types.string:
        return True
    # consevatively, assume mutable
    return False

def copy_propagate(blocks, typemap):
    """compute copy propagation information for each block using fixed-point
     iteration on data flow equations:
     in_b = intersect(predec(B))
     out_b = gen_b | (in_b - kill_b)
    """
    cfg = compute_cfg_from_blocks(blocks)
    entry = cfg.entry_point()

    # format: dict of block labels to copies as tuples
    # label -> (l,r)
    c_data = init_copy_propagate_data(blocks, entry, typemap)
    (gen_copies, all_copies, kill_copies, in_copies, out_copies) = c_data

    old_point = None
    new_point = copy.deepcopy(out_copies)
    # comparison works since dictionary of built-in types
    while old_point != new_point:
        for label in blocks.keys():
            if label == entry:
                continue
            predecs = [i for i, _d in cfg.predecessors(label)]
            # in_b =  intersect(predec(B))
            in_copies[label] = out_copies[predecs[0]].copy()
            for p in predecs:
                in_copies[label] &= out_copies[p]

            # out_b = gen_b | (in_b - kill_b)
            out_copies[label] = (gen_copies[label]
                                 | (in_copies[label] - kill_copies[label]))
        old_point = new_point
        new_point = copy.deepcopy(out_copies)
    if config.DEBUG_ARRAY_OPT == 1:
        print("copy propagate out_copies:", out_copies)
    return in_copies, out_copies


def init_copy_propagate_data(blocks, entry, typemap):
    """get initial condition of copy propagation data flow for each block.
    """
    # gen is all definite copies, extra_kill is additional ones that may hit
    # for example, parfors can have control flow so they may hit extra copies
    gen_copies, extra_kill = get_block_copies(blocks, typemap)
    # set of all program copies
    all_copies = set()
    for l, s in gen_copies.items():
        all_copies |= gen_copies[l]
    kill_copies = {}
    for label, gen_set in gen_copies.items():
        kill_copies[label] = set()
        for lhs, rhs in all_copies:
            if lhs in extra_kill[label] or rhs in extra_kill[label]:
                kill_copies[label].add((lhs, rhs))
            # a copy is killed if it is not in this block and lhs or rhs are
            # assigned in this block
            assigned = {lhs for lhs, rhs in gen_set}
            if ((lhs, rhs) not in gen_set
                    and (lhs in assigned or rhs in assigned)):
                kill_copies[label].add((lhs, rhs))
    # set initial values
    # all copies are in for all blocks except entry
    in_copies = {l: all_copies.copy() for l in blocks.keys()}
    in_copies[entry] = set()
    out_copies = {}
    for label in blocks.keys():
        # out_b = gen_b | (in_b - kill_b)
        out_copies[label] = (gen_copies[label]
                             | (in_copies[label] - kill_copies[label]))
    out_copies[entry] = gen_copies[entry]
    return (gen_copies, all_copies, kill_copies, in_copies, out_copies)


# other packages that define new nodes add calls to get copies in them
# format: {type:function}
copy_propagate_extensions = {}


def get_block_copies(blocks, typemap):
    """get copies generated and killed by each block
    """
    block_copies = {}
    extra_kill = {}
    for label, block in blocks.items():
        assign_dict = {}
        extra_kill[label] = set()
        # assignments as dict to replace with latest value
        for stmt in block.body:
            for T, f in copy_propagate_extensions.items():
                if isinstance(stmt, T):
                    gen_set, kill_set = f(stmt, typemap)
                    for lhs, rhs in gen_set:
                        assign_dict[lhs] = rhs
                    # if a=b is in dict and b is killed, a is also killed
                    new_assign_dict = {}
                    for l, r in assign_dict.items():
                        if l not in kill_set and r not in kill_set:
                            new_assign_dict[l] = r
                        if r in kill_set:
                            extra_kill[label].add(l)
                    assign_dict = new_assign_dict
                    extra_kill[label] |= kill_set
            if isinstance(stmt, ir.Assign):
                lhs = stmt.target.name
                if isinstance(stmt.value, ir.Var):
                    rhs = stmt.value.name
                    # copy is valid only if same type (see
                    # TestCFunc.test_locals)
                    if typemap[lhs] == typemap[rhs]:
                        assign_dict[lhs] = rhs
                        continue
                if isinstance(stmt.value,
                              ir.Expr) and stmt.value.op == 'inplace_binop':
                    in1_var = stmt.value.lhs.name
                    in1_typ = typemap[in1_var]
                    # inplace_binop assigns first operand if mutable
                    if not (isinstance(in1_typ, types.Number)
                            or in1_typ == types.string):
                        extra_kill[label].add(in1_var)
                        # if a=b is in dict and b is killed, a is also killed
                        new_assign_dict = {}
                        for l, r in assign_dict.items():
                            if l != in1_var and r != in1_var:
                                new_assign_dict[l] = r
                            if r == in1_var:
                                extra_kill[label].add(l)
                        assign_dict = new_assign_dict
                extra_kill[label].add(lhs)
        block_cps = set(assign_dict.items())
        block_copies[label] = block_cps
    return block_copies, extra_kill


# other packages that define new nodes add calls to apply copy propagate in them
# format: {type:function}
apply_copy_propagate_extensions = {}


def apply_copy_propagate(blocks, in_copies, name_var_table, typemap, calltypes,
                         save_copies=None):
    """apply copy propagation to IR: replace variables when copies available"""
    # save_copies keeps an approximation of the copies that were applied, so
    # that the variable names of removed user variables can be recovered to some
    # extent.
    if save_copies is None:
        save_copies = []

    for label, block in blocks.items():
        var_dict = {l: name_var_table[r] for l, r in in_copies[label]}
        # assignments as dict to replace with latest value
        for stmt in block.body:
            if type(stmt) in apply_copy_propagate_extensions:
                f = apply_copy_propagate_extensions[type(stmt)]
                f(stmt, var_dict, name_var_table,
                    typemap, calltypes, save_copies)
            # only rhs of assignments should be replaced
            # e.g. if x=y is available, x in x=z shouldn't be replaced
            elif isinstance(stmt, ir.Assign):
                stmt.value = replace_vars_inner(stmt.value, var_dict)
            else:
                replace_vars_stmt(stmt, var_dict)
            fix_setitem_type(stmt, typemap, calltypes)
            for T, f in copy_propagate_extensions.items():
                if isinstance(stmt, T):
                    gen_set, kill_set = f(stmt, typemap)
                    for lhs, rhs in gen_set:
                        if rhs in name_var_table:
                            var_dict[lhs] = name_var_table[rhs]
                    for l, r in var_dict.copy().items():
                        if l in kill_set or r.name in kill_set:
                            var_dict.pop(l)
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                lhs = stmt.target.name
                rhs = stmt.value.name
                # rhs could be replaced with lhs from previous copies
                if lhs != rhs:
                    # copy is valid only if same type (see
                    # TestCFunc.test_locals)
                    if typemap[lhs] == typemap[rhs] and rhs in name_var_table:
                        var_dict[lhs] = name_var_table[rhs]
                    else:
                        var_dict.pop(lhs, None)
                    # a=b kills previous t=a
                    lhs_kill = []
                    for k, v in var_dict.items():
                        if v.name == lhs:
                            lhs_kill.append(k)
                    for k in lhs_kill:
                        var_dict.pop(k, None)
            if (isinstance(stmt, ir.Assign)
                                        and not isinstance(stmt.value, ir.Var)):
                lhs = stmt.target.name
                var_dict.pop(lhs, None)
                # previous t=a is killed if a is killed
                lhs_kill = []
                for k, v in var_dict.items():
                    if v.name == lhs:
                        lhs_kill.append(k)
                for k in lhs_kill:
                    var_dict.pop(k, None)
        save_copies.extend(var_dict.items())

    return save_copies

def fix_setitem_type(stmt, typemap, calltypes):
    """Copy propagation can replace setitem target variable, which can be array
    with 'A' layout. The replaced variable can be 'C' or 'F', so we update
    setitem call type reflect this (from matrix power test)
    """
    if not isinstance(stmt, (ir.SetItem, ir.StaticSetItem)):
        return
    t_typ = typemap[stmt.target.name]
    s_typ = calltypes[stmt].args[0]
    # test_optional t_typ can be Optional with array
    if not isinstance(
            s_typ,
            types.npytypes.Array) or not isinstance(
            t_typ,
            types.npytypes.Array):
        return
    if s_typ.layout == 'A' and t_typ.layout != 'A':
        new_s_typ = s_typ.copy(layout=t_typ.layout)
        calltypes[stmt].args = (
            new_s_typ,
            calltypes[stmt].args[1],
            calltypes[stmt].args[2])
    return


def dprint_func_ir(func_ir, title, blocks=None):
    """Debug print function IR, with an optional blocks argument
    that may differ from the IR's original blocks.
    """
    if config.DEBUG_ARRAY_OPT == 1:
        ir_blocks = func_ir.blocks
        func_ir.blocks = ir_blocks if blocks == None else blocks
        name = func_ir.func_id.func_qualname
        print(("IR %s: %s" % (title, name)).center(80, "-"))
        func_ir.dump()
        print("-" * 40)
        func_ir.blocks = ir_blocks


def find_topo_order(blocks, cfg = None):
    """find topological order of blocks such that true branches are visited
    first (e.g. for_break test in test_dataflow).
    """
    if cfg == None:
        cfg = compute_cfg_from_blocks(blocks)
    post_order = []
    seen = set()

    def _dfs_rec(node):
        if node not in seen:
            seen.add(node)
            succs = cfg._succs[node]
            last_inst = blocks[node].body[-1]
            if isinstance(last_inst, ir.Branch):
                succs = [last_inst.falsebr, last_inst.truebr]
            for dest in succs:
                if (node, dest) not in cfg._back_edges:
                    _dfs_rec(dest)
            post_order.append(node)

    _dfs_rec(cfg.entry_point())
    post_order.reverse()
    return post_order


# other packages that define new nodes add calls to get call table
# format: {type:function}
call_table_extensions = {}


def get_call_table(blocks, call_table=None, reverse_call_table=None):
    """returns a dictionary of call variables and their references.
    """
    # call_table example: c = np.zeros becomes c:["zeroes", np]
    # reverse_call_table example: c = np.zeros becomes np_var:c
    if call_table is None:
        call_table = {}
    if reverse_call_table is None:
        reverse_call_table = {}

    topo_order = find_topo_order(blocks)
    for label in reversed(topo_order):
        for inst in reversed(blocks[label].body):
            if isinstance(inst, ir.Assign):
                lhs = inst.target.name
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op == 'call':
                    call_table[rhs.func.name] = []
                if isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
                    if lhs in call_table:
                        call_table[lhs].append(rhs.attr)
                        reverse_call_table[rhs.value.name] = lhs
                    if lhs in reverse_call_table:
                        call_var = reverse_call_table[lhs]
                        call_table[call_var].append(rhs.attr)
                        reverse_call_table[rhs.value.name] = call_var
                if isinstance(rhs, ir.Global):
                    if lhs in call_table:
                        call_table[lhs].append(rhs.value)
                    if lhs in reverse_call_table:
                        call_var = reverse_call_table[lhs]
                        call_table[call_var].append(rhs.value)
                if isinstance(rhs, ir.FreeVar):
                    if lhs in call_table:
                        call_table[lhs].append(rhs.value)
                    if lhs in reverse_call_table:
                        call_var = reverse_call_table[lhs]
                        call_table[call_var].append(rhs.value)
            for T, f in call_table_extensions.items():
                if isinstance(inst, T):
                    f(inst, call_table, reverse_call_table)
    return call_table, reverse_call_table


# other packages that define new nodes add calls to get tuple table
# format: {type:function}
tuple_table_extensions = {}


def get_tuple_table(blocks, tuple_table=None):
    """returns a dictionary of tuple variables and their values.
    """
    if tuple_table is None:
        tuple_table = {}

    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                lhs = inst.target.name
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op == 'build_tuple':
                    tuple_table[lhs] = rhs.items
                if isinstance(rhs, ir.Const) and isinstance(rhs.value, tuple):
                    tuple_table[lhs] = rhs.value
            for T, f in tuple_table_extensions.items():
                if isinstance(inst, T):
                    f(inst, tuple_table)
    return tuple_table


def get_stmt_writes(stmt):
    writes = set()
    if isinstance(stmt, (ir.Assign, ir.SetItem, ir.StaticSetItem)):
        writes.add(stmt.target.name)
    return writes


def rename_labels(blocks):
    """rename labels of function body blocks according to topological sort.
    The set of labels of these blocks will remain unchanged.
    """
    topo_order = find_topo_order(blocks)

    # make a block with return last if available (just for readability)
    return_label = -1
    for l, b in blocks.items():
        if isinstance(b.body[-1], ir.Return):
            return_label = l
    # some cases like generators can have no return blocks
    if return_label != -1:
        topo_order.remove(return_label)
        topo_order.append(return_label)

    label_map = {}
    all_labels = sorted(topo_order, reverse=True)
    for label in topo_order:
        label_map[label] = all_labels.pop()
    # update target labels in jumps/branches
    for b in blocks.values():
        term = b.terminator
        if isinstance(term, ir.Jump):
            term.target = label_map[term.target]
        if isinstance(term, ir.Branch):
            term.truebr = label_map[term.truebr]
            term.falsebr = label_map[term.falsebr]
    # update blocks dictionary keys
    new_blocks = {}
    for k, b in blocks.items():
        new_label = label_map[k]
        new_blocks[new_label] = b

    return new_blocks


def simplify_CFG(blocks):
    """transform chains of blocks that have no loop into a single block"""
    # first, inline single-branch-block to its predecessors
    cfg = compute_cfg_from_blocks(blocks)
    def find_single_branch(label):
        block = blocks[label]
        return len(block.body) == 1 and isinstance(block.body[0], ir.Branch)
    single_branch_blocks = list(filter(find_single_branch, blocks.keys()))
    marked_for_del = set()
    for label in single_branch_blocks:
        inst = blocks[label].body[0]
        predecessors = cfg.predecessors(label)
        delete_block = True
        for (p, q) in predecessors:
            block = blocks[p]
            if isinstance(block.body[-1], ir.Jump):
                block.body[-1] = copy.copy(inst)
            else:
                delete_block = False
        if delete_block:
            marked_for_del.add(label)
    # Delete marked labels
    for label in marked_for_del:
        del blocks[label]
    merge_adjacent_blocks(blocks)
    return rename_labels(blocks)


arr_math = ['min', 'max', 'sum', 'prod', 'mean', 'var', 'std',
            'cumsum', 'cumprod', 'argmin', 'argmax', 'argsort',
            'nonzero', 'ravel']


def canonicalize_array_math(func_ir, typemap, calltypes, typingctx):
    # save array arg to call
    # call_varname -> array
    blocks = func_ir.blocks
    saved_arr_arg = {}
    topo_order = find_topo_order(blocks)
    for label in topo_order:
        block = blocks[label]
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                lhs = stmt.target.name
                rhs = stmt.value
                # replace A.func with np.func, and save A in saved_arr_arg
                if (rhs.op == 'getattr' and rhs.attr in arr_math
                        and isinstance(
                            typemap[rhs.value.name], types.npytypes.Array)):
                    rhs = stmt.value
                    arr = rhs.value
                    saved_arr_arg[lhs] = arr
                    scope = arr.scope
                    loc = arr.loc
                    # g_np_var = Global(numpy)
                    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
                    typemap[g_np_var.name] = types.misc.Module(numpy)
                    g_np = ir.Global('np', numpy, loc)
                    g_np_assign = ir.Assign(g_np, g_np_var, loc)
                    rhs.value = g_np_var
                    new_body.append(g_np_assign)
                    func_ir._definitions[g_np_var.name] = [g_np]
                    # update func var type
                    func = getattr(numpy, rhs.attr)
                    func_typ = get_np_ufunc_typ(func)
                    typemap.pop(lhs)
                    typemap[lhs] = func_typ
                if rhs.op == 'call' and rhs.func.name in saved_arr_arg:
                    # add array as first arg
                    arr = saved_arr_arg[rhs.func.name]
                    rhs.args = [arr] + rhs.args
                    # update call type signature to include array arg
                    old_sig = calltypes.pop(rhs)
                    calltypes[rhs] = typemap[rhs.func.name].get_call_type(
                        typingctx, [typemap[arr.name]] + list(old_sig.args), {})

            new_body.append(stmt)
        block.body = new_body
    return


# format: {type:function}
array_accesses_extensions = {}


def get_array_accesses(blocks, accesses=None):
    """returns a set of arrays accessed and their indices.
    """
    if accesses is None:
        accesses = set()

    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.SetItem):
                accesses.add((inst.target.name, inst.index.name))
            if isinstance(inst, ir.StaticSetItem):
                accesses.add((inst.target.name, inst.index_var.name))
            if isinstance(inst, ir.Assign):
                lhs = inst.target.name
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op == 'getitem':
                    accesses.add((rhs.value.name, rhs.index.name))
                if isinstance(rhs, ir.Expr) and rhs.op == 'static_getitem':
                    index = rhs.index
                    # slice is unhashable, so just keep the variable
                    if index is None or is_slice_index(index):
                        index = rhs.index_var.name
                    accesses.add((rhs.value.name, index))
            for T, f in array_accesses_extensions.items():
                if isinstance(inst, T):
                    f(inst, accesses)
    return accesses

def is_slice_index(index):
    """see if index is a slice index or has slice in it"""
    if isinstance(index, slice):
        return True
    if isinstance(index, tuple):
        for i in index:
            if isinstance(i, slice):
                return True
    return False

def merge_adjacent_blocks(blocks):
    cfg = compute_cfg_from_blocks(blocks)
    # merge adjacent blocks
    removed = set()
    for label in list(blocks.keys()):
        if label in removed:
            continue
        block = blocks[label]
        succs = list(cfg.successors(label))
        while True:
            if len(succs) != 1:
                break
            next_label = succs[0][0]
            if next_label in removed:
                break
            preds = list(cfg.predecessors(next_label))
            succs = list(cfg.successors(next_label))
            if len(preds) != 1 or preds[0][0] != label:
                break
            next_block = blocks[next_label]
            # XXX: commented out since scope objects are not consistent
            # thoughout the compiler. for example, pieces of code are compiled
            # and inlined on the fly without proper scope merge.
            # if block.scope != next_block.scope:
            #     break
            # merge
            block.body.pop()  # remove Jump
            block.body += next_block.body
            del blocks[next_label]
            removed.add(next_label)
            label = next_label

def restore_copy_var_names(blocks, save_copies, typemap):
    """
    restores variable names of user variables after applying copy propagation
    """
    rename_dict = {}
    for (a, b) in save_copies:
        # a is string name, b is variable
        # if a is user variable and b is generated temporary and b is not
        # already renamed
        if (not a.startswith('$') and b.name.startswith('$')
                                                and b.name not in rename_dict):
            new_name = mk_unique_var('${}'.format(a));
            rename_dict[b.name] = new_name
            typ = typemap.pop(b.name)
            typemap[new_name] = typ

    replace_var_names(blocks, rename_dict)

def simplify(func_ir, typemap, calltypes):
    remove_dels(func_ir.blocks)
    # get copies in to blocks and out from blocks
    in_cps, out_cps = copy_propagate(func_ir.blocks, typemap)
    # table mapping variable names to ir.Var objects to help replacement
    name_var_table = get_name_var_table(func_ir.blocks)
    save_copies = apply_copy_propagate(
        func_ir.blocks,
        in_cps,
        name_var_table,
        typemap,
        calltypes)
    restore_copy_var_names(func_ir.blocks, save_copies, typemap)
    # remove dead code to enable fusion
    remove_dead(func_ir.blocks, func_ir.arg_names, func_ir, typemap)
    func_ir.blocks = simplify_CFG(func_ir.blocks)
    if config.DEBUG_ARRAY_OPT == 1:
        dprint_func_ir(func_ir, "after simplify")

class GuardException(Exception):
    pass

def require(cond):
    """
    Raise GuardException if the given condition is False.
    """
    if not cond:
       raise GuardException

def guard(func, *args, **kwargs):
    """
    Run a function with given set of arguments, and guard against
    any GuardException raised by the function by returning None,
    or the expected return results if no such exception was raised.
    """
    try:
        return func(*args, **kwargs)
    except GuardException:
        return None

def get_definition(func_ir, name, **kwargs):
    """
    Same as func_ir.get_definition(name), but raise GuardException if
    exception KeyError is caught.
    """
    try:
        return func_ir.get_definition(name, **kwargs)
    except KeyError:
        raise GuardException

def build_definitions(blocks, definitions=None):
    """Build the definitions table of the given blocks by scanning
    through all blocks and instructions, useful when the definitions
    table is out-of-sync.
    Will return a new definition table if one is not passed.
    """
    if definitions is None:
        definitions = collections.defaultdict(list)

    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                name = inst.target.name
                definition = definitions.get(name, [])
                if definition == []:
                    definitions[name] = definition
                definition.append(inst.value)
            if type(inst) in build_defs_extensions:
                f = build_defs_extensions[type(inst)]
                f(inst, definitions)

    return definitions

build_defs_extensions = {}

def find_callname(func_ir, expr, typemap=None, definition_finder=get_definition):
    """Check if a call expression is calling a numpy function, and
    return the callee's function name and module name (both are strings),
    or raise GuardException. For array attribute calls such as 'a.f(x)'
    when 'a' is a numpy array, the array variable 'a' is returned
    in place of the module name.
    """
    require(isinstance(expr, ir.Expr) and expr.op == 'call')
    callee = expr.func
    callee_def = definition_finder(func_ir, callee)
    attrs = []
    obj = None
    while True:
        if isinstance(callee_def, (ir.Global, ir.FreeVar)):
            # require(callee_def.value == numpy)
            # these checks support modules like numpy, numpy.random as well as
            # calls like len() and intrinsitcs like assertEquiv
            keys = ['name', '_name', '__name__']
            value = None
            for key in keys:
                if hasattr(callee_def.value, key):
                    value = getattr(callee_def.value, key)
                    break
            if not value:
                raise GuardException
            attrs.append(value)
            def_val = callee_def.value
            # get the underlying definition of Intrinsic object to be able to
            # find the module effectively.
            # Otherwise, it will return numba.extending
            if isinstance(def_val, numba.extending._Intrinsic):
                def_val = def_val._defn
            if hasattr(def_val, '__module__'):
                mod_name = def_val.__module__
                # it might be a numpy function imported directly
                if (hasattr(numpy, value)
                        and def_val == getattr(numpy, value)):
                    attrs += ['numpy']
                # it might be a np.random function imported directly
                elif (hasattr(numpy.random, value)
                        and def_val == getattr(numpy.random, value)):
                    attrs += ['random', 'numpy']
                elif mod_name is not None:
                    attrs.append(mod_name)
            else:
                class_name = def_val.__class__.__name__
                if class_name == 'builtin_function_or_method':
                    class_name = 'builtin'
                if class_name != 'module':
                    attrs.append(class_name)
            break
        elif isinstance(callee_def, ir.Expr) and callee_def.op == 'getattr':
            obj = callee_def.value
            attrs.append(callee_def.attr)
            if typemap and obj.name in typemap:
                typ = typemap[obj.name]
                if isinstance(typ, types.npytypes.Array):
                    return attrs[0], obj
            callee_def = definition_finder(func_ir, obj)
        else:
            # obj.func calls where obj is not np array
            if obj is not None:
                return '.'.join(reversed(attrs)), obj
            raise GuardException
    return attrs[0], '.'.join(reversed(attrs[1:]))

def find_build_sequence(func_ir, var):
    """Check if a variable is constructed via build_tuple or
    build_list or build_set, and return the sequence and the
    operator, or raise GuardException otherwise.
    Note: only build_tuple is immutable, so use with care.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    build_ops = ['build_tuple', 'build_list', 'build_set']
    require(var_def.op in build_ops)
    return var_def.items, var_def.op

def find_const(func_ir, var):
    """Check if a variable is defined as constant, and return
    the constant value, or raise GuardException otherwise.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Const))
    return var_def.value

def compile_to_numba_ir(mk_func, glbls, typingctx=None, arg_typs=None,
                        typemap=None, calltypes=None):
    """
    Compile a function or a make_function node to Numba IR.

    Rename variables and
    labels to avoid conflict if inlined somewhere else. Perform type inference
    if typingctx and other typing inputs are available and update typemap and
    calltypes.
    """
    from numba import compiler
    # mk_func can be actual function or make_function node
    if hasattr(mk_func, 'code'):
        code = mk_func.code
    elif hasattr(mk_func, '__code__'):
        code = mk_func.__code__
    else:
        raise NotImplementedError("function type not recognized {}".format(mk_func))
    f_ir = get_ir_of_code(glbls, code)
    remove_dels(f_ir.blocks)

    # relabel by adding an offset
    global _max_label
    f_ir.blocks = add_offset_to_labels(f_ir.blocks, _max_label + 1)
    max_label = max(f_ir.blocks.keys())
    _max_label = max_label

    # rename all variables to avoid conflict
    var_table = get_name_var_table(f_ir.blocks)
    new_var_dict = {}
    for name, var in var_table.items():
        new_var_dict[name] = mk_unique_var(name)
    replace_var_names(f_ir.blocks, new_var_dict)

    # perform type inference if typingctx is available and update type
    # data structures typemap and calltypes
    if typingctx:
        f_typemap, f_return_type, f_calltypes = compiler.type_inference_stage(
                typingctx, f_ir, arg_typs, None)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        typemap.update(f_typemap)
        calltypes.update(f_calltypes)
    return f_ir

def get_ir_of_code(glbls, fcode):
    """
    Compile a code object to get its IR.
    """
    nfree = len(fcode.co_freevars)
    func_env = "\n".join(["  c_%d = None" % i for i in range(nfree)])
    func_clo = ",".join(["c_%d" % i for i in range(nfree)])
    func_arg = ",".join(["x_%d" % i for i in range(fcode.co_argcount)])
    func_text = "def g():\n%s\n  def f(%s):\n    return (%s)\n  return f" % (
        func_env, func_arg, func_clo)
    loc = {}
    exec_(func_text, glbls, loc)

    # hack parameter name .0 for Python 3 versions < 3.6
    if utils.PYVERSION >= (3,) and utils.PYVERSION < (3, 6):
        co_varnames = list(fcode.co_varnames)
        if co_varnames[0] == ".0":
            co_varnames[0] = "implicit0"
        fcode = pytypes.CodeType(
            fcode.co_argcount,
            fcode.co_kwonlyargcount,
            fcode.co_nlocals,
            fcode.co_stacksize,
            fcode.co_flags,
            fcode.co_code,
            fcode.co_consts,
            fcode.co_names,
            tuple(co_varnames),
            fcode.co_filename,
            fcode.co_name,
            fcode.co_firstlineno,
            fcode.co_lnotab,
            fcode.co_freevars,
            fcode.co_cellvars)

    f = loc['g']()
    f.__code__ = fcode
    f.__name__ = fcode.co_name
    from numba import compiler
    ir = compiler.run_frontend(f)
    # we need to run the before inference rewrite pass to normalize the IR
    # XXX: check rewrite pass flag?
    # for example, Raise nodes need to become StaticRaise before type inference
    class DummyPipeline(object):
        def __init__(self, f_ir):
            self.typingctx = None
            self.targetctx = None
            self.args = None
            self.func_ir = f_ir
            self.typemap = None
            self.return_type = None
            self.calltypes = None
    rewrites.rewrite_registry.apply('before-inference',
                                    DummyPipeline(ir), ir)
    return ir

def replace_arg_nodes(block, args):
    """
    Replace ir.Arg(...) with variables
    """
    for stmt in block.body:
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
            idx = stmt.value.index
            assert(idx < len(args))
            stmt.value = args[idx]
    return

def replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for block in blocks.values():
        casts = []
        for i, stmt in enumerate(block.body):
            if isinstance(stmt, ir.Return):
                assert(i + 1 == len(block.body))
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))
                # remove cast of the returned value
                for cast in casts:
                    if cast.target.name == stmt.value.name:
                        cast.value = cast.value.value
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and stmt.value.op == 'cast':
                casts.append(stmt)

def gen_np_call(func_as_str, func, lhs, args, typingctx, typemap, calltypes):
    scope = args[0].scope
    loc = args[0].loc

    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: <something>_attr = getattr(g_np_var, func_as_str)
    np_attr_call = ir.Expr.getattr(g_np_var, func_as_str, loc)
    attr_var = ir.Var(scope, mk_unique_var("$np_attr_attr"), loc)
    func_var_typ = get_np_ufunc_typ(func)
    typemap[attr_var.name] = func_var_typ
    attr_assign = ir.Assign(np_attr_call, attr_var, loc)
    # np call: lhs = np_attr(*args)
    np_call = ir.Expr.call(attr_var, args, (), loc)
    arg_types = [typemap[x.name] for x in args]
    func_typ = func_var_typ.get_call_type(typingctx, arg_types, {})
    calltypes[np_call] = func_typ
    np_assign = ir.Assign(np_call, lhs, loc)
    return [g_np_assign, attr_assign, np_assign]

def dump_blocks(blocks):
    for label, block in blocks.items():
        print(label, ":")
        for stmt in block.body:
            print("    ", stmt)

def is_get_setitem(stmt):
    """stmt is getitem assignment or setitem (and static cases)"""
    return is_getitem(stmt) or is_setitem(stmt)


def is_getitem(stmt):
    """true if stmt is a getitem or static_getitem assignment"""
    return (isinstance(stmt, ir.Assign)
            and isinstance(stmt.value, ir.Expr)
            and stmt.value.op in ['getitem', 'static_getitem'])

def is_setitem(stmt):
    """true if stmt is a SetItem or StaticSetItem node"""
    return isinstance(stmt, (ir.SetItem, ir.StaticSetItem))

def index_var_of_get_setitem(stmt):
    """get index variable for getitem/setitem nodes (and static cases)"""
    if is_getitem(stmt):
        if stmt.value.op == 'getitem':
            return stmt.value.index
        else:
            return stmt.value.index_var

    if is_setitem(stmt):
        if isinstance(stmt, ir.SetItem):
            return stmt.index
        else:
            return stmt.index_var

    return None

def set_index_var_of_get_setitem(stmt, new_index):
    if is_getitem(stmt):
        if stmt.value.op == 'getitem':
            stmt.value.index = new_index
        else:
            stmt.value.index_var = new_index
    elif is_setitem(stmt):
        if isinstance(stmt, ir.SetItem):
            stmt.index = new_index
        else:
            stmt.index_var = new_index
    else:
        raise ValueError("getitem or setitem node expected but received {}".format(
                     stmt))

def is_namedtuple_class(c):
    """check if c is a namedtuple class"""
    if not isinstance(c, type):
        return False
    # should have only tuple as superclass
    bases = c.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    # should have _make method
    if not hasattr(c, '_make'):
        return False
    # should have _fields that is all string
    fields = getattr(c, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(f, str) for f in fields)
