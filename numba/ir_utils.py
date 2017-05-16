from numba import ir, types, typing, config, analysis
from numba.typing.templates import signature
import numpy
from numba.analysis import (compute_live_map, compute_use_defs,
                            compute_cfg_from_blocks)
import copy

_unique_var_count = 0
def mk_unique_var(prefix):
    global _unique_var_count
    var = prefix + "." + str(_unique_var_count)
    _unique_var_count = _unique_var_count + 1
    return var

_max_label = 0
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
        else:
            # tuple_var = build_tuple([size_var...])
            ndims = len(size_var)
            tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
            if typemap:
                typemap[tuple_var.name] = types.containers.UniTuple(types.intp, ndims)
            # constant sizes need to be assigned to vars
            new_sizes = []
            for size in size_var:
                if isinstance(size, ir.Var):
                    new_size = size
                else:
                    assert isinstance(size, int)
                    new_size = ir.Var(scope, mk_unique_var("$alloc_size"), loc)
                    if typemap:
                        typemap[new_size.name] = types.intp
                    size_assign = ir.Assign(ir.Const(size, loc), new_size, loc)
                    out.append(size_assign)
                new_sizes.append(new_size)
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
    np_typ_getattr = ir.Expr.getattr(g_np_var, str(dtype), loc)
    typ_var_assign = ir.Assign(np_typ_getattr, typ_var, loc)
    alloc_call = ir.Expr.call(attr_var, [size_var, typ_var], (), loc)
    if calltypes:
        calltypes[alloc_call] = typemap[attr_var.name].get_call_type(
            typing.Context(), [size_typ, types.functions.NumberClass(dtype)], {})
    #signature(
    #    types.npytypes.Array(dtype, ndims, 'C'), size_typ,
    #    types.functions.NumberClass(dtype))
    alloc_assign = ir.Assign(alloc_call, lhs, loc)

    out.extend([g_np_assign, attr_assign, typ_var_assign, alloc_assign])
    return out

def get_np_ufunc_typ(func):
    """get type of the incoming function from builtin registry"""
    for (k,v) in typing.npydecl.registry.globals:
        if k==func:
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
        typing.Context(), [types.intp]*len(args), {})
    #signature(types.range_state64_type, types.intp)
    range_call_var = ir.Var(scope, mk_unique_var("$range_c_var"), loc)
    typemap[range_call_var.name] = types.iterators.RangeType(types.intp)
    range_call_assign = ir.Assign(range_call, range_call_var, loc)
    # iter_var = getiter(range_call_var)
    iter_call = ir.Expr.getiter(range_call_var ,loc)
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
    if start==0 and step==1:
        return nodes, [g_stop_var]

    if isinstance(start, ir.Var):
        g_start_var = start
    else:
        assert isinstance(start, int)
        g_start_var = ir.Var(scope, mk_unique_var("$range_start"), loc)
        if typemap:
            typemap[g_start_var.name] = types.intp
        start_assign = ir.Assign(ir.Const(start, loc), g_start_var)
        nodes.append(start_assign)
    if step==1:
        return nodes, [g_start_var, g_stop_var]

    if isinstance(step, ir.Var):
        g_step_var = step
    else:
        assert isinstance(step, int)
        g_step_var = ir.Var(scope, mk_unique_var("$range_step"), loc)
        if typemap:
            typemap[g_step_var.name] = types.intp
        step_assign = ir.Assign(ir.Const(step, loc), g_step_var)
        nodes.append(step_assign)

    return nodes, [g_start_var, g_stop_var, g_step_var]

def get_global_func_typ(func):
    """get type variable for func() from builtin registry"""
    for (k,v) in typing.templates.builtin_registry.globals:
        if k==func:
            return v
    raise RuntimeError("func type not found {}".format(func))

def mk_loop_header(typemap, phi_var, calltypes, scope, loc):
    """make a block that is a loop header updating iteration variables.
    target labels in branch need to be set.
    """
    # iternext_var = iternext(phi_var)
    iternext_var = ir.Var(scope, mk_unique_var("$iternext_var"), loc)
    typemap[iternext_var.name] = types.containers.Pair(types.intp, types.boolean)
    iternext_call = ir.Expr.iternext(phi_var, loc)
    calltypes[iternext_call] = signature(
        types.containers.Pair(types.intp, types.boolean), types.range_iter64_type)
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
        if ft.key==op:
            func_typ = types.Function(ft).get_call_type(typing.Context(),
                arg_typs, {})
            if func_typ is not None:
                return func_typ
    raise RuntimeError("unknown array operation")

def legalize_names(varnames):
    """returns a dictionary for conversion of variable names to legal
    parameter names.
    """
    var_map = {}
    for var in varnames:
        new_name = var.replace("_","__").replace("$", "_").replace(".", "_")
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
    for l,r in namedict.items():
        if l!=r:
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
    for l,r in vardict.items():
        if l!=r.name:
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
    for t,f in visit_vars_extensions.items():
        if isinstance(stmt,t):
            f(stmt, callback, cbdata)
            return
    if isinstance(stmt, ir.Assign):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Arg):
        stmt.name = visit_vars_inner(stmt.name, callback, cbdata)
    elif isinstance(stmt, ir.Return):
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
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
    else:
        pass # TODO: raise NotImplementedError("no replacement for IR node: ", stmt)
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

def add_offset_to_labels(blocks, offset):
    """add an offset to all block labels and jump/branch targets
    """
    new_blocks = {}
    for l,b in blocks.items():
        term = b.body[-1]
        if isinstance(term, ir.Jump):
            term.target += offset
        if isinstance(term, ir.Branch):
            term.truebr += offset
            term.falsebr += offset
        new_blocks[l+offset] = b
    return new_blocks

def remove_dels(blocks):
    """remove ir.Del nodes"""
    for block in blocks.values():
        new_body = []
        for stmt in block.body:
            if not isinstance(stmt, ir.Del):
                new_body.append(stmt)
        block.body = new_body
    return

def remove_dead(blocks, args):
    """dead code elimination using liveness and CFG info.
    Returns True if something has been removed, or False if nothing is removed."""
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    arg_aliases = find_potential_aliases(blocks, args)
    call_table,_ = get_call_table(blocks)

    removed = False
    for label, block in blocks.items():
        # find live variables at each statement to delete dead assignment
        lives = { v.name for v in block.terminator.list_vars() }
        # find live variables at the end of block
        for out_blk, _data in cfg.successors(label):
            lives |= live_map[out_blk]
        if label in cfg.exit_points():
            lives |= arg_aliases
        removed |= remove_dead_block(block, lives, call_table, arg_aliases)
    return removed

# other packages that define new nodes add calls to remove dead code in them
# format: {type:function}
remove_dead_extensions = {}

def remove_dead_block(block, lives, call_table, args):
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
        # let external calls handle stmt if type matches
        for t,f in remove_dead_extensions.items():
            if isinstance(stmt,t):
                f(stmt, lives, args)
        # ignore assignments that their lhs is not live or lhs==rhs
        if isinstance(stmt, ir.Assign):
            lhs = stmt.target
            rhs = stmt.value
            if lhs.name not in lives and has_no_side_effect(rhs, lives, call_table):
                removed = True
                continue
            if isinstance(rhs, ir.Var) and lhs.name==rhs.name:
                removed = True
                continue
            # TODO: remove other nodes like SetItem etc.
        if isinstance(stmt, ir.SetItem):
            if stmt.target.name not in lives:
                continue

        lives |= { v.name for v in stmt.list_vars() }
        if isinstance(stmt, ir.Assign):
            lives.remove(lhs.name)
        for T, def_func in analysis.ir_extension_defs.items():
            if isinstance(stmt, T):
                lives -= def_func(stmt)
        new_body.append(stmt)
    new_body.reverse()
    block.body = new_body
    return removed

def has_no_side_effect(rhs, lives, call_table):
    # TODO: find side-effect free calls like Numpy calls
    if isinstance(rhs, ir.Expr) and rhs.op=='call':
        func_name = rhs.func.name
        if func_name not in call_table:
            return False
        call_list = call_table[func_name]
        if call_list==['empty', numpy] or call_list==[slice]:
            return True
        return False
    if isinstance(rhs, ir.Expr) and rhs.op=='inplace_binop':
        return rhs.lhs.name not in lives
    if isinstance(rhs, ir.Yield):
        return False
    return True

def find_potential_aliases(blocks, args):
    aliases = set(args)
    for bl in blocks.values():
        for instr in bl.body:
            if isinstance(instr, ir.Assign):
                expr = instr.value
                lhs = instr.target.name
                if isinstance(expr, ir.Var) and expr.name in aliases:
                    aliases.add(lhs)
    return aliases

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
    while old_point!=new_point:
        for label in blocks.keys():
            if label==entry:
                continue
            predecs = [i for i,_d in cfg.predecessors(label)]
            # in_b =  intersect(predec(B))
            in_copies[label] = out_copies[predecs[0]].copy()
            for p in predecs:
                in_copies[label] &= out_copies[p]

            # out_b = gen_b | (in_b - kill_b)
            out_copies[label] = (gen_copies[label]
                | (in_copies[label] - kill_copies[label]))
        old_point = new_point
        new_point = copy.deepcopy(out_copies)
    if config.DEBUG_ARRAY_OPT==1:
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
    for l,s in gen_copies.items():
        all_copies |= gen_copies[l]
    kill_copies = {}
    for label, gen_set in gen_copies.items():
        kill_copies[label] = set()
        for lhs,rhs in all_copies:
            if lhs in extra_kill[label] or rhs in extra_kill[label]:
                kill_copies[label].add((lhs,rhs))
            # a copy is killed if it is not in this block and lhs or rhs are
            # assigned in this block
            assigned = { lhs for lhs,rhs in gen_set }
            if ((lhs,rhs) not in gen_set
                    and (lhs in assigned or rhs in assigned)):
                kill_copies[label].add((lhs,rhs))
    # set initial values
    # all copies are in for all blocks except entry
    in_copies = { l:all_copies.copy() for l in blocks.keys() }
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
            for T,f in copy_propagate_extensions.items():
                if isinstance(stmt,T):
                    gen_set, kill_set = f(stmt, typemap)
                    for lhs,rhs in gen_set:
                        assign_dict[lhs] = rhs
                    extra_kill[label] |= kill_set
            if isinstance(stmt, ir.Assign):
                lhs = stmt.target.name
                if isinstance(stmt.value, ir.Var):
                    rhs = stmt.value.name
                    # copy is valid only if same type (see TestCFunc.test_locals)
                    if typemap[lhs]==typemap[rhs]:
                        assign_dict[lhs] = rhs
                        continue
                extra_kill[label].add(lhs)
        block_copies[label] = set(assign_dict.items())
    return block_copies, extra_kill

# other packages that define new nodes add calls to apply copy propagate in them
# format: {type:function}
apply_copy_propagate_extensions = {}

def apply_copy_propagate(blocks, in_copies, name_var_table, ext_func, ext_data,
        typemap, calltypes):
    """apply copy propagation to IR: replace variables when copies available"""
    for label, block in blocks.items():
        var_dict = {l:name_var_table[r] for l,r in in_copies[label]}
        # assignments as dict to replace with latest value
        for stmt in block.body:
            ext_func(stmt, var_dict, ext_data)
            for T,f in apply_copy_propagate_extensions.items():
                if isinstance(stmt,T):
                    f(stmt, var_dict, name_var_table, ext_func, ext_data,
                        typemap, calltypes)
            # only rhs of assignments should be replaced
            # e.g. if x=y is available, x in x=z shouldn't be replaced
            if isinstance(stmt, ir.Assign):
                stmt.value = replace_vars_inner(stmt.value, var_dict)
            else:
                replace_vars_stmt(stmt, var_dict)
            fix_setitem_type(stmt, typemap, calltypes)
            for T,f in copy_propagate_extensions.items():
                if isinstance(stmt,T):
                    gen_set, kill_set = f(stmt, typemap)
                    for lhs,rhs in gen_set:
                        var_dict[lhs] = name_var_table[rhs]
                    for l,r in var_dict.copy().items():
                        if l in kill_set or r.name in kill_set:
                            var_dict.pop(l)
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                lhs = stmt.target.name
                rhs = stmt.value.name
                # rhs could be replaced with lhs from previous copies
                if lhs!=rhs:
                    # copy is valid only if same type (see TestCFunc.test_locals)
                    if typemap[lhs]==typemap[rhs]:
                        var_dict[lhs] = name_var_table[rhs]
                    else:
                        var_dict.pop(lhs, None)
                    # a=b kills previous t=a
                    lhs_kill = []
                    for k,v in var_dict.items():
                        if v.name==lhs:
                            lhs_kill.append(k)
                    for k in lhs_kill:
                        var_dict.pop(k, None)
    return

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
    if not isinstance(s_typ, types.npytypes.Array) or not isinstance(t_typ, types.npytypes.Array):
        return
    if s_typ.layout=='A' and t_typ.layout!='A':
        new_s_typ = s_typ.copy(layout=t_typ.layout)
        calltypes[stmt].args = (new_s_typ, calltypes[stmt].args[1], calltypes[stmt].args[2])
    return


def dprint_func_ir(func_ir, title):
    if config.DEBUG_ARRAY_OPT==1:
        name = func_ir.func_id.func_qualname
        print(("IR %s: %s" % (title, name)).center(80, "-"))
        func_ir.dump()
        print("-"*40)

def find_topo_order(blocks):
    """find topological order of blocks such that true branches are visited
    first (e.g. for_break test in test_dataflow).
    """
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

def get_call_table(blocks, call_table={}, reverse_call_table={}):
    """returns a dictionary of call variables and their references.
    """
    # call_table xample: c = np.zeros becomes c:["zeroes", np]
    # reverse_call_table example: c = np.zeros becomes np_var:c

    topo_order = find_topo_order(blocks)
    for label in reversed(topo_order):
        for inst in reversed(blocks[label].body):
            if isinstance(inst, ir.Assign):
                lhs = inst.target.name
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op=='call':
                    call_table[rhs.func.name] = []
                if isinstance(rhs, ir.Expr) and rhs.op=='getattr':
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
            for T,f in call_table_extensions.items():
                if isinstance(inst,T):
                    f(inst, call_table, reverse_call_table)
    return call_table, reverse_call_table

def get_stmt_writes(stmt):
    writes = set()
    if isinstance(stmt, (ir.Assign, ir.SetItem, ir.StaticSetItem)):
        writes.add(stmt.target.name)
    return writes

def rename_labels(blocks):
    """rename labels of function body blocks according to topological sort.
    lowering requires this order.
    """
    topo_order = find_topo_order(blocks)

    # make a block with return last if available (just for readability)
    return_label = -1
    for l,b in blocks.items():
        if isinstance(b.body[-1], ir.Return):
            return_label = l
    # some cases like generators can have no return blocks
    if return_label!=-1:
        topo_order.remove(return_label)
        topo_order.append(return_label)

    label_map = {}
    new_label = 0
    for label in topo_order:
        label_map[label] = new_label
        new_label += 1
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
