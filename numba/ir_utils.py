from numba import ir, types, typing
from numba.typing.templates import signature
import numpy
from numba.analysis import *

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

INT_TYPE = types.scalars.Integer.from_bitwidth(64)
BOOL_TYPE = types.scalars.Boolean("bool")

def mk_alloc(typemap, calltypes, lhs, size_var, dtype, scope, loc):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    """
    out = []
    ndims = 1
    size_typ = INT_TYPE
    if isinstance(size_var, tuple):
        # tuple_var = build_tuple([size_var...])
        ndims = len(size_var)
        tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
        typemap[tuple_var.name] = types.containers.UniTuple(INT_TYPE, ndims)
        tuple_call = ir.Expr.build_tuple(list(size_var), loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out.append(tuple_assign)
        size_var = tuple_var
        size_typ = types.containers.UniTuple(INT_TYPE, ndims)
    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: empty_attr = getattr(g_np_var, empty)
    empty_attr_call = ir.Expr.getattr(g_np_var, "empty", loc)
    attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
    typemap[attr_var.name] = _get_empty_func_typ()
    attr_assign = ir.Assign(empty_attr_call, attr_var, loc)
    # alloc call: lhs = empty_attr(size_var, typ_var)
    typ_var = ir.Var(scope, mk_unique_var("$np_typ_var"), loc)
    typemap[typ_var.name] = types.functions.NumberClass(dtype)
    # assuming str(dtype) returns valid np dtype string
    np_typ_getattr = ir.Expr.getattr(g_np_var, str(dtype), loc)
    typ_var_assign = ir.Assign(np_typ_getattr, typ_var, loc)
    alloc_call = ir.Expr.call(attr_var, [size_var, typ_var], (), loc)
    calltypes[alloc_call] = typemap[attr_var.name].get_call_type(
        typing.Context(), [size_typ, types.functions.NumberClass(dtype)], {})
    #signature(
    #    types.npytypes.Array(dtype, ndims, 'C'), size_typ,
    #    types.functions.NumberClass(dtype))
    alloc_assign = ir.Assign(alloc_call, lhs, loc)

    out.extend([g_np_assign, attr_assign, typ_var_assign, alloc_assign])
    return out

def _get_empty_func_typ():
    for (k,v) in typing.npydecl.registry.globals:
        if k==numpy.empty:
            return v
    raise RuntimeError("empty() type not found")

def mk_range_block(typemap, size_var, calltypes, scope, loc):
    """make a block that initializes loop range and iteration variables.
    target label in jump needs to be set.
    """
    # g_range_var = Global(range)
    g_range_var = ir.Var(scope, mk_unique_var("$range_g_var"), loc)
    typemap[g_range_var.name] = _get_range_func_typ()
    g_range = ir.Global('range', range, loc)
    g_range_assign = ir.Assign(g_range, g_range_var, loc)
    # range_call_var = call g_range_var(size_var)
    range_call = ir.Expr.call(g_range_var, [size_var], (), loc)
    calltypes[range_call] = typemap[g_range_var.name].get_call_type(
        typing.Context(), [types.int64], {})
    #signature(types.range_state64_type, types.int64)
    range_call_var = ir.Var(scope, mk_unique_var("$range_c_var"), loc)
    typemap[range_call_var.name] = types.iterators.RangeType(INT_TYPE)
    range_call_assign = ir.Assign(range_call, range_call_var, loc)
    # iter_var = getiter(range_call_var)
    iter_call = ir.Expr.getiter(range_call_var ,loc)
    calltypes[iter_call] = signature(types.range_iter64_type,
        types.range_state64_type)
    iter_var = ir.Var(scope, mk_unique_var("$iter_var"), loc)
    typemap[iter_var.name] = types.iterators.RangeIteratorType(INT_TYPE)
    iter_call_assign = ir.Assign(iter_call, iter_var, loc)
    # $phi = iter_var
    phi_var = ir.Var(scope, mk_unique_var("$phi"), loc)
    typemap[phi_var.name] = types.iterators.RangeIteratorType(INT_TYPE)
    phi_assign = ir.Assign(iter_var, phi_var, loc)
    # jump to header
    jump_header = ir.Jump(-1, loc)
    range_block = ir.Block(scope, loc)
    range_block.body = [g_range_assign, range_call_assign, iter_call_assign,
        phi_assign, jump_header]
    return range_block

def _get_range_func_typ():
    for (k,v) in typing.templates.builtin_registry.globals:
        if k==range:
            return v
    raise RuntimeError("range type not found")

def mk_loop_header(typemap, phi_var, calltypes, scope, loc):
    """make a block that is a loop header updating iteration variables.
    target labels in branch need to be set.
    """
    # iternext_var = iternext(phi_var)
    iternext_var = ir.Var(scope, mk_unique_var("$iternext_var"), loc)
    typemap[iternext_var.name] = types.containers.Pair(INT_TYPE, BOOL_TYPE)
    iternext_call = ir.Expr.iternext(phi_var, loc)
    calltypes[iternext_call] = signature(
        types.containers.Pair(INT_TYPE, BOOL_TYPE), types.range_iter64_type)
    iternext_assign = ir.Assign(iternext_call, iternext_var, loc)
    # pair_first_var = pair_first(iternext_var)
    pair_first_var = ir.Var(scope, mk_unique_var("$pair_first_var"), loc)
    typemap[pair_first_var.name] = INT_TYPE
    pair_first_call = ir.Expr.pair_first(iternext_var, loc)
    pair_first_assign = ir.Assign(pair_first_call, pair_first_var, loc)
    # pair_second_var = pair_second(iternext_var)
    pair_second_var = ir.Var(scope, mk_unique_var("$pair_second_var"), loc)
    typemap[pair_second_var.name] = BOOL_TYPE
    pair_second_call = ir.Expr.pair_second(iternext_var, loc)
    pair_second_assign = ir.Assign(pair_second_call, pair_second_var, loc)
    # phi_b_var = pair_first_var
    phi_b_var = ir.Var(scope, mk_unique_var("$phi"), loc)
    typemap[phi_b_var.name] = INT_TYPE
    phi_b_assign = ir.Assign(pair_first_var, phi_b_var, loc)
    # branch pair_second_var body_block out_block
    branch = ir.Branch(pair_second_var, -1, -1, loc)
    header_block = ir.Block(scope, loc)
    header_block.body = [iternext_assign, pair_first_assign,
        pair_second_assign, phi_b_assign, branch]
    return header_block

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

def replace_var_names(blocks, namedict):
    def replace_name(var, namedict):
        assert isinstance(var, ir.Var)
        var.name = namedict.get(var.name, var.name)
    visit_vars(blocks, replace_name, namedict)

def replace_vars(blocks, vardict):
    def replace_var(var, vardict):
        assert isinstance(var, ir.Var)
        new_var = vardict.get(var, var)
        var.scope = new_var.scope
        var.name = new_var.name
        var.loc = new_var.loc
    visit_vars(blocks, replace_var, vardict)

# other packages that define new nodes add calls to visit variables in them
# format: {type:function}
visit_vars_extensions = {}

def visit_vars(blocks, callback, cbdata):
    """go over statements of block bodies and replace variable names with
    dictionary.
    """
    for block in blocks.values():
        for stmt in block.body:
            # let external calls handle stmt if type matches
            for t,f in visit_vars_extensions.items():
                if isinstance(stmt,t):
                    f(stmt, callback, cbdata)
                    return
            if isinstance(stmt, ir.Assign):
                visit_vars_inner(stmt.target, callback, cbdata)
                visit_vars_inner(stmt.value, callback, cbdata)
            elif isinstance(stmt, ir.Arg):
                visit_vars_inner(stmt.name, callback, cbdata)
            elif isinstance(stmt, ir.Return):
                visit_vars_inner(stmt.value, callback, cbdata)
            elif isinstance(stmt, ir.Branch):
                visit_vars_inner(stmt.cond, callback, cbdata)
            elif isinstance(stmt, ir.Jump):
                visit_vars_inner(stmt.target, callback, cbdata)
            elif isinstance(stmt, ir.Del):
                visit_vars_inner(stmt.value, callback, cbdata)
            elif isinstance(stmt, ir.DelAttr):
                visit_vars_inner(stmt.target, callback, cbdata)
                visit_vars_inner(stmt.attr, callback, cbdata)
            elif isinstance(stmt, ir.SetAttr):
                visit_vars_inner(stmt.target, callback, cbdata)
                visit_vars_inner(stmt.attr, callback, cbdata)
                visit_vars_inner(stmt.value, callback, cbdata)
            elif isinstance(stmt, ir.DelItem):
                visit_vars_inner(stmt.target, callback, cbdata)
                visit_vars_inner(stmt.index, callback, cbdata)
            elif isinstance(stmt, ir.StaticSetItem):
                visit_vars_inner(stmt.target, callback, cbdata)
                visit_vars_inner(stmt.index_var, callback, cbdata)
                visit_vars_inner(stmt.value, callback, cbdata)
            elif isinstance(stmt, ir.SetItem):
                visit_vars_inner(stmt.target, callback, cbdata)
                visit_vars_inner(stmt.index, callback, cbdata)
                visit_vars_inner(stmt.value, callback, cbdata)
            else:
                raise NotImplementedError("no replacement for IR node: ", stmt)
    return

def visit_vars_inner(node, callback, cbdata):
    if isinstance(node, ir.Var):
        callback(node, cbdata)
    elif isinstance(node, list):
        [visit_vars_inner(n, callback, cbdata) for n in node]
    elif isinstance(node, ir.Expr):
        # if node.op in ['binop', 'inplace_binop']:
        #     lhs = node.lhs.name
        #     rhs = node.rhs.name
        #     node.lhs.name = callback, cbdata.get(lhs, lhs)
        #     node.rhs.name = callback, cbdata.get(rhs, rhs)
        for arg in node._kws.keys():
            visit_vars_inner(node._kws[arg], callback, cbdata)
    return

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
    for block in blocks.values():
        new_body = []
        for stmt in block.body:
            if not isinstance(stmt, ir.Del):
                new_body.append(stmt)
        block.body = new_body
    return

def remove_dead(blocks):
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)

    for label, block in blocks.items():
        # find live variables at each statement to delete dead assignment
        lives = { v.name for v in block.terminator.list_vars() }
        # find live variables at the end of block
        for out_blk, _data in cfg.successors(label):
            lives |= live_map[out_blk]
        remove_dead_block(block, lives)
    return

# other packages that define new nodes add calls to remove dead code in them
# format: {type:function}
remove_dead_extensions = {}

def remove_dead_block(block, lives):
    # add statements in reverse order
    new_body = [block.terminator]
    # for each statement in reverse order, excluding terminator
    for stmt in reversed(block.body[:-1]):
        # let external calls handle stmt if type matches
        for t,f in remove_dead_extensions.items():
            if isinstance(stmt,t):
                f(stmt, lives)
        # ignore assignments that their lhs is not live
        if not isinstance(stmt, ir.Assign) or stmt.target.name in lives:
            lives |= { v.name for v in stmt.list_vars() }
            new_body.append(stmt)
    new_body.reverse()
    block.body = new_body
    return
