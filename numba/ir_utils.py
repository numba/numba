from numba import ir, types, typing

unique_var_count = 0
def mk_unique_var(prefix):
    global unique_var_count
    var = prefix + "." + str(unique_var_count)
    unique_var_count = unique_var_count + 1
    return var

INT_TYPE = types.scalars.Integer.from_bitwidth(64)
BOOL_TYPE = types.scalars.Boolean("bool")

def mk_range_block(typemap, size_var, scope, loc):
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
    range_call_var = ir.Var(scope, mk_unique_var("$range_c_var"), loc)
    typemap[range_call_var.name] = types.iterators.RangeType(INT_TYPE)
    range_call_assign = ir.Assign(range_call, range_call_var, loc)
    # iter_var = getiter(range_call_var)
    iter_call = ir.Expr.getiter(range_call_var ,loc)
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

def mk_loop_header(typemap, phi_var, scope, loc):
    """make a block that is a loop header updating iteration variables.
    target labels in branch need to be set.
    """
    # iternext_var = iternext(phi_var)
    iternext_var = ir.Var(scope, mk_unique_var("$iternext_var"), loc)
    typemap[iternext_var.name] = types.containers.Pair(INT_TYPE, BOOL_TYPE)
    iternext_call = ir.Expr.iternext(phi_var, loc)
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
