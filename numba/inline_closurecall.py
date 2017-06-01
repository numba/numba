from numba import config, ir, ir_utils, utils
import types

from numba.ir_utils import (
    mk_unique_var,
    next_label,
    add_offset_to_labels,
    replace_vars,
    remove_dels,
    remove_dead,
    rename_labels)

from numba.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.transforms import _extract_loop_lifting_candidates, _loop_lift_get_candidate_infos
from numba.targets.rangeobj import range_iter_len
import numpy


class GuardException(Exception):
    pass

def require(cond):
    """
    Raise GuardException if the given condition is False.
    """
    if not cond:
       raise GuardException

def guard(func, *args):
    """
    Run a function with given set of of arguments, and turn any
    GuardException raised in the process into returning None.
    """
    try:
        return func(*args)
    except GuardException:
        return None

class InlineClosureCallPass(object):
    """InlineClosureCallPass class looks for direct calls to locally defined
    closures, and inlines the body of the closure function to the call site.
    """

    def __init__(self, func_ir, run_frontend):
        self.func_ir = func_ir
        self.run_frontend = run_frontend

    def run(self):
        """Run inline closure call pass.
        """
        modified = False
        work_list = list(self.func_ir.blocks.items())
        _debug_print("START InlineClosureCall")
        while work_list:
            label, block = work_list.pop()
            for i in range(len(block.body)):
                instr = block.body[i]
                if isinstance(instr, ir.Assign):
                    lhs = instr.target
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        try:
                            func_def = self.func_ir.get_definition(expr.func)
                        except KeyError:
                            func_def = None
                        _debug_print(
                            "found call to ", expr.func, " def = ", func_def)
                        if isinstance(
                                func_def, ir.Expr) and func_def.op == "make_function":
                            new_blocks = self.inline_closure_call(
                                block, i, func_def)
                            for block in new_blocks:
                                work_list.append(block)
                            modified = True
                            # current block is modified, skip the rest
                            break

        if modified:
            _fix_nested_array(self.func_ir)
            remove_dels(self.func_ir.blocks)
            # repeat dead code elimintation until nothing can be further
            # removed
            while (remove_dead(self.func_ir.blocks, self.func_ir.arg_names)):
                pass
            self.func_ir.blocks = rename_labels(self.func_ir.blocks)
        _debug_print("AFTER InlineClosureCall")

    def inline_closure_call(self, block, i, callee):
        """Inline the body of `callee` at its callsite (`i`-th instruction of `block`)
        """
        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        _debug_print("Found closure call: ", instr, " with callee = ", callee)
        func_ir = self.func_ir
        # first, get the IR of the callee
        from_ir = self.get_ir_of_code(callee.code)
        from_blocks = from_ir.blocks

        if config.DEBUG_INLINE_CLOSURE:
            print("Before inline arraycall: ")
            from_ir.dump()
        guard(_inline_arraycall, func_ir, block, from_ir, from_blocks, i)
        if config.DEBUG_INLINE_CLOSURE:
            print("After inline arraycall: ")
            from_ir.dump()

        # 1. relabel from_ir by adding an offset
        max_label = max(func_ir.blocks.keys())
        from_blocks = add_offset_to_labels(from_blocks, max_label + 1)
        from_ir.blocks = from_blocks
        min_label = min(from_blocks.keys())
        max_label = max(from_blocks.keys())
        #    reset globals in ir_utils before we use it
        ir_utils._max_label = max_label
        # 2. rename all local variables in from_ir with new locals created in
        # func_ir
        from_scopes = _get_all_scopes(from_blocks)
        _debug_print("obj_IR has scopes: ", from_scopes)
        #    one function should only have one local scope
        assert(len(from_scopes) == 1)
        from_scope = from_scopes[0]
        var_dict = {}
        for var in from_scope.localvars._con.values():
            if not (var.name in callee.code.co_freevars):
                new_var = scope.define(mk_unique_var(var.name), loc=var.loc)
                var_dict[var.name] = new_var
        _debug_print("Before local var rename: var_dict = ", var_dict)
        _debug_dump(from_ir)
        replace_vars(from_blocks, var_dict)
        _debug_print("After local var rename: ")
        _debug_dump(from_ir)
        # 3. replace formal parameters with actual arguments
        args = list(call_expr.args)
        if callee.defaults:
            _debug_print("defaults", callee.defaults)
            if isinstance(callee.defaults, tuple):  # Python 3.5
                args = args + list(callee.defaults)
            elif isinstance(callee.defaults, ir.Var) or isinstance(callee.defaults, str):
                defaults = func_ir.get_definition(callee.defaults)
                assert(isinstance(defaults, ir.Const))
                loc = defaults.loc
                args = args + [ir.Const(value=v, loc=loc)
                               for v in defaults.value]
            else:
                raise NotImplementedError(
                    "Unsupported defaults to make_function: {}".format(defaults))
        _replace_args_with(from_blocks, args)
        _debug_print("After arguments rename: ")
        _debug_dump(from_ir)
        # 4. replace freevar with actual closure var
        if callee.closure:
            closure = func_ir.get_definition(callee.closure)
            assert(isinstance(closure, ir.Expr)
                   and closure.op == 'build_tuple')
            assert(len(callee.code.co_freevars) == len(closure.items))
            _debug_print("callee's closure = ", closure)
            _replace_freevars(from_blocks, closure.items)
            _debug_print("After closure rename: ")
            _debug_dump(from_ir)
        # 5. split caller blocks into two
        new_blocks = []
        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1:]
        new_label = next_label()
        func_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))
        # 6. replace Return with assignment to LHS
        topo_order = find_topo_order(from_blocks)
        _replace_returns(from_blocks, instr.target, new_label)
        #    remove the old definition of instr.target too
        if (instr.target.name in func_ir._definitions):
            func_ir._definitions[instr.target.name] = []
        # 7. insert all new blocks, and add back definitions
        for label in topo_order:
            # block scope must point to parent's
            block = from_blocks[label]
            block.scope = scope
            _add_definition(func_ir, block)
            _simplify_range_len(func_ir, block)
            func_ir.blocks[label] = block
            new_blocks.append((label, block))
        _debug_print("After merge: ")
        _debug_dump(func_ir)
        return new_blocks

    def get_ir_of_code(self, fcode):
        """
        Compile a code object to get its IR.
        """
        glbls = self.func_ir.func_id.func.__globals__
        nfree = len(fcode.co_freevars)
        func_env = "\n".join(["  c_%d = None" % i for i in range(nfree)])
        func_clo = ",".join(["c_%d" % i for i in range(nfree)])
        func_arg = ",".join(["x_%d" % i for i in range(fcode.co_argcount)])
        func_text = "def g():\n%s\n  def f(%s):\n    return (%s)\n  return f" % (
            func_env, func_arg, func_clo)
        loc = {}
        exec(func_text, glbls, loc)

        # hack parameter name .0 for Python 3 versions < 3.6
        if utils.PYVERSION >= (3,) and utils.PYVERSION < (3, 6):
            co_varnames = list(fcode.co_varnames)
            if co_varnames[0] == ".0":
                co_varnames[0] = "implicit0"
            fcode = types.CodeType(
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
        ir = self.run_frontend(f)
        return ir


def _debug_print(*args):
    if config.DEBUG_INLINE_CLOSURE:
        print(args)


def _debug_dump(func_ir):
    if config.DEBUG_INLINE_CLOSURE:
        func_ir.dump()


def _get_all_scopes(blocks):
    """Get all block-local scopes from an IR.
    """
    all_scopes = []
    for label, block in blocks.items():
        if not (block.scope in all_scopes):
            all_scopes.append(block.scope)
    return all_scopes


def _replace_args_with(blocks, args):
    """
    Replace ir.Arg(...) with real arguments from call site
    """
    for label, block in blocks.items():
        assigns = block.find_insts(ir.Assign)
        for stmt in assigns:
            if isinstance(stmt.value, ir.Arg):
                idx = stmt.value.index
                assert(idx < len(args))
                stmt.value = args[idx]


def _replace_freevars(blocks, args):
    """
    Replace ir.FreeVar(...) with real variables from parent function
    """
    for label, block in blocks.items():
        assigns = block.find_insts(ir.Assign)
        for stmt in assigns:
            if isinstance(stmt.value, ir.FreeVar):
                idx = stmt.value.index
                assert(idx < len(args))
                stmt.value = args[idx]


def _replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for label, block in blocks.items():
        for i in range(len(block.body)):
            stmt = block.body[i]
            if isinstance(stmt, ir.Return):
                assert(i + 1 == len(block.body))
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))


def _add_definition(func_ir, block):
    """
    Add variable definitions to parent func_ir
    """
    definitions = func_ir._definitions
    assigns = block.find_insts(ir.Assign)
    for stmt in assigns:
        print("add definition: ", stmt)
        definitions[stmt.target.name].append(stmt.value)

def _look_for_arraycall(func_ir, block, i):
    """Look for statement like "x = numpy.array(y)", or "x[..] = y"
       immediately after the closure call, where y is the value returned
       in the closure call.
    """
    array_var = None
    array_call_index = None
    list_var_dead_after_array_call = False
    list_var = block.body[i].target
    require(isinstance(list_var, ir.Var))
    print("list_var = ", list_var)
    i = i + 1
    while i < len(block.body):
        instr = block.body[i]
        print("look_for_arraycall", instr)
        if isinstance(instr, ir.Del):
            print("Del")
            if array_var and instr.value == list_var.name:
                print("list_var ", list_var, " is dead")
                list_var_dead_after_array_call = True
                break
            pass
        elif isinstance(instr, ir.Assign):
            # found x = array(list_var)
            lhs  = instr.target
            expr = instr.value
            if (guard(_find_numpy_call, func_ir, expr) == 'array' and
                isinstance(expr.args[0], ir.Var) and
                expr.args[0].name == list_var.name):
                print("found array call")
                array_var = lhs
                array_stmt_index = i
        elif (isinstance(instr, ir.SetItem) and
              isinstance(instr.value, ir.Var) and
              instr.value.name == list_var.name):
            # found x[..] = list_var, potentially nested array detected
            array_var = instr.target
            array_def = func_ir.get_definition(array_var)
            print("array_def = ", array_def)
            require(guard(_find_numpy_call, func_ir, array_def) == 'empty')
            print("array_def is numpy.empty call")
            array_stmt_index = i
        else:
            break
        i = i + 1
    require(array_var and list_var_dead_after_array_call)
    return array_stmt_index

def _find_numpy_call(func_ir, expr):
    """Check if a call expression is calling numpy functions, and
    return the callee's function name if it is, or None otherwise.
    """
    require(isinstance(expr, ir.Expr) and expr.op == 'call')
    callee = expr.func
    callee_def = func_ir.get_definition(callee)
    print("callee = ", callee, " def = ", callee_def)
    require(isinstance(callee_def, ir.Expr) and callee_def.op == 'getattr')
    # and callee_def.attr == 'array':
    module = callee_def.value
    module_def = func_ir.get_definition(module)
    print("module = ", module, " def = ", module_def)
    require(isinstance(module_def, ir.Global) and module_def.value == numpy)
    return callee_def.attr

def _inline_arraycall(parent_ir, parent_block, from_ir, from_blocks, i):
    array_call_index = _look_for_arraycall(parent_ir, parent_block, i)
    print("found valid array call or set: ", parent_block.body[array_call_index])

    # try to identify loop
    from_cfg = compute_cfg_from_blocks(from_blocks)
    loops = [ loop for loop in find_top_level_loops(from_cfg)]
    print("from cfg: ")
    from_cfg.dump()
    print("toplevel loops = ", loops)
    # We require only one top-level loop
    require(len(loops) == 1)
    loop = loops[0]
    # Return statement is in exit block
    require(len(loop.exits) == 1)
    exit_block = from_blocks[list(loop.exits)[0]]
    returns = list(exit_block.find_insts(ir.Return))
    # We require only one return statement
    require(len(returns) == 1)
    return_var = returns[0].value
    return_var_def = from_ir.get_definition(return_var)
    print("return variable = ", return_var, " def = ", return_var_def)
    # Check if def is a cast
    require(isinstance(return_var_def, ir.Expr) and return_var_def.op == 'cast')
    cast_var = return_var_def.value
    cast_def = from_ir.get_definition(cast_var)
    print("cast var = ", cast_var, " def = ", cast_def)
    # Check if the definition is a build_list
    require(isinstance(cast_def, ir.Expr) and cast_def.op ==  'build_list')
    # look for list_append in loop body
    list_append_stmts = []
    for label in loop.body:
        block = from_blocks[label]
        print("check loop body block ", label)
        for stmt in block.find_insts(ir.Assign):
            lhs = stmt.target
            expr = stmt.value
            if isinstance(expr, ir.Expr) and expr.op == 'call':
                func_def = from_ir.get_definition(expr.func)
                if isinstance(func_def, ir.Expr) and func_def.op == 'getattr' \
                  and func_def.attr == 'append':
                    list_def = from_ir.get_definition(func_def.value)
                    print("list_def = ", list_def, list_def == cast_def)
                    if list_def == cast_def:
                        # found matching append call
                        list_append_stmts.append((block, stmt))
    # Require only one list_append, otherwise we won't know the indices
    require(len(list_append_stmts) == 1)
    # Find iterator
    iter_vars = []
    loop_header = from_blocks[loop.header]
    for stmt in loop_header.find_insts(ir.Assign):
        expr = stmt.value
        if isinstance(expr, ir.Expr) and expr.op == 'iternext':
            iter_def = from_ir.get_definition(expr.value)
            print("iter_def = ", iter_def)
            # Require the iterator to be arg(0)
            require(isinstance(iter_def, ir.Arg) and iter_def.index == 0)
            iter_vars.append(expr.value)

    # Require only one iterator in loop header
    require(len(iter_vars) == 1)
    iter_var = iter_vars[0] # variable that holds the range object
    require(len(loop.entries) == 1)

    loop_entry = from_blocks[list(loop.entries)[0]]
    terminator = loop_entry.terminator
    scope = loop_entry.scope
    loc = loop_entry.loc
    stmts = []
    for i in range(len(loop_entry.body) - 1):
        stmt = loop_entry.body[i]
        # Remove list construction
        if stmt.value == list_def:
            print("replace list_def expr by None")
            stmt.value = ir.Const(None, loc=stmt.loc)
        stmts.append(stmt)

    # Insert statement to get size of the iterator
    size_var = scope.make_temp(loc)
    size_tuple_var = scope.make_temp(loc)
    len_func = scope.make_temp(loc)
    stmts.append(ir.Assign(value=ir.Global('range_iter_len', range_iter_len, loc=loc), target=len_func, loc=loc))
    stmts.append(ir.Assign(
                 value=ir.Expr.call(len_func, (iter_var,), (), loc=loc),
                 target=size_var, loc=loc))
    stmts.append(ir.Assign(
                 value=ir.Expr.build_tuple(items=[size_var], loc=loc),
                 target=size_tuple_var, loc=loc))
    array_var = scope.make_temp(loc)
    # Insert array allocation
    array_var = scope.make_temp(loc)
    numpy_var = scope.make_temp(loc)
    empty_func = scope.make_temp(loc)
    stmts.append(ir.Assign(value=ir.Global('numpy', numpy, loc=loc),
                 target=numpy_var, loc=loc))
    stmts.append(ir.Assign(value=ir.Expr.getattr(value=numpy_var, attr='empty', loc=loc),
                 target=empty_func, loc=loc))
    stmts.append(ir.Assign(
                 value=ir.Expr.call(empty_func, (size_tuple_var,), (), loc=loc),
                 target=array_var, loc=loc))
    # Make index var
    index_var = scope.make_temp(loc)
    stmts.append(ir.Assign(
                 value=ir.Const(value=-1, loc=loc),
                 target=index_var, loc=loc))
    # modify loop_entry
    stmts.append(terminator)
    loop_entry.body = stmts

    # Add index increment to loop header
    terminator = loop_header.terminator
    stmts = loop_header.body[0:-1]
    next_index_var = scope.make_temp(loc)
    one = scope.make_temp(loc)
    stmts.append(ir.Assign(
                 value=ir.Const(value=1,loc=loc),
                 target=one, loc=loc))
    stmts.append(ir.Assign(
                 value=ir.Expr.binop(fn='+', lhs=index_var, rhs=one, loc=loc),
                 target=next_index_var, loc=loc))
    stmts.append(ir.Assign(
                 value=next_index_var,
                 target=index_var, loc=loc))
    stmts.append(terminator)
    loop_header.body = stmts

    # Change list_append into array assign
    # Prepend stmts to loop_head
    append_block, append_stmt = list_append_stmts[0]
    for i in range(len(append_block.body)):
        if append_block.body[i] == append_stmt:
            print("Replace append with SetItem")
            append_block.body[i] = ir.SetItem(target=array_var, index=index_var,
                                              value=append_stmt.value.args[0], loc=append_stmt.loc)

    # Change return statement
    returns[0].value = array_var

    # Remove outer array call, where a = array(b) => a = b
    # print("found valid array call: ", parent_block.body[array_call_index])
    stmt = parent_block.body[array_call_index]
    if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
        # must be array call, not SetItem
        stmt.value = stmt.value.args[0]

    # finally returns the array created
    return stmt.target

def _simplify_range_len(func_ir, block):
    """simplify range length computation in the hope that we can cutdown dependencies on array allocation call
    """
    scope = block.scope

    def simplify(stmt):
        loc = stmt.loc
        require(isinstance(stmt, ir.Assign) and
                isinstance(stmt.value, ir.Expr) and
                stmt.value.op == 'call')
        expr = stmt.value
        func_def = func_ir.get_definition(expr.func)
        require(isinstance(func_def, ir.Global) and func_def.value == range_iter_len)
        print("found range_iter_len call:", func_def, expr)
        range_iter_var = expr.args[0]
        range_iter_def = func_ir.get_definition(range_iter_var)
        print("found range def: ", range_iter_def)
        require(isinstance(range_iter_def, ir.Expr) and range_iter_def.op == 'getiter')
        range_var = range_iter_def.value
        range_def = func_ir.get_definition(range_var)
        print("found range def:", range_def)
        require(isinstance(range_def, ir.Expr) and range_def.op == 'call')
        range_func = range_def.func
        range_func_def = func_ir.get_definition(range_func)
        require(isinstance(range_func_def, ir.Global) and range_func_def.value == range)
        n_range_args = len(range_def.args)
        # only support range(n), or range(start, stop) with step 1
        require(n_range_args == 1 or n_range_args == 2)
        stmts = []
        if len(range_def.args) == 1:
            size_var = func_ir.get_definition(range_def.args[0], lhs_only=True)
        elif len(range_def.args) == 2:
            start = func_ir.get_definition(range_def.args[0], lhs_only=True)
            stop = func_ir.get_definition(range_def.args[1], lhs_only=True)
            print("found range start, stop: ", start, stop)
            one = scope.make_temp(loc)
            size_var_tmp = scope.make_temp(loc)
            size_var = scope.make_temp(loc)
            definitions = func_ir._definitions
            # one = 1
            value = ir.Const(value=1,loc=loc)
            stmts.append(ir.Assign(value=value, target=one, loc=loc))
            definitions[one.name].append(value)
            # size_var_tmp = start - stop
            value = ir.Expr.binop(fn='-', lhs=start, rhs=stop, loc=loc)
            stmts.append(ir.Assign(value=value, target=size_var_tmp, loc=loc))
            definitions[size_var_tmp.name].append(value)
            # size_var = size_var_tmp + 1
            value = ir.Expr.binop(fn='+', lhs=size_var_tmp, rhs=one, loc=loc)
            stmts.append(ir.Assign(value=value, target=size_var, loc=loc))
            definitions[size_var.name].append(value)
            print("size_var = ", size_var)
        else:
            return False
        stmt.value = size_var
        func_ir._definitions[stmt.target.name] = [size_var]
        return stmts

    block_len = len(block.body)
    i = 0
    while i < block_len:
        stmts = guard(simplify, block.body[i])
        if stmts:
            block.body = block.body[:i-1] + stmts + block.body[i:]
            block_len = len(block.body)
            i += 3
        i += 1

def _fix_nested_array(func_ir):
    """Look for assignment like: a[..] = b, where both a and b are numpy arrays, and
    try to eliminate array b by expanding a with an extra dimension.
    """
    """
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    all_loops = list(cfg.loops().values())
    def find_nest_level(label):
        level = 0
        for loop in all_loops:
            if label in loop.body:
                level += 1
    """

    def find_array_def(arr):
        arr_def = func_ir.get_definition(arr)
        print("find_array_def", arr, arr_def)
        if isinstance(arr_def, ir.Expr):
            if guard(_find_numpy_call, func_ir, arr_def) == 'empty':
                return arr_def
            elif arr_def.op == 'getitem':
                return find_array_def(arr_def.value)
        raise GuardException

    def fix_array_assign(stmt):
        require(isinstance(stmt, ir.SetItem))
        require(isinstance(stmt.value, ir.Var))
        print("found SetItem: ", stmt)
        lhs = stmt.target
        lhs_def = find_array_def(lhs)
        print("found lhs_def: ", lhs_def)
        rhs_def = func_ir.get_definition(stmt.value)
        print("found rhs_def: ", rhs_def)
        require(isinstance(rhs_def, ir.Expr))
        require(_find_numpy_call(func_ir, rhs_def) == 'empty')
        # find inner array dimension
        dim_def = func_ir.get_definition(rhs_def.args[0])
        require(isinstance(dim_def, ir.Expr) and dim_def.op == 'build_tuple')
        print("dim_def = ", dim_def)
        extra_dims = [ func_ir.get_definition(x, lhs_only=True) for x in dim_def.items ]
        print("extra_dims = ", extra_dims)
        # expand size tuple
        size_tuple_def = func_ir.get_definition(lhs_def.args[0])
        require(isinstance(size_tuple_def, ir.Expr) and size_tuple_def.op == 'build_tuple')
        print("size_tuple_def = ", size_tuple_def)
        size_tuple_def.items += extra_dims
        # replace rhs_def with getitem
        rhs_def.op = 'getitem'
        rhs_def.value = func_ir.get_definition(lhs, lhs_only=True)
        rhs_def.index = stmt.index
        del rhs_def._kws['func']
        del rhs_def._kws['args']
        del rhs_def._kws['vararg']
        del rhs_def._kws['kws']
        # success
        return True

    for label, block in func_ir.blocks.items():
        block_len = len(block.body)
        i = 0
        while i < block_len:
            stmt = block.body[i]
            if guard(fix_array_assign, stmt):
                block.body = block.body[:i] + block.body[i+1:]
                block_len -= 1
            i += 1

