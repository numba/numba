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

enable_inline_arraycall = True


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
        debug_print = _make_debug_print("InlineClosureCallPass")
        debug_print("START")
        while work_list:
            label, block = work_list.pop()
            for i in range(len(block.body)):
                instr = block.body[i]
                if isinstance(instr, ir.Assign):
                    lhs = instr.target
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        func_def = guard(_get_definition, self.func_ir, expr.func)
                        debug_print("found call to ", expr.func, " def = ", func_def)
                        if isinstance(func_def, ir.Expr) and func_def.op == "make_function":
                            new_blocks = self.inline_closure_call(block, i, func_def)
                            for block in new_blocks:
                                work_list.append(block)
                            modified = True
                            # current block is modified, skip the rest
                            break

        if modified:
            if enable_inline_arraycall:
                _fix_nested_array(self.func_ir)
            remove_dels(self.func_ir.blocks)
            # repeat dead code elimintation until nothing can be further
            # removed
            while (remove_dead(self.func_ir.blocks, self.func_ir.arg_names)):
                pass
            self.func_ir.blocks = rename_labels(self.func_ir.blocks)
        debug_print("END")

    def inline_closure_call(self, block, i, callee):
        """Inline the body of `callee` at its callsite (`i`-th instruction of `block`)
        """
        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        debug_print = _make_debug_print("inline_closure_call")
        debug_print("Found closure call: ", instr, " with callee = ", callee)
        func_ir = self.func_ir
        # first, get the IR of the callee
        callee_ir = self.get_ir_of_code(callee.code)
        callee_blocks = callee_ir.blocks

        debug_print("Before inline arraycall")
        _debug_dump(callee_ir)
        if enable_inline_arraycall:
            guard(_inline_arraycall, func_ir, block, callee_ir, callee_blocks, i)
        debug_print("After inline arraycall")
        _debug_dump(callee_ir)

        # 1. relabel callee_ir by adding an offset
        max_label = max(func_ir.blocks.keys())
        callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        #    reset globals in ir_utils before we use it
        ir_utils._max_label = max_label
        ir_utils.visit_vars_extensions = {}
        debug_print("After relabel")
        _debug_dump(callee_ir)

        # 2. rename all local variables in callee_ir with new locals created in func_ir
        callee_scopes = _get_all_scopes(callee_blocks)
        debug_print("callee_scopes = ", callee_scopes)
        #    one function should only have one local scope
        assert(len(callee_scopes) == 1)
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in callee_scope.localvars._con.values():
            if not (var.name in callee.code.co_freevars):
                new_var = scope.define(mk_unique_var(var.name), loc=var.loc)
                var_dict[var.name] = new_var
        debug_print("var_dict = ", var_dict)
        replace_vars(callee_blocks, var_dict)
        debug_print("After local var rename")
        _debug_dump(callee_ir)

        # 3. replace formal parameters with actual arguments
        args = list(call_expr.args)
        if callee.defaults:
            debug_print("defaults = ", callee.defaults)
            if isinstance(callee.defaults, tuple): # Python 3.5
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
        _replace_args_with(callee_blocks, args)
        debug_print("After arguments rename: ")
        _debug_dump(callee_ir)

        # 4. replace freevar with actual closure var
        if callee.closure:
            closure = func_ir.get_definition(callee.closure)
            assert(isinstance(closure, ir.Expr)
                   and closure.op == 'build_tuple')
            assert(len(callee.code.co_freevars) == len(closure.items))
            debug_print("callee's closure = ", closure)
            _replace_freevars(callee_blocks, closure.items)
            debug_print("After closure rename")
            _debug_dump(callee_ir)

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
        topo_order = find_topo_order(callee_blocks)
        _replace_returns(callee_blocks, instr.target, new_label)
        #    remove the old definition of instr.target too
        if (instr.target.name in func_ir._definitions):
            func_ir._definitions[instr.target.name] = []

        # 7. insert all new blocks, and add back definitions
        for label in topo_order:
            # block scope must point to parent's
            block = callee_blocks[label]
            block.scope = scope
            _add_definition(func_ir, block)
            func_ir.blocks[label] = block
            new_blocks.append((label, block))
        debug_print("After merge in")
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

def _make_debug_print(prefix):
    def debug_print(*args):
        if config.DEBUG_INLINE_CLOSURE:
            print(prefix + ": " + "".join(str(x) for x in args))
    return debug_print

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
    Add variable definitions found in a block to parent func_ir.
    """
    definitions = func_ir._definitions
    assigns = block.find_insts(ir.Assign)
    for stmt in assigns:
        definitions[stmt.target.name].append(stmt.value)

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
    Run a function with given set of arguments, and guard against
    any GuardException raised by the function by returning None,
    or the expected return results if no such exception was raised.
    """
    try:
        return func(*args)
    except GuardException:
        return None

def _get_definition(func_ir, name, **kwargs):
    """
    Same as func_ir.get_definition(name), but raise GuardException if
    exception KeyError is caught.
    """
    try:
        return func_ir.get_definition(name, **kwargs)
    except KeyError:
        raise GuardException


def _find_arraycall(func_ir, block, i):
    """Look for statement like "x = numpy.array(y)" or "x[..] = y"
    immediately after the closure call that creates list y (the i-th
    statement in block).  Return the statement index if found, or
    raise GuardException.
    """
    array_var = None
    array_call_index = None
    list_var_dead_after_array_call = False
    list_var = block.body[i].target
    require(isinstance(list_var, ir.Var))
    i = i + 1
    while i < len(block.body):
        instr = block.body[i]
        if isinstance(instr, ir.Del):
            # Stop the process if list_var becomes dead
            if array_var and instr.value == list_var.name:
                list_var_dead_after_array_call = True
                break
            pass
        elif isinstance(instr, ir.Assign):
            # Found array_var = array(list_var)
            lhs  = instr.target
            expr = instr.value
            if (guard(_find_numpy_call, func_ir, expr) == 'array' and
                isinstance(expr.args[0], ir.Var) and
                expr.args[0].name == list_var.name):
                array_var = lhs
                array_stmt_index = i
        elif (isinstance(instr, ir.SetItem) and
              isinstance(instr.value, ir.Var) and
              instr.value.name == list_var.name):
            # Found array_var[..] = list_var, the case for nested array
            array_var = instr.target
            array_def = _get_definition(func_ir, array_var)
            require(guard(_find_numpy_call, func_ir, array_def) == 'empty')
            array_stmt_index = i
        else:
            # Bail out otherwise
            break
        i = i + 1
    # require array_var is found, and list_var is dead after array_call.
    require(array_var and list_var_dead_after_array_call)
    _make_debug_print("find_array_call")(block.body[array_stmt_index])
    return array_stmt_index

def _find_numpy_call(func_ir, expr):
    """Check if a call expression is calling a numpy function, and
    return the callee's function name if it is, or raise GuardException.
    """
    require(isinstance(expr, ir.Expr) and expr.op == 'call')
    callee = expr.func
    callee_def = _get_definition(func_ir, callee)
    require(isinstance(callee_def, ir.Expr) and callee_def.op == 'getattr')
    module = callee_def.value
    module_def = _get_definition(func_ir, module)
    require(isinstance(module_def, ir.Global) and module_def.value == numpy)
    _make_debug_print("find_numpy_call")(callee_def.attr)
    return callee_def.attr

def _find_iter_range(func_ir, expr):
    """Find the iterator's actual range if it is either range(n), or range(m, n),
    otherwise return raise GuardException.
    """
    debug_print = _make_debug_print("find_iter_range")
    require(isinstance(expr, ir.Expr) and expr.op == 'call' and len(expr.args) == 1)
    arg = expr.args[0]
    arg_def = _get_definition(func_ir, arg)
    debug_print("arg = ", arg, " def = ", arg_def)
    require(isinstance(arg_def, ir.Expr) and arg_def.op == 'getiter')
    range_var = arg_def.value
    range_def = _get_definition(func_ir, range_var)
    debug_print("range_var = ", range_var, " range_def = ", range_def)
    require(isinstance(range_def, ir.Expr) and range_def.op == 'call')
    func_var = range_def.func
    func_def = _get_definition(func_ir, func_var)
    debug_print("func_var = ", func_var, " func_def = ", func_def)
    require(isinstance(func_def, ir.Global) and func_def.value == range)
    nargs = len(range_def.args)
    if nargs == 1:
        stop = _get_definition(func_ir, range_def.args[0], lhs_only=True)
        return (0, range_def.args[0])
    elif nargs == 2:
        start = _get_definition(func_ir, range_def.args[0], lhs_only=True)
        stop = _get_definition(func_ir, range_def.args[1], lhs_only=True)
        return (start, stop)
    else:
        raise GuardException


def _inline_arraycall(caller_ir, caller_block, callee_ir, callee_blocks, i):
    """Look for array(list) call in the caller, and turn list operations into array operations
    in the callee if the following conditions are met:
      1. The caller block contains an array call on the list;
      2. The list variable is no longer live after array call;
      3. The list is created in the callee (the closure function) from a single loop;
      4. The loop is created from an range iterator whose length is known prior to the loop;
      5. There is only one list_append operation in the loop body;
      6. The block that contains list_append dominates the loop head, which ensures list
         length is the same as loop length;
    If any condition check fails, no modification will be made to the incoming IR.
    """
    debug_print = _make_debug_print("inline_arraycall")
    closure_call_expr = caller_block.body[i].value
    array_call_index = _find_arraycall(caller_ir, caller_block, i)

    # Identify loop structure
    callee_cfg = compute_cfg_from_blocks(callee_blocks)
    loops = [ loop for loop in find_top_level_loops(callee_cfg)]
    debug_print("callee CFG")
    _debug_dump(callee_cfg)
    debug_print("top-level loops", loops)
    # There should be only one top-level loop in the callee
    require(len(loops) == 1)
    loop = loops[0]
    # Return statement must be in exit block
    require(len(loop.exits) == 1)
    exit_block = callee_blocks[list(loop.exits)[0]]
    returns = list(exit_block.find_insts(ir.Return))
    # There should be only one return statement
    require(len(returns) == 1)
    return_var = returns[0].value
    return_var_def = _get_definition(callee_ir, return_var)
    debug_print("return_var = ", return_var, " def = ", return_var_def)
    # Check if return_var_def is a cast
    require(isinstance(return_var_def, ir.Expr) and return_var_def.op == 'cast')
    cast_var = return_var_def.value
    cast_def = _get_definition(callee_ir, cast_var)
    debug_print("cast_var = ", cast_var, " def = ", cast_def)
    # Check if the definition is a build_list
    require(isinstance(cast_def, ir.Expr) and cast_def.op ==  'build_list')

    # Look for list_append in loop body
    list_append_stmts = []
    for label in loop.body:
        block = callee_blocks[label]
        debug_print("check loop body block ", label)
        for stmt in block.find_insts(ir.Assign):
            lhs = stmt.target
            expr = stmt.value
            if isinstance(expr, ir.Expr) and expr.op == 'call':
                func_def = _get_definition(callee_ir, expr.func)
                if isinstance(func_def, ir.Expr) and func_def.op == 'getattr' \
                  and func_def.attr == 'append':
                    list_def = _get_definition(callee_ir, func_def.value)
                    debug_print("list_def = ", list_def, list_def == cast_def)
                    if list_def == cast_def:
                        # found matching append call
                        list_append_stmts.append((label, block, stmt))

    # Require only one list_append, otherwise we won't know the indices
    require(len(list_append_stmts) == 1)
    append_block_label, append_block, append_stmt = list_append_stmts[0]

    # Check if append_block (besides loop entry) dominates loop header.
    # Since CFG doesn't give us this info without loop entry, we approximate
    # by checking if the predecessor set of the header block is the same
    # as loop_entries plus append_block, which is certainly more restrictive
    # than necessary, and can be relaxed if needed.
    preds = set(l for l, b in callee_cfg.predecessors(loop.header))
    debug_print("preds = ", preds, (loop.entries | set([append_block_label])))
    require(preds == (loop.entries | set([append_block_label])))

    # Find iterator in loop header
    iter_vars = []
    iter_first_vars = []
    loop_header = callee_blocks[loop.header]
    for stmt in loop_header.find_insts(ir.Assign):
        expr = stmt.value
        if isinstance(expr, ir.Expr):
            if expr.op == 'iternext':
                iter_def = _get_definition(callee_ir, expr.value)
                debug_print("iter_def = ", iter_def)
                # Require the iterator to be arg(0)
                require(isinstance(iter_def, ir.Arg) and iter_def.index == 0)
                iter_vars.append(expr.value)
            elif expr.op == 'pair_first':
                iter_first_vars.append(stmt.target)

    # Require only one iterator in loop header
    require(len(iter_vars) == 1 and len(iter_first_vars) == 1)
    iter_var = iter_vars[0] # variable that holds the iterator object
    iter_first_var = iter_first_vars[0] # variable that holds the value out of iterator

    # Final requirement: only one loop entry, and we're going to modify it by:
    # 1. replacing the list definition with an array definition;
    # 2. adding a counter for the array iteration.
    require(len(loop.entries) == 1)
    loop_entry = callee_blocks[list(loop.entries)[0]]
    terminator = loop_entry.terminator
    scope = loop_entry.scope
    loc = loop_entry.loc
    stmts = []
    # Skip list construction and skip terminator, add the rest to stmts
    for i in range(len(loop_entry.body) - 1):
        stmt = loop_entry.body[i]
        if stmt.value == list_def:
            debug_print("replace list_def expr by None")
            stmt.value = ir.Const(None, loc=stmt.loc)
        stmts.append(stmt)

    # Define an index_var to index the array.
    # If the range happens to be single step ranges like range(n), or range(m, n),
    # then the index_var correlates to iterator index; otherwise we'll have to
    # define a new counter.
    range_def = guard(_find_iter_range, caller_ir, closure_call_expr)
    index_var = scope.make_temp(loc)
    index_offset_var = None
    if range_def and range_def[0] == 0:
        # iterator starts with 0, index_var can just be iter_first_var
        index_var = iter_first_var
    else:
        # index_var = -1 # starting the index with -1 since it will incremented in loop header
        stmts.append(ir.Assign(
                 value=ir.Const(value=-1, loc=loc),
                 target=index_var, loc=loc))

    # Insert statement to get the size of the loop iterator
    size_var = scope.make_temp(loc)
    if range_def:
        start, stop = range_def
        # simple range with start and stop, which are defined only in caller,
        # So we'll have to pass them as parameters
        nargs = len(closure_call_expr.args)
        if start == 0:
            closure_call_expr.args.append(stop)
            stmts.append(ir.Assign(
                         value=ir.Arg(index=nargs, name='size', loc=loc),
                         target=size_var, loc=loc))
        else:
            closure_call_expr.args.append(start)
            closure_call_expr.args.append(stop)
            start_var = scope.make_temp(loc)
            stop_var = scope.make_temp(loc)
            # start_var = Arg(nargs)
            stmts.append(ir.Assign(
                         value=ir.Arg(index=nargs, name='start', loc=loc),
                         target=start_var, loc=loc))
            # stop_var = Arg(nargs+1)
            stmts.append(ir.Assign(
                         value=ir.Arg(index=nargs+1, name='stop', loc=loc),
                         target=stop_var, loc=loc))
            # size_var = stop_var - start_var
            stmts.append(ir.Assign(
                         value=ir.Expr.binop(fn='-', lhs=stop_var, rhs=start_var, loc=loc),
                         target=size_var, loc=loc))
    else:
        len_func = scope.make_temp(loc)
        stmts.append(ir.Assign(
                     value=ir.Global('range_iter_len', range_iter_len, loc=loc),
                     target=len_func, loc=loc))
        # size_var = len_func(iter_var)
        stmts.append(ir.Assign(
                     value=ir.Expr.call(len_func, (iter_var,), (), loc=loc),
                     target=size_var, loc=loc))

    size_tuple_var = scope.make_temp(loc)
    # size_tuple_var = build_tuple(size_var)
    stmts.append(ir.Assign(
                 value=ir.Expr.build_tuple(items=[size_var], loc=loc),
                 target=size_tuple_var, loc=loc))
    array_var = scope.make_temp(loc)
    # Insert array allocation
    array_var = scope.make_temp(loc)
    numpy_var = scope.make_temp(loc)
    empty_func = scope.make_temp(loc)
    # numpy_var = numpy
    stmts.append(ir.Assign(value=ir.Global('numpy', numpy, loc=loc),
                 target=numpy_var, loc=loc))
    # empty_func = numpy_var.empty
    stmts.append(ir.Assign(value=ir.Expr.getattr(value=numpy_var, attr='empty', loc=loc),
                 target=empty_func, loc=loc))
    # array_var = empty_func(size_tuple_var)
    stmts.append(ir.Assign(
                 value=ir.Expr.call(empty_func, (size_tuple_var,), (), loc=loc),
                 target=array_var, loc=loc))

    # Add back terminator
    stmts.append(terminator)
    # Modify loop_entry
    loop_entry.body = stmts

    if not (range_def and range_def[0] == 0):
        # Insert index_var increment to the end of loop header
        terminator = loop_header.terminator
        stmts = loop_header.body[0:-1]
        next_index_var = scope.make_temp(loc)
        one = scope.make_temp(loc)
        # one = 1
        stmts.append(ir.Assign(
                     value=ir.Const(value=1,loc=loc),
                     target=one, loc=loc))
        # next_index_var = index_var + 1
        stmts.append(ir.Assign(
                     value=ir.Expr.binop(fn='+', lhs=index_var, rhs=one, loc=loc),
                     target=next_index_var, loc=loc))
        # index_var = next_index_var
        stmts.append(ir.Assign(
                     value=next_index_var,
                     target=index_var, loc=loc))
        stmts.append(terminator)
        loop_header.body = stmts

    # In append_block, change list_append into array assign
    for i in range(len(append_block.body)):
        if append_block.body[i] == append_stmt:
            debug_print("Replace append with SetItem")
            append_block.body[i] = ir.SetItem(target=array_var, index=index_var,
                                              value=append_stmt.value.args[0], loc=append_stmt.loc)

    # Change return statement to return array_var instead
    returns[0].value = array_var

    # replace outer array call, by changing "a = array(b)" to "a = b"
    stmt = caller_block.body[array_call_index]
    # stmt can be either array call or SetItem, we only replace array call
    if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
        stmt.value = stmt.value.args[0]

    # finally returns the array created
    return stmt.target


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
        """Find numpy array definition such as arr = numpy.empty(...).  If it is
        arr = b[...], find array defintion of b recursively.
        """
        arr_def = func_ir.get_definition(arr)
        _make_debug_print("find_array_def")(arr, arr_def)
        if isinstance(arr_def, ir.Expr):
            if guard(_find_numpy_call, func_ir, arr_def) == 'empty':
                return arr_def
            elif arr_def.op == 'getitem':
                return find_array_def(arr_def.value)
        raise GuardException

    def fix_array_assign(stmt):
        """For assignment like lhs[idx] = rhs, where both a and b are arrays, do the
        following:
        1. find the definition of rhs, which has to be a call to numpy.empty
        2. find the source array creation for lhs, insert an extra dimension of size of b.
        3. replace the definition of rhs = numpy.empty(...) with rhs = lhs[idx]
        """
        require(isinstance(stmt, ir.SetItem))
        require(isinstance(stmt.value, ir.Var))
        debug_print = _make_debug_print("fix_array_assign")
        debug_print("found SetItem: ", stmt)
        lhs = stmt.target
        # Find the source array creation of lhs
        lhs_def = find_array_def(lhs)
        debug_print("found lhs_def: ", lhs_def)
        rhs_def = _get_definition(func_ir, stmt.value)
        debug_print("found rhs_def: ", rhs_def)
        require(isinstance(rhs_def, ir.Expr))
        require(_find_numpy_call(func_ir, rhs_def) == 'empty')
        # Find the array dimension of rhs
        dim_def = _get_definition(func_ir, rhs_def.args[0])
        require(isinstance(dim_def, ir.Expr) and dim_def.op == 'build_tuple')
        debug_print("dim_def = ", dim_def)
        extra_dims = [ _get_definition(func_ir, x, lhs_only=True) for x in dim_def.items ]
        debug_print("extra_dims = ", extra_dims)
        # Expand size tuple when creating lhs_def with extra_dims
        size_tuple_def = _get_definition(func_ir, lhs_def.args[0])
        require(isinstance(size_tuple_def, ir.Expr) and size_tuple_def.op == 'build_tuple')
        debug_print("size_tuple_def = ", size_tuple_def)
        size_tuple_def.items += extra_dims
        # In-place modify rhs_def to be getitem
        rhs_def.op = 'getitem'
        rhs_def.value = _get_definition(func_ir, lhs, lhs_only=True)
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

