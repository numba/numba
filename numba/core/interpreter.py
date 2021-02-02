import builtins
import collections
import dis
import operator
import logging

from numba.core import errors, dataflow, controlflow, ir, config
from numba.core.errors import NotDefinedError, error_extras
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
                              INPLACE_BINOPS_TO_OPERATORS,)
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple


class _UNKNOWN_VALUE(object):
    """Represents an unknown value, this is for ease of debugging purposes only.
    """

    def __init__(self, varname):
        self._varname = varname

    def __repr__(self):
        return "_UNKNOWN_VALUE({})".format(self._varname)


_logger = logging.getLogger(__name__)


class Assigner(object):
    """
    This object keeps track of potential assignment simplifications
    inside a code block.
    For example `$O.1 = x` followed by `y = $0.1` can be simplified
    into `y = x`, but it's not possible anymore if we have `x = z`
    in-between those two instructions.

    NOTE: this is not only an optimization, but is actually necessary
    due to certain limitations of Numba - such as only accepting the
    returning of an array passed as function argument.
    """

    def __init__(self):
        # { destination variable name -> source Var object }
        self.dest_to_src = {}
        # Basically a reverse mapping of dest_to_src:
        # { source variable name -> all destination names in dest_to_src }
        self.src_invalidate = collections.defaultdict(list)
        self.unused_dests = set()

    def assign(self, srcvar, destvar):
        """
        Assign *srcvar* to *destvar*. Return either *srcvar* or a possible
        simplified assignment source (earlier assigned to *srcvar*).
        """
        srcname = srcvar.name
        destname = destvar.name
        if destname in self.src_invalidate:
            # destvar will change, invalidate all previously known
            # simplifications
            for d in self.src_invalidate.pop(destname):
                self.dest_to_src.pop(d)
        if srcname in self.dest_to_src:
            srcvar = self.dest_to_src[srcname]
        if destvar.is_temp:
            self.dest_to_src[destname] = srcvar
            self.src_invalidate[srcname].append(destname)
            self.unused_dests.add(destname)
        return srcvar

    def get_assignment_source(self, destname):
        """
        Get a possible assignment source (a ir.Var instance) to replace
        *destname*, otherwise None.
        """
        if destname in self.dest_to_src:
            return self.dest_to_src[destname]
        self.unused_dests.discard(destname)
        return None


def peep_hole_list_to_tuple(func_ir):
    """
    This peephole rewrites a bytecode sequence new to Python 3.9 that looks
    like e.g.:

    def foo(a):
        return (*a,)

    41          0 BUILD_LIST               0
                2 LOAD_FAST                0 (a)
                4 LIST_EXTEND              1
                6 LIST_TO_TUPLE
                8 RETURN_VAL

    essentially, the unpacking of tuples is written as a list which is appended
    to/extended and then "magicked" into a tuple by the new LIST_TO_TUPLE
    opcode.

    This peephole repeatedly analyses the bytecode in a block looking for a
    window between a `LIST_TO_TUPLE` and `BUILD_LIST` and...

    1. Turns the BUILD_LIST into a BUILD_TUPLE
    2. Sets an accumulator's initial value as the target of the BUILD_TUPLE
    3. Searches for 'extend' on the original list and turns these into binary
       additions on the accumulator.
    4. Searches for 'append' on the original list and turns these into a
       `BUILD_TUPLE` which is then appended via binary addition to the
       accumulator.
    5. Assigns the accumulator to the variable that exits the peephole and the
       rest of the block/code refers to as the result of the unpack operation.
    6. Patches up
    """
    _DEBUG = False

    # For all blocks
    for offset, blk in func_ir.blocks.items():
        # keep doing the peephole rewrite until nothing is left that matches
        while True:
            # first try and find a matching region
            # i.e. BUILD_LIST...<stuff>...LIST_TO_TUPLE
            def find_postive_region():
                found = False
                for idx in reversed(range(len(blk.body))):
                    stmt = blk.body[idx]
                    if isinstance(stmt, ir.Assign):
                        value = stmt.value
                        if (isinstance(value, ir.Expr) and
                                value.op == 'list_to_tuple'):
                            target_list = value.info[0]
                            found = True
                            bt = (idx, stmt)
                    if found:
                        if isinstance(stmt, ir.Assign):
                            if stmt.target.name == target_list:
                                region = (bt, (idx, stmt))
                                return region

            region = find_postive_region()
            # if there's a peep hole region then do something with it
            if region is not None:
                peep_hole = blk.body[region[1][0] : region[0][0]]
                if _DEBUG:
                    print("\nWINDOW:")
                    for x in peep_hole:
                        print(x)
                    print("")

                appends = []
                extends = []
                init = region[1][1]
                const_list = init.target.name
                # Walk through the peep_hole and find things that are being
                # "extend"ed and "append"ed to the BUILD_LIST
                for x in peep_hole:
                    if isinstance(x, ir.Assign):
                        if isinstance(x.value, ir.Expr):
                            expr = x.value
                            if (expr.op == 'getattr' and
                                    expr.value.name == const_list):
                                # it's not strictly necessary to split out
                                # extends and appends, but it helps with
                                # debugging to do so!
                                if expr.attr == 'extend':
                                    extends.append(x.target.name)
                                elif expr.attr == 'append':
                                    appends.append(x.target.name)
                                else:
                                    assert 0
                # go back through the peep hole build new IR based on it.
                new_hole = []

                def append_and_fix(x):
                    """ Adds to the new_hole and fixes up definitions"""
                    new_hole.append(x)
                    if x.target.name in func_ir._definitions:
                        # if there's already a definition, drop it, should only
                        # be 1 as the way cpython emits the sequence for
                        # `list_to_tuple` should ensure this.
                        assert len(func_ir._definitions[x.target.name]) == 1
                        func_ir._definitions[x.target.name].clear()
                    func_ir._definitions[x.target.name].append(x.value)

                the_build_list = init.target

                # Do the transform on the peep hole
                if _DEBUG:
                    print("\nBLOCK:")
                    blk.dump()

                # This section basically accumulates list appends and extends
                # as binop(+) on tuples, it drops all the getattr() for extend
                # and append as they are now dead and replaced with binop(+).
                # It also switches out the build_list for a build_tuple and then
                # ensures everything is wired up and defined ok.
                t2l_agn = region[0][1]
                acc = the_build_list
                for x in peep_hole:
                    if isinstance(x, ir.Assign):
                        if isinstance(x.value, ir.Expr):
                            expr = x.value
                            if expr.op == 'getattr':
                                if (x.target.name in extends or
                                        x.target.name in appends):
                                    # drop definition, it's being wholesale
                                    # replaced.
                                    func_ir._definitions.pop(x.target.name)
                                    continue
                                else:
                                    # a getattr on something we're not
                                    # interested in
                                    new_hole.append(x)
                            elif expr.op == 'call':
                                fname = expr.func.name
                                if fname in extends or fname in appends:
                                    arg = expr.args[0]
                                    if isinstance(arg, ir.Var):
                                        tmp_name = "%s_var_%s" % (fname,
                                                                  arg.name)
                                        if fname in appends:
                                            bt = ir.Expr.build_tuple([arg,],
                                                                     expr.loc)
                                        else:
                                            bt = arg
                                        var = ir.Var(arg.scope, tmp_name,
                                                     expr.loc)
                                        asgn = ir.Assign(bt, var, expr.loc)
                                        append_and_fix(asgn)
                                        arg = var

                                    # this needs to be a binary add
                                    new = ir.Expr.binop(fn=operator.add,
                                                        lhs=acc,
                                                        rhs=arg,
                                                        loc=x.loc)
                                    asgn = ir.Assign(new, x.target, expr.loc)
                                    append_and_fix(asgn)
                                    acc = asgn.target
                                else:
                                    # there could be a call in the unpack, like
                                    # *(a, x.append(y))
                                    new_hole.append(x)
                            elif (expr.op == 'build_list' and
                                    x.target.name == const_list):
                                new = ir.Expr.build_tuple(expr.items, expr.loc)
                                asgn = ir.Assign(new, x.target, expr.loc)
                                # Not a temporary any more
                                append_and_fix(asgn)
                            else:
                                new_hole.append(x)
                        else:
                            new_hole.append(x)

                    else:
                        # stick everything else in as-is
                        new_hole.append(x)
                # Finally write the result back into the original build list as
                # everything refers to it.
                new_hole.append(ir.Assign(acc, t2l_agn.target,
                                          the_build_list.loc))
                if _DEBUG:
                    print("\nNEW HOLE:")
                    for x in new_hole:
                        print(x)

                # and then update the block body with the modified region
                cpy = blk.body[:]
                head = cpy[:region[1][0]]
                tail = blk.body[region[0][0] + 1:]
                tmp = head + new_hole + tail
                blk.body.clear()
                blk.body.extend(tmp)

                if _DEBUG:
                    print("\nDUMP post hole:")
                    blk.dump()

            else:
                # else escape
                break

    return func_ir


def peep_hole_delete_with_exit(func_ir):
    """
    This rewrite removes variables used to store the `__exit__` function
    loaded by SETUP_WITH.
    """
    dead_vars = set()

    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            # Any statement that uses a variable with the '$setup_with_exitfn'
            # prefix is considered dead.
            used = set(stmt.list_vars())
            for v in used:
                if v.name.startswith('$setup_with_exitfn'):
                    dead_vars.add(v)
            # Any assignment that uses any of the dead variable is considered
            # dead.
            if used & dead_vars:
                if isinstance(stmt, ir.Assign):
                    dead_vars.add(stmt.target)

        new_body = []
        for stmt in blk.body:
            # Skip any statements that uses anyone of the dead variable.
            if not (set(stmt.list_vars()) & dead_vars):
                new_body.append(stmt)
        blk.body.clear()
        blk.body.extend(new_body)

    return func_ir


class Interpreter(object):
    """A bytecode interpreter that builds up the IR.
    """

    def __init__(self, func_id):
        self.func_id = func_id
        self.arg_count = func_id.arg_count
        self.arg_names = func_id.arg_names
        self.loc = self.first_loc = ir.Loc.from_function_id(func_id)
        self.is_generator = func_id.is_generator

        # { inst offset : ir.Block }
        self.blocks = {}
        # { name: [definitions] } of local variables
        self.definitions = collections.defaultdict(list)
        # A set to keep track of all exception variables.
        # To be used in _legalize_exception_vars()
        self._exception_vars = set()

    def interpret(self, bytecode):
        """
        Generate IR for this bytecode.
        """
        self.bytecode = bytecode

        self.scopes = []
        global_scope = ir.Scope(parent=None, loc=self.loc)
        self.scopes.append(global_scope)

        if PYVERSION < (3, 7):
            # Control flow analysis
            self.cfa = controlflow.ControlFlowAnalysis(bytecode)
            self.cfa.run()
            if config.DUMP_CFG:
                self.cfa.dump()

            # Data flow analysis
            self.dfa = dataflow.DataFlowAnalysis(self.cfa)
            self.dfa.run()
        else:
            flow = Flow(bytecode)
            flow.run()
            self.dfa = AdaptDFA(flow)
            self.cfa = AdaptCFA(flow)
            if config.DUMP_CFG:
                self.cfa.dump()

        # Temp states during interpretation
        self.current_block = None
        self.current_block_offset = None
        self.syntax_blocks = []
        self.dfainfo = None

        self.scopes.append(ir.Scope(parent=self.current_scope, loc=self.loc))
        # Interpret loop
        for inst, kws in self._iter_inst():
            self._dispatch(inst, kws)
        self._legalize_exception_vars()
        # Prepare FunctionIR
        func_ir = ir.FunctionIR(self.blocks, self.is_generator, self.func_id,
                                self.first_loc, self.definitions,
                                self.arg_count, self.arg_names)
        _logger.debug(func_ir.dump_to_string())

        # post process the IR to rewrite opcodes/byte sequences that are too
        # involved to risk handling as part of direct interpretation
        peepholes = []
        if PYVERSION == (3, 9):
            peepholes.append(peep_hole_list_to_tuple)
        peepholes.append(peep_hole_delete_with_exit)

        post_processed_ir = self.post_process(peepholes, func_ir)
        return post_processed_ir

    def post_process(self, peepholes, func_ir):
        for peep in peepholes:
            func_ir = peep(func_ir)
        return func_ir

    def _legalize_exception_vars(self):
        """Search for unsupported use of exception variables.
        Note, they cannot be stored into user variable.
        """
        # Build a set of exception variables
        excvars = self._exception_vars.copy()
        # Propagate the exception variables to LHS of assignment
        for varname, defnvars in self.definitions.items():
            for v in defnvars:
                if isinstance(v, ir.Var):
                    k = v.name
                    if k in excvars:
                        excvars.add(varname)
        # Filter out the user variables.
        uservar = list(filter(lambda x: not x.startswith('$'), excvars))
        if uservar:
            # Complain about the first user-variable storing an exception
            first = uservar[0]
            loc = self.current_scope.get(first).loc
            msg = "Exception object cannot be stored into variable ({})."
            raise errors.UnsupportedError(msg.format(first), loc=loc)

    def init_first_block(self):
        # Define variables receiving the function arguments
        for index, name in enumerate(self.arg_names):
            val = ir.Arg(index=index, name=name, loc=self.loc)
            self.store(val, name)

    def _iter_inst(self):
        for blkct, block in enumerate(self.cfa.iterliveblocks()):
            firstinst = self.bytecode[block.offset]
            self.loc = self.loc.with_lineno(firstinst.lineno)
            self._start_new_block(block.offset)
            if blkct == 0:
                # Is first block
                self.init_first_block()
            for offset, kws in self.dfainfo.insts:
                inst = self.bytecode[offset]
                self.loc = self.loc.with_lineno(inst.lineno)
                yield inst, kws
            self._end_current_block()

    def _start_new_block(self, offset):
        oldblock = self.current_block
        self.insert_block(offset)
        # Ensure the last block is terminated
        if oldblock is not None and not oldblock.is_terminated:
            # Handle ending try block.
            tryblk = self.dfainfo.active_try_block
            # If there's an active try-block and the handler block is live.
            if tryblk is not None and tryblk['end'] in self.cfa.graph.nodes():
                # We are in a try-block, insert a branch to except-block.
                # This logic cannot be in self._end_current_block()
                # because we the non-raising next block-offset.
                branch = ir.Branch(
                    cond=self.get('$exception_check'),
                    truebr=tryblk['end'],
                    falsebr=offset,
                    loc=self.loc,
                )
                oldblock.append(branch)
            # Handle normal case
            else:
                jmp = ir.Jump(offset, loc=self.loc)
                oldblock.append(jmp)
        # Get DFA block info
        self.dfainfo = self.dfa.infos[self.current_block_offset]
        self.assigner = Assigner()
        # Check out-of-scope syntactic-block
        while self.syntax_blocks:
            if offset >= self.syntax_blocks[-1].exit:
                self.syntax_blocks.pop()
            else:
                break

    def _end_current_block(self):
        # Handle try block
        if not self.current_block.is_terminated:
            tryblk = self.dfainfo.active_try_block
            if tryblk is not None:
                self._insert_exception_check()
        # Handle normal block cleanup
        self._remove_unused_temporaries()
        self._insert_outgoing_phis()

    def _inject_call(self, func, gv_name, res_name=None):
        """A helper function to inject a call to *func* which is a python
        function.
        Parameters
        ----------
        func : callable
            The function object to be called.
        gv_name : str
            The variable name to be used to store the function object.
        res_name : str; optional
            The variable name to be used to store the call result.
            If ``None``, a name is created automatically.
        """
        gv_fn = ir.Global(gv_name, func, loc=self.loc)
        self.store(value=gv_fn, name=gv_name, redefine=True)
        callres = ir.Expr.call(self.get(gv_name), (), (), loc=self.loc)
        res_name = res_name or '$callres_{}'.format(gv_name)
        self.store(value=callres, name=res_name, redefine=True)

    def _insert_try_block_begin(self):
        """Insert IR-nodes to mark the start of a `try` block.
        """
        self._inject_call(eh.mark_try_block, 'mark_try_block')

    def _insert_try_block_end(self):
        """Insert IR-nodes to mark the end of a `try` block.
        """
        self._inject_call(eh.end_try_block, 'end_try_block')

    def _insert_exception_variables(self):
        """Insert IR-nodes to initialize the exception variables.
        """
        tryblk = self.dfainfo.active_try_block
        # Get exception variables
        endblk = tryblk['end']
        edgepushed = self.dfainfo.outgoing_edgepushed.get(endblk)
        # Note: the last value on the stack is the exception value
        # Note: due to the current limitation, all exception variables are None
        if edgepushed:
            const_none = ir.Const(value=None, loc=self.loc)
            # For each variable going to the handler block.
            for var in edgepushed:
                if var in self.definitions:
                    raise AssertionError(
                        "exception variable CANNOT be defined by other code",
                    )
                self.store(value=const_none, name=var)
                self._exception_vars.add(var)

    def _insert_exception_check(self):
        """Called before the end of a block to inject checks if raised.
        """
        self._insert_exception_variables()
        # Do exception check
        self._inject_call(eh.exception_check, 'exception_check',
                          '$exception_check')

    def _remove_unused_temporaries(self):
        """
        Remove assignments to unused temporary variables from the
        current block.
        """
        new_body = []
        replaced_var = {}
        for inst in self.current_block.body:
            # the same temporary is assigned to multiple variables in cases
            # like a = b[i] = 1, so need to handle replaced temporaries in
            # later setitem/setattr nodes
            if (isinstance(inst, (ir.SetItem, ir.SetAttr))
                    and inst.value.name in replaced_var):
                inst.value = replaced_var[inst.value.name]
            elif isinstance(inst, ir.Assign):
                if (inst.target.is_temp
                        and inst.target.name in self.assigner.unused_dests):
                    continue
                # the same temporary is assigned to multiple variables in cases
                # like a = b = 1, so need to handle replaced temporaries in
                # later assignments
                if (isinstance(inst.value, ir.Var)
                        and inst.value.name in replaced_var):
                    inst.value = replaced_var[inst.value.name]
                    new_body.append(inst)
                    continue
                # chained unpack cases may reuse temporary
                # e.g. a = (b, c) = (x, y)
                if (isinstance(inst.value, ir.Expr)
                        and inst.value.op == "exhaust_iter"
                        and inst.value.value.name in replaced_var):
                    inst.value.value = replaced_var[inst.value.value.name]
                    new_body.append(inst)
                    continue
                # eliminate temporary variables that are assigned to user
                # variables right after creation. E.g.:
                # $1 = f(); a = $1 -> a = f()
                # the temporary variable is not reused elsewhere since CPython
                # bytecode is stack-based and this pattern corresponds to a pop
                if (isinstance(inst.value, ir.Var) and inst.value.is_temp
                        and new_body and isinstance(new_body[-1], ir.Assign)):
                    prev_assign = new_body[-1]
                    # _var_used_in_binop check makes sure we don't create a new
                    # inplace binop operation which can fail
                    # (see TestFunctionType.test_in_iter_func_call)
                    if (prev_assign.target.name == inst.value.name
                            and not self._var_used_in_binop(
                                inst.target.name, prev_assign.value)):
                        replaced_var[inst.value.name] = inst.target
                        prev_assign.target = inst.target
                        # replace temp var definition in target with proper defs
                        self.definitions[inst.target.name].remove(inst.value)
                        self.definitions[inst.target.name].extend(
                            self.definitions.pop(inst.value.name)
                        )
                        continue

            new_body.append(inst)

        self.current_block.body = new_body

    def _var_used_in_binop(self, varname, expr):
        """return True if 'expr' is a binary expression and 'varname' is used
        in it as an argument
        """
        return (isinstance(expr, ir.Expr)
                and expr.op in ("binop", "inplace_binop")
                and (varname == expr.lhs.name or varname == expr.rhs.name))

    def _insert_outgoing_phis(self):
        """
        Add assignments to forward requested outgoing values
        to subsequent blocks.
        """
        for phiname, varname in self.dfainfo.outgoing_phis.items():
            target = self.current_scope.get_or_define(phiname,
                                                      loc=self.loc)
            stmt = ir.Assign(value=self.get(varname), target=target,
                             loc=self.loc)
            self.definitions[target.name].append(stmt.value)
            if not self.current_block.is_terminated:
                self.current_block.append(stmt)
            else:
                self.current_block.insert_before_terminator(stmt)

    def get_global_value(self, name):
        """
        Get a global value from the func_global (first) or
        as a builtins (second).  If both failed, return a ir.UNDEFINED.
        """
        try:
            return self.func_id.func.__globals__[name]
        except KeyError:
            return getattr(builtins, name, ir.UNDEFINED)

    def get_closure_value(self, index):
        """
        Get a value from the cell contained in this function's closure.
        If not set, return a ir.UNDEFINED.
        """
        cell = self.func_id.func.__closure__[index]
        try:
            return cell.cell_contents
        except ValueError:
            return ir.UNDEFINED

    @property
    def current_scope(self):
        return self.scopes[-1]

    @property
    def code_consts(self):
        return self.bytecode.co_consts

    @property
    def code_locals(self):
        return self.bytecode.co_varnames

    @property
    def code_names(self):
        return self.bytecode.co_names

    @property
    def code_cellvars(self):
        return self.bytecode.co_cellvars

    @property
    def code_freevars(self):
        return self.bytecode.co_freevars

    def _dispatch(self, inst, kws):
        assert self.current_block is not None
        fname = "op_%s" % inst.opname.replace('+', '_')
        try:
            fn = getattr(self, fname)
        except AttributeError:
            raise NotImplementedError(inst)
        else:
            try:
                return fn(inst, **kws)
            except errors.NotDefinedError as e:
                if e.loc is None:
                    loc = self.loc
                else:
                    loc = e.loc

                err = errors.NotDefinedError(e.name, loc=loc)
                if not config.FULL_TRACEBACKS:
                    raise err from None
                else:
                    raise err

    # --- Scope operations ---

    def store(self, value, name, redefine=False):
        """
        Store *value* (a Expr or Var instance) into the variable named *name*
        (a str object). Returns the target variable.
        """
        if redefine or self.current_block_offset in self.cfa.backbone:
            rename = not (name in self.code_cellvars)
            target = self.current_scope.redefine(name, loc=self.loc,
                                                 rename=rename)
        else:
            target = self.current_scope.get_or_define(name, loc=self.loc)
        if isinstance(value, ir.Var):
            value = self.assigner.assign(value, target)
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        self.current_block.append(stmt)
        self.definitions[target.name].append(value)
        return target

    def get(self, name):
        """
        Get the variable (a Var instance) with the given *name*.
        """
        # Implicit argument for comprehension starts with '.'
        # See Parameter class in inspect.py (from Python source)
        if name[0] == '.' and name[1:].isdigit():
            name = 'implicit{}'.format(name[1:])

        # Try to simplify the variable lookup by returning an earlier
        # variable assigned to *name*.
        var = self.assigner.get_assignment_source(name)
        if var is None:
            var = self.current_scope.get(name)
        return var

    # --- Block operations ---

    def insert_block(self, offset, scope=None, loc=None):
        scope = scope or self.current_scope
        loc = loc or self.loc
        blk = ir.Block(scope=scope, loc=loc)
        self.blocks[offset] = blk
        self.current_block = blk
        self.current_block_offset = offset
        return blk

    # --- Bytecode handlers ---

    def op_NOP(self, inst):
        pass

    def op_PRINT_ITEM(self, inst, item, printvar, res):
        item = self.get(item)
        printgv = ir.Global("print", print, loc=self.loc)
        self.store(value=printgv, name=printvar)
        call = ir.Expr.call(self.get(printvar), (item,), (), loc=self.loc)
        self.store(value=call, name=res)

    def op_PRINT_NEWLINE(self, inst, printvar, res):
        printgv = ir.Global("print", print, loc=self.loc)
        self.store(value=printgv, name=printvar)
        call = ir.Expr.call(self.get(printvar), (), (), loc=self.loc)
        self.store(value=call, name=res)

    def op_UNPACK_SEQUENCE(self, inst, iterable, stores, tupleobj):
        count = len(stores)
        # Exhaust the iterable into a tuple-like object
        tup = ir.Expr.exhaust_iter(value=self.get(iterable), loc=self.loc,
                                   count=count)
        self.store(name=tupleobj, value=tup)

        # then index the tuple-like object to extract the values
        for i, st in enumerate(stores):
            expr = ir.Expr.static_getitem(self.get(tupleobj),
                                          index=i, index_var=None,
                                          loc=self.loc)
            self.store(expr, st)

    def op_BUILD_SLICE(self, inst, start, stop, step, res, slicevar):
        start = self.get(start)
        stop = self.get(stop)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        if step is None:
            sliceinst = ir.Expr.call(self.get(slicevar), (start, stop), (),
                                     loc=self.loc)
        else:
            step = self.get(step)
            sliceinst = ir.Expr.call(self.get(slicevar), (start, stop, step),
                                     (), loc=self.loc)
        self.store(value=sliceinst, name=res)

    def op_SLICE_0(self, inst, base, res, slicevar, indexvar, nonevar):
        base = self.get(base)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        index = ir.Expr.call(self.get(slicevar), (none, none), (), loc=self.loc)
        self.store(value=index, name=indexvar)

        expr = ir.Expr.getitem(base, self.get(indexvar), loc=self.loc)
        self.store(value=expr, name=res)

    def op_SLICE_1(self, inst, base, start, nonevar, res, slicevar, indexvar):
        base = self.get(base)
        start = self.get(start)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (start, none), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        expr = ir.Expr.getitem(base, self.get(indexvar), loc=self.loc)
        self.store(value=expr, name=res)

    def op_SLICE_2(self, inst, base, nonevar, stop, res, slicevar, indexvar):
        base = self.get(base)
        stop = self.get(stop)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (none, stop,), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        expr = ir.Expr.getitem(base, self.get(indexvar), loc=self.loc)
        self.store(value=expr, name=res)

    def op_SLICE_3(self, inst, base, start, stop, res, slicevar, indexvar):
        base = self.get(base)
        start = self.get(start)
        stop = self.get(stop)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (start, stop), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        expr = ir.Expr.getitem(base, self.get(indexvar), loc=self.loc)
        self.store(value=expr, name=res)

    def op_STORE_SLICE_0(self, inst, base, value, slicevar, indexvar, nonevar):
        base = self.get(base)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        index = ir.Expr.call(self.get(slicevar), (none, none), (), loc=self.loc)
        self.store(value=index, name=indexvar)

        stmt = ir.SetItem(base, self.get(indexvar), self.get(value),
                          loc=self.loc)
        self.current_block.append(stmt)

    def op_STORE_SLICE_1(self, inst, base, start, nonevar, value, slicevar,
                         indexvar):
        base = self.get(base)
        start = self.get(start)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (start, none), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        stmt = ir.SetItem(base, self.get(indexvar), self.get(value),
                          loc=self.loc)
        self.current_block.append(stmt)

    def op_STORE_SLICE_2(self, inst, base, nonevar, stop, value, slicevar,
                         indexvar):
        base = self.get(base)
        stop = self.get(stop)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (none, stop,), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        stmt = ir.SetItem(base, self.get(indexvar), self.get(value),
                          loc=self.loc)
        self.current_block.append(stmt)

    def op_STORE_SLICE_3(self, inst, base, start, stop, value, slicevar,
                         indexvar):
        base = self.get(base)
        start = self.get(start)
        stop = self.get(stop)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (start, stop), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)
        stmt = ir.SetItem(base, self.get(indexvar), self.get(value),
                          loc=self.loc)
        self.current_block.append(stmt)

    def op_DELETE_SLICE_0(self, inst, base, slicevar, indexvar, nonevar):
        base = self.get(base)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        index = ir.Expr.call(self.get(slicevar), (none, none), (), loc=self.loc)
        self.store(value=index, name=indexvar)

        stmt = ir.DelItem(base, self.get(indexvar), loc=self.loc)
        self.current_block.append(stmt)

    def op_DELETE_SLICE_1(self, inst, base, start, nonevar, slicevar, indexvar):
        base = self.get(base)
        start = self.get(start)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (start, none), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        stmt = ir.DelItem(base, self.get(indexvar), loc=self.loc)
        self.current_block.append(stmt)

    def op_DELETE_SLICE_2(self, inst, base, nonevar, stop, slicevar, indexvar):
        base = self.get(base)
        stop = self.get(stop)

        nonegv = ir.Const(None, loc=self.loc)
        self.store(value=nonegv, name=nonevar)
        none = self.get(nonevar)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (none, stop,), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)

        stmt = ir.DelItem(base, self.get(indexvar), loc=self.loc)
        self.current_block.append(stmt)

    def op_DELETE_SLICE_3(self, inst, base, start, stop, slicevar, indexvar):
        base = self.get(base)
        start = self.get(start)
        stop = self.get(stop)

        slicegv = ir.Global("slice", slice, loc=self.loc)
        self.store(value=slicegv, name=slicevar)

        index = ir.Expr.call(self.get(slicevar), (start, stop), (),
                             loc=self.loc)
        self.store(value=index, name=indexvar)
        stmt = ir.DelItem(base, self.get(indexvar), loc=self.loc)
        self.current_block.append(stmt)

    def op_LOAD_FAST(self, inst, res):
        srcname = self.code_locals[inst.arg]
        self.store(value=self.get(srcname), name=res)

    def op_STORE_FAST(self, inst, value):
        dstname = self.code_locals[inst.arg]
        value = self.get(value)
        self.store(value=value, name=dstname)

    def op_DELETE_FAST(self, inst):
        dstname = self.code_locals[inst.arg]
        self.current_block.append(ir.Del(dstname, loc=self.loc))

    def op_DUP_TOPX(self, inst, orig, duped):
        for src, dst in zip(orig, duped):
            self.store(value=self.get(src), name=dst)

    op_DUP_TOP = op_DUP_TOPX
    op_DUP_TOP_TWO = op_DUP_TOPX

    def op_STORE_ATTR(self, inst, target, value):
        attr = self.code_names[inst.arg]
        sa = ir.SetAttr(target=self.get(target), value=self.get(value),
                        attr=attr, loc=self.loc)
        self.current_block.append(sa)

    def op_DELETE_ATTR(self, inst, target):
        attr = self.code_names[inst.arg]
        sa = ir.DelAttr(target=self.get(target), attr=attr, loc=self.loc)
        self.current_block.append(sa)

    def op_LOAD_ATTR(self, inst, item, res):
        item = self.get(item)
        attr = self.code_names[inst.arg]
        getattr = ir.Expr.getattr(item, attr, loc=self.loc)
        self.store(getattr, res)

    def op_LOAD_CONST(self, inst, res):
        value = self.code_consts[inst.arg]
        if isinstance(value, tuple):
            st = []
            for x in value:
                nm = '$const_%s' % str(x)
                val_const = ir.Const(x, loc=self.loc)
                target = self.store(val_const, name=nm, redefine=True)
                st.append(target)
            const = ir.Expr.build_tuple(st, loc=self.loc)
        elif isinstance(value, frozenset):
            st = []
            for x in value:
                nm = '$const_%s' % str(x)
                val_const = ir.Const(x, loc=self.loc)
                target = self.store(val_const, name=nm, redefine=True)
                st.append(target)
            const = ir.Expr.build_set(st, loc=self.loc)
        else:
            const = ir.Const(value, loc=self.loc)
        self.store(const, res)

    def op_LOAD_GLOBAL(self, inst, res):
        name = self.code_names[inst.arg]
        value = self.get_global_value(name)
        gl = ir.Global(name, value, loc=self.loc)
        self.store(gl, res)

    def op_LOAD_DEREF(self, inst, res):
        n_cellvars = len(self.code_cellvars)
        if inst.arg < n_cellvars:
            name = self.code_cellvars[inst.arg]
            gl = self.get(name)
        else:
            idx = inst.arg - n_cellvars
            name = self.code_freevars[idx]
            value = self.get_closure_value(idx)
            gl = ir.FreeVar(idx, name, value, loc=self.loc)
        self.store(gl, res)

    def op_STORE_DEREF(self, inst, value):
        n_cellvars = len(self.code_cellvars)
        if inst.arg < n_cellvars:
            dstname = self.code_cellvars[inst.arg]
        else:
            dstname = self.code_freevars[inst.arg - n_cellvars]
        value = self.get(value)
        self.store(value=value, name=dstname)

    def op_SETUP_LOOP(self, inst):
        assert self.blocks[inst.offset] is self.current_block
        loop = ir.Loop(inst.offset, exit=(inst.next + inst.arg))
        self.syntax_blocks.append(loop)

    def op_SETUP_WITH(self, inst, contextmanager, exitfn=None):
        assert self.blocks[inst.offset] is self.current_block
        # Handle with
        exitpt = inst.next + inst.arg
        wth = ir.With(inst.offset, exit=exitpt)
        self.syntax_blocks.append(wth)
        ctxmgr = self.get(contextmanager)
        self.current_block.append(ir.EnterWith(contextmanager=ctxmgr,
                                               begin=inst.offset,
                                               end=exitpt, loc=self.loc,))
        # exitfn is None in py3.6
        if exitfn is not None:
            # Store exit fn
            exit_fn_obj = ir.Const(None, loc=self.loc)
            self.store(value=exit_fn_obj, name=exitfn)

    def op_SETUP_EXCEPT(self, inst):
        # Removed since python3.8
        self._insert_try_block_begin()

    def op_SETUP_FINALLY(self, inst):
        self._insert_try_block_begin()

    def op_WITH_CLEANUP(self, inst):
        "no-op"

    def op_WITH_CLEANUP_START(self, inst):
        "no-op"

    def op_WITH_CLEANUP_FINISH(self, inst):
        "no-op"

    def op_END_FINALLY(self, inst):
        "no-op"

    def op_BEGIN_FINALLY(self, inst, temps):
        # The *temps* are the exception variables
        const_none = ir.Const(None, loc=self.loc)
        for tmp in temps:
            # Set to None for now
            self.store(const_none, name=tmp)
            self._exception_vars.add(tmp)

    if PYVERSION < (3, 6):

        def op_CALL_FUNCTION(self, inst, func, args, kws, res, vararg):
            func = self.get(func)
            args = [self.get(x) for x in args]
            if vararg is not None:
                vararg = self.get(vararg)

            # Process keywords
            keyvalues = []
            removethese = []
            for k, v in kws:
                k, v = self.get(k), self.get(v)
                for inst in self.current_block.body:
                    if isinstance(inst, ir.Assign) and inst.target is k:
                        removethese.append(inst)
                        keyvalues.append((inst.value.value, v))

            # Remove keyword constant statements
            for inst in removethese:
                self.current_block.remove(inst)

            expr = ir.Expr.call(func, args, keyvalues, loc=self.loc,
                                vararg=vararg)
            self.store(expr, res)

        op_CALL_FUNCTION_VAR = op_CALL_FUNCTION
    else:
        def op_CALL_FUNCTION(self, inst, func, args, res):
            func = self.get(func)
            args = [self.get(x) for x in args]
            expr = ir.Expr.call(func, args, (), loc=self.loc)
            self.store(expr, res)

        def op_CALL_FUNCTION_KW(self, inst, func, args, names, res):
            func = self.get(func)
            args = [self.get(x) for x in args]
            # Find names const
            names = self.get(names)
            for inst in self.current_block.body:
                if isinstance(inst, ir.Assign) and inst.target is names:
                    self.current_block.remove(inst)
                    # scan up the block looking for the values, remove them
                    # and find their name strings
                    named_items = []
                    for x in inst.value.items:
                        for y in self.current_block.body[::-1]:
                            if x == y.target:
                                self.current_block.remove(y)
                                named_items.append(y.value.value)
                                break
                    keys = named_items
                    break

            nkeys = len(keys)
            posvals = args[:-nkeys]
            kwvals = args[-nkeys:]
            keyvalues = list(zip(keys, kwvals))

            expr = ir.Expr.call(func, posvals, keyvalues, loc=self.loc)
            self.store(expr, res)

        def op_CALL_FUNCTION_EX(self, inst, func, vararg, res):
            func = self.get(func)
            vararg = self.get(vararg)
            expr = ir.Expr.call(func, [], [], loc=self.loc, vararg=vararg)
            self.store(expr, res)

    def _build_tuple_unpack(self, inst, tuples, temps, is_assign):
        first = self.get(tuples[0])
        if is_assign:
            # it's assign-like, defer handling to an intrinsic that will have
            # type information.
            # Can deal with tuples only, i.e. y = (*x,). where x = <tuple>
            gv_name = "unpack_single_tuple"
            gv_fn = ir.Global(gv_name, unpack_single_tuple, loc=self.loc,)
            self.store(value=gv_fn, name=gv_name, redefine=True)
            exc = ir.Expr.call(self.get(gv_name), args=(first,), kws=(),
                               loc=self.loc,)
            self.store(exc, temps[0])
        else:
            for other, tmp in zip(map(self.get, tuples[1:]), temps):
                out = ir.Expr.binop(fn=operator.add, lhs=first, rhs=other,
                                    loc=self.loc)
                self.store(out, tmp)
                first = self.get(tmp)

    def op_BUILD_TUPLE_UNPACK_WITH_CALL(self, inst, tuples, temps, is_assign):
        # just unpack the input tuple, call inst will be handled afterwards
        self._build_tuple_unpack(inst, tuples, temps, is_assign)

    def op_BUILD_TUPLE_UNPACK(self, inst, tuples, temps, is_assign):
        self._build_tuple_unpack(inst, tuples, temps, is_assign)

    def op_LIST_TO_TUPLE(self, inst, const_list, res):
        expr = ir.Expr.dummy('list_to_tuple', (const_list,), loc=self.loc)
        self.store(expr, res)

    def op_BUILD_CONST_KEY_MAP(self, inst, keys, keytmps, values, res):
        # Unpack the constant key-tuple and reused build_map which takes
        # a sequence of (key, value) pair.
        keyvar = self.get(keys)
        # TODO: refactor this pattern. occurred several times.
        for inst in self.current_block.body:
            if isinstance(inst, ir.Assign) and inst.target is keyvar:
                self.current_block.remove(inst)
                # scan up the block looking for the values, remove them
                # and find their name strings
                named_items = []
                for x in inst.value.items:
                    for y in self.current_block.body[::-1]:
                        if x == y.target:
                            self.current_block.remove(y)
                            named_items.append(y.value.value)
                            break
                keytup = named_items
                break
        assert len(keytup) == len(values)
        keyconsts = [ir.Const(value=x, loc=self.loc) for x in keytup]
        for kval, tmp in zip(keyconsts, keytmps):
            self.store(kval, tmp)
        items = list(zip(map(self.get, keytmps), map(self.get, values)))

        # sort out literal values
        literal_items = []
        for v in values:
            defns = self.definitions[v]
            if len(defns) != 1:
                break
            defn = defns[0]
            if not isinstance(defn, ir.Const):
                break
            literal_items.append(defn.value)

        def resolve_const(v):
            defns = self.definitions[v]
            if len(defns) != 1:
                return _UNKNOWN_VALUE(self.get(v).name)
            defn = defns[0]
            if not isinstance(defn, ir.Const):
                return _UNKNOWN_VALUE(self.get(v).name)
            return defn.value

        if len(literal_items) != len(values):
            literal_dict = {x: resolve_const(y) for x, y in
                            zip(keytup, values)}
        else:
            literal_dict = {x:y for x, y in zip(keytup, literal_items)}

        # to deal with things like {'a': 1, 'a': 'cat', 'b': 2, 'a': 2j}
        # store the index of the actual used value for a given key, this is
        # used when lowering to pull the right value out into the tuple repr
        # of a mixed value type dictionary.
        value_indexes = {}
        for i, k in enumerate(keytup):
            value_indexes[k] = i

        expr = ir.Expr.build_map(items=items,
                                 size=2,
                                 literal_value=literal_dict,
                                 value_indexes=value_indexes,
                                 loc=self.loc)

        self.store(expr, res)

    def op_GET_ITER(self, inst, value, res):
        expr = ir.Expr.getiter(value=self.get(value), loc=self.loc)
        self.store(expr, res)

    def op_FOR_ITER(self, inst, iterator, pair, indval, pred):
        """
        Assign new block other this instruction.
        """
        assert inst.offset in self.blocks, "FOR_ITER must be block head"

        # Emit code
        val = self.get(iterator)

        pairval = ir.Expr.iternext(value=val, loc=self.loc)
        self.store(pairval, pair)

        iternext = ir.Expr.pair_first(value=self.get(pair), loc=self.loc)
        self.store(iternext, indval)

        isvalid = ir.Expr.pair_second(value=self.get(pair), loc=self.loc)
        self.store(isvalid, pred)

        # Conditional jump
        br = ir.Branch(cond=self.get(pred), truebr=inst.next,
                       falsebr=inst.get_jump_target(),
                       loc=self.loc)
        self.current_block.append(br)

    def op_BINARY_SUBSCR(self, inst, target, index, res):
        index = self.get(index)
        target = self.get(target)
        expr = ir.Expr.getitem(target, index=index, loc=self.loc)
        self.store(expr, res)

    def op_STORE_SUBSCR(self, inst, target, index, value):
        index = self.get(index)
        target = self.get(target)
        value = self.get(value)
        stmt = ir.SetItem(target=target, index=index, value=value,
                          loc=self.loc)
        self.current_block.append(stmt)

    def op_DELETE_SUBSCR(self, inst, target, index):
        index = self.get(index)
        target = self.get(target)
        stmt = ir.DelItem(target=target, index=index, loc=self.loc)
        self.current_block.append(stmt)

    def op_BUILD_TUPLE(self, inst, items, res):
        expr = ir.Expr.build_tuple(items=[self.get(x) for x in items],
                                   loc=self.loc)
        self.store(expr, res)

    def op_BUILD_LIST(self, inst, items, res):
        expr = ir.Expr.build_list(items=[self.get(x) for x in items],
                                  loc=self.loc)
        self.store(expr, res)

    def op_BUILD_SET(self, inst, items, res):
        expr = ir.Expr.build_set(items=[self.get(x) for x in items],
                                 loc=self.loc)
        self.store(expr, res)

    def op_SET_UPDATE(self, inst, target, value, updatevar, res):
        target = self.get(target)
        value = self.get(value)
        updateattr = ir.Expr.getattr(target, 'update', loc=self.loc)
        self.store(value=updateattr, name=updatevar)
        updateinst = ir.Expr.call(self.get(updatevar), (value,), (),
                                  loc=self.loc)
        self.store(value=updateinst, name=res)

    def op_BUILD_MAP(self, inst, items, size, res):
        got_items = [(self.get(k), self.get(v)) for k, v in items]

        # sort out literal values, this is a bit contrived but is to handle
        # situations like `{1: 10, 1: 10}` where the size of the literal dict
        # is smaller than the definition
        def get_literals(target):
            literal_items = []
            values = [self.get(v.name) for v in target]
            for v in values:
                defns = self.definitions[v.name]
                if len(defns) != 1:
                    break
                defn = defns[0]
                if not isinstance(defn, ir.Const):
                    break
                literal_items.append(defn.value)
            return literal_items

        literal_keys = get_literals(x[0] for x in got_items)
        literal_values = get_literals(x[1] for x in got_items)

        has_literal_keys = len(literal_keys) == len(got_items)
        has_literal_values = len(literal_values) == len(got_items)

        value_indexes = {}
        if not has_literal_keys and not has_literal_values:
            literal_dict = None
        elif has_literal_keys and not has_literal_values:
            literal_dict = {x: _UNKNOWN_VALUE(y[1]) for x, y in
                            zip(literal_keys, got_items)}
            for i, k in enumerate(literal_keys):
                value_indexes[k] = i
        else:
            literal_dict = {x: y for x, y in zip(literal_keys, literal_values)}
            for i, k in enumerate(literal_keys):
                value_indexes[k] = i

        expr = ir.Expr.build_map(items=got_items, size=size,
                                 literal_value=literal_dict,
                                 value_indexes=value_indexes,
                                 loc=self.loc)
        self.store(expr, res)

    def op_STORE_MAP(self, inst, dct, key, value):
        stmt = ir.StoreMap(dct=self.get(dct), key=self.get(key),
                           value=self.get(value), loc=self.loc)
        self.current_block.append(stmt)

    def op_UNARY_NEGATIVE(self, inst, value, res):
        value = self.get(value)
        expr = ir.Expr.unary('-', value=value, loc=self.loc)
        return self.store(expr, res)

    def op_UNARY_POSITIVE(self, inst, value, res):
        value = self.get(value)
        expr = ir.Expr.unary('+', value=value, loc=self.loc)
        return self.store(expr, res)

    def op_UNARY_INVERT(self, inst, value, res):
        value = self.get(value)
        expr = ir.Expr.unary('~', value=value, loc=self.loc)
        return self.store(expr, res)

    def op_UNARY_NOT(self, inst, value, res):
        value = self.get(value)
        expr = ir.Expr.unary('not', value=value, loc=self.loc)
        return self.store(expr, res)

    def _binop(self, op, lhs, rhs, res):
        op = BINOPS_TO_OPERATORS[op]
        lhs = self.get(lhs)
        rhs = self.get(rhs)
        expr = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
        self.store(expr, res)

    def _inplace_binop(self, op, lhs, rhs, res):
        immuop = BINOPS_TO_OPERATORS[op]
        op = INPLACE_BINOPS_TO_OPERATORS[op + '=']
        lhs = self.get(lhs)
        rhs = self.get(rhs)
        expr = ir.Expr.inplace_binop(op, immuop, lhs=lhs, rhs=rhs,
                                     loc=self.loc)
        self.store(expr, res)

    def op_BINARY_ADD(self, inst, lhs, rhs, res):
        self._binop('+', lhs, rhs, res)

    def op_BINARY_SUBTRACT(self, inst, lhs, rhs, res):
        self._binop('-', lhs, rhs, res)

    def op_BINARY_MULTIPLY(self, inst, lhs, rhs, res):
        self._binop('*', lhs, rhs, res)

    def op_BINARY_DIVIDE(self, inst, lhs, rhs, res):
        self._binop('/?', lhs, rhs, res)

    def op_BINARY_TRUE_DIVIDE(self, inst, lhs, rhs, res):
        self._binop('/', lhs, rhs, res)

    def op_BINARY_FLOOR_DIVIDE(self, inst, lhs, rhs, res):
        self._binop('//', lhs, rhs, res)

    def op_BINARY_MODULO(self, inst, lhs, rhs, res):
        self._binop('%', lhs, rhs, res)

    def op_BINARY_POWER(self, inst, lhs, rhs, res):
        self._binop('**', lhs, rhs, res)

    def op_BINARY_MATRIX_MULTIPLY(self, inst, lhs, rhs, res):
        self._binop('@', lhs, rhs, res)

    def op_BINARY_LSHIFT(self, inst, lhs, rhs, res):
        self._binop('<<', lhs, rhs, res)

    def op_BINARY_RSHIFT(self, inst, lhs, rhs, res):
        self._binop('>>', lhs, rhs, res)

    def op_BINARY_AND(self, inst, lhs, rhs, res):
        self._binop('&', lhs, rhs, res)

    def op_BINARY_OR(self, inst, lhs, rhs, res):
        self._binop('|', lhs, rhs, res)

    def op_BINARY_XOR(self, inst, lhs, rhs, res):
        self._binop('^', lhs, rhs, res)

    def op_INPLACE_ADD(self, inst, lhs, rhs, res):
        self._inplace_binop('+', lhs, rhs, res)

    def op_INPLACE_SUBTRACT(self, inst, lhs, rhs, res):
        self._inplace_binop('-', lhs, rhs, res)

    def op_INPLACE_MULTIPLY(self, inst, lhs, rhs, res):
        self._inplace_binop('*', lhs, rhs, res)

    def op_INPLACE_DIVIDE(self, inst, lhs, rhs, res):
        self._inplace_binop('/?', lhs, rhs, res)

    def op_INPLACE_TRUE_DIVIDE(self, inst, lhs, rhs, res):
        self._inplace_binop('/', lhs, rhs, res)

    def op_INPLACE_FLOOR_DIVIDE(self, inst, lhs, rhs, res):
        self._inplace_binop('//', lhs, rhs, res)

    def op_INPLACE_MODULO(self, inst, lhs, rhs, res):
        self._inplace_binop('%', lhs, rhs, res)

    def op_INPLACE_POWER(self, inst, lhs, rhs, res):
        self._inplace_binop('**', lhs, rhs, res)

    def op_INPLACE_MATRIX_MULTIPLY(self, inst, lhs, rhs, res):
        self._inplace_binop('@', lhs, rhs, res)

    def op_INPLACE_LSHIFT(self, inst, lhs, rhs, res):
        self._inplace_binop('<<', lhs, rhs, res)

    def op_INPLACE_RSHIFT(self, inst, lhs, rhs, res):
        self._inplace_binop('>>', lhs, rhs, res)

    def op_INPLACE_AND(self, inst, lhs, rhs, res):
        self._inplace_binop('&', lhs, rhs, res)

    def op_INPLACE_OR(self, inst, lhs, rhs, res):
        self._inplace_binop('|', lhs, rhs, res)

    def op_INPLACE_XOR(self, inst, lhs, rhs, res):
        self._inplace_binop('^', lhs, rhs, res)

    def op_JUMP_ABSOLUTE(self, inst):
        jmp = ir.Jump(inst.get_jump_target(), loc=self.loc)
        self.current_block.append(jmp)

    def op_JUMP_FORWARD(self, inst):
        jmp = ir.Jump(inst.get_jump_target(), loc=self.loc)
        self.current_block.append(jmp)

    def op_POP_BLOCK(self, inst, kind=None):
        if kind is None:
            self.syntax_blocks.pop()
        elif kind == 'try':
            self._insert_try_block_end()

    def op_RETURN_VALUE(self, inst, retval, castval):
        self.store(ir.Expr.cast(self.get(retval), loc=self.loc), castval)
        ret = ir.Return(self.get(castval), loc=self.loc)
        self.current_block.append(ret)

    def op_COMPARE_OP(self, inst, lhs, rhs, res):
        op = dis.cmp_op[inst.arg]
        if op == 'in' or op == 'not in':
            lhs, rhs = rhs, lhs

        if op == 'not in':
            self._binop('in', lhs, rhs, res)
            tmp = self.get(res)
            out = ir.Expr.unary('not', value=tmp, loc=self.loc)
            self.store(out, res)
        elif op == 'exception match':
            gv_fn = ir.Global(
                "exception_match", eh.exception_match, loc=self.loc,
            )
            exc_match_name = '$exc_match'
            self.store(value=gv_fn, name=exc_match_name, redefine=True)
            lhs = self.get(lhs)
            rhs = self.get(rhs)
            exc = ir.Expr.call(
                self.get(exc_match_name), args=(lhs, rhs), kws=(), loc=self.loc,
            )
            self.store(exc, res)
        else:
            self._binop(op, lhs, rhs, res)

    def op_IS_OP(self, inst, lhs, rhs, res):
        # invert if op case is 1
        op = 'is not' if inst.arg == 1 else 'is'
        self._binop(op, lhs, rhs, res)

    def op_CONTAINS_OP(self, inst, lhs, rhs, res):
        lhs, rhs = rhs, lhs
        self._binop('in', lhs, rhs, res)
        # invert if op case is 1
        if inst.arg == 1:
            tmp = self.get(res)
            out = ir.Expr.unary('not', value=tmp, loc=self.loc)
            self.store(out, res)

    def op_BREAK_LOOP(self, inst, end=None):
        if end is None:
            loop = self.syntax_blocks[-1]
            assert isinstance(loop, ir.Loop)
            end = loop.exit
        jmp = ir.Jump(target=end, loc=self.loc)
        self.current_block.append(jmp)

    def _op_JUMP_IF(self, inst, pred, iftrue):
        brs = {
            True: inst.get_jump_target(),
            False: inst.next,
        }
        truebr = brs[iftrue]
        falsebr = brs[not iftrue]

        name = "bool%s" % (inst.offset)
        gv_fn = ir.Global("bool", bool, loc=self.loc)
        self.store(value=gv_fn, name=name)

        callres = ir.Expr.call(self.get(name), (self.get(pred),), (),
                               loc=self.loc)

        pname = "$%spred" % (inst.offset)
        predicate = self.store(value=callres, name=pname)
        bra = ir.Branch(cond=predicate, truebr=truebr, falsebr=falsebr,
                        loc=self.loc)
        self.current_block.append(bra)

    def op_JUMP_IF_FALSE(self, inst, pred):
        self._op_JUMP_IF(inst, pred=pred, iftrue=False)

    def op_JUMP_IF_TRUE(self, inst, pred):
        self._op_JUMP_IF(inst, pred=pred, iftrue=True)

    def op_POP_JUMP_IF_FALSE(self, inst, pred):
        self._op_JUMP_IF(inst, pred=pred, iftrue=False)

    def op_POP_JUMP_IF_TRUE(self, inst, pred):
        self._op_JUMP_IF(inst, pred=pred, iftrue=True)

    def op_JUMP_IF_FALSE_OR_POP(self, inst, pred):
        self._op_JUMP_IF(inst, pred=pred, iftrue=False)

    def op_JUMP_IF_TRUE_OR_POP(self, inst, pred):
        self._op_JUMP_IF(inst, pred=pred, iftrue=True)

    def op_JUMP_IF_NOT_EXC_MATCH(self, inst, pred, tos, tos1):
        truebr = inst.next
        falsebr = inst.get_jump_target()
        gv_fn = ir.Global(
            "exception_match", eh.exception_match, loc=self.loc,
        )
        exc_match_name = '$exc_match'
        self.store(value=gv_fn, name=exc_match_name, redefine=True)
        lhs = self.get(tos1)
        rhs = self.get(tos)
        exc = ir.Expr.call(
            self.get(exc_match_name), args=(lhs, rhs), kws=(), loc=self.loc,
        )
        predicate = self.store(exc, pred)
        bra = ir.Branch(cond=predicate, truebr=truebr, falsebr=falsebr,
                        loc=self.loc)
        self.current_block.append(bra)

    def op_RERAISE(self, inst, exc):
        # Numba can't handle this case and it's caught else where, this is a
        # runtime guard in case this is reached by unknown means.
        msg = (f"Unreachable condition reached (op code RERAISE executed)"
               f"{error_extras['reportable']}")
        stmt = ir.StaticRaise(AssertionError, (msg,), self.loc)
        self.current_block.append(stmt)

    def op_RAISE_VARARGS(self, inst, exc):
        if exc is not None:
            exc = self.get(exc)
        tryblk = self.dfainfo.active_try_block
        if tryblk is not None:
            # In a try block
            stmt = ir.TryRaise(exception=exc, loc=self.loc)
            self.current_block.append(stmt)
            self._insert_try_block_end()
            self.current_block.append(ir.Jump(tryblk['end'], loc=self.loc))
        else:
            # Not in a try block
            stmt = ir.Raise(exception=exc, loc=self.loc)
            self.current_block.append(stmt)

    def op_YIELD_VALUE(self, inst, value, res):
        # initialize index to None.  it's being set later in post-processing
        index = None
        inst = ir.Yield(value=self.get(value), index=index, loc=self.loc)
        return self.store(inst, res)

    def op_MAKE_FUNCTION(self, inst, name, code, closure, annotations,
                         kwdefaults, defaults, res):
        if annotations is not None:
            msg = "op_MAKE_FUNCTION with annotations is not implemented"
            raise NotImplementedError(msg)
        if kwdefaults is not None:
            msg = "op_MAKE_FUNCTION with kwdefaults is not implemented"
            raise NotImplementedError(msg)
        if defaults:
            if isinstance(defaults, tuple):
                defaults = tuple([self.get(name) for name in defaults])
            else:
                defaults = self.get(defaults)

        assume_code_const = self.definitions[code][0]
        if not isinstance(assume_code_const, ir.Const):
            msg = (
                "Unsupported use of closure. "
                "Probably caused by complex control-flow constructs; "
                "e.g. try-except"
            )
            raise errors.UnsupportedError(msg, loc=self.loc)
        fcode = assume_code_const.value
        if name:
            name = self.get(name)
        if closure:
            closure = self.get(closure)
        expr = ir.Expr.make_function(name, fcode, closure, defaults, self.loc)
        self.store(expr, res)

    def op_MAKE_CLOSURE(self, inst, name, code, closure, annotations,
                        kwdefaults, defaults, res):
        self.op_MAKE_FUNCTION(inst, name, code, closure, annotations,
                              kwdefaults, defaults, res)

    def op_LOAD_CLOSURE(self, inst, res):
        n_cellvars = len(self.code_cellvars)
        if inst.arg < n_cellvars:
            name = self.code_cellvars[inst.arg]
            try:
                gl = self.get(name)
            except NotDefinedError:
                msg = "Unsupported use of op_LOAD_CLOSURE encountered"
                raise NotImplementedError(msg)
        else:
            idx = inst.arg - n_cellvars
            name = self.code_freevars[idx]
            value = self.get_closure_value(idx)
            gl = ir.FreeVar(idx, name, value, loc=self.loc)
        self.store(gl, res)

    def op_LIST_APPEND(self, inst, target, value, appendvar, res):
        target = self.get(target)
        value = self.get(value)
        appendattr = ir.Expr.getattr(target, 'append', loc=self.loc)
        self.store(value=appendattr, name=appendvar)
        appendinst = ir.Expr.call(self.get(appendvar), (value,), (),
                                  loc=self.loc)
        self.store(value=appendinst, name=res)

    def op_LIST_EXTEND(self, inst, target, value, extendvar, res):
        target = self.get(target)
        value = self.get(value)
        # If the statements between the current instruction and the target
        # are N * consts followed by build_tuple AND the target has no items,
        # it's a situation where a list is being statically initialised, rewrite
        # the build_tuple as a build_list, drop the extend, and wire up the
        # target as the result from the build_tuple that's been rewritten.

        # See if this is the first statement in a block, if so its probably from
        # control flow in a tuple unpack like:
        # `(*(1, (2,) if predicate else (3,)))`
        # this cannot be handled as present so raise
        if not self.current_block.body:
            msg = ("An unsupported bytecode sequence has been encountered: "
                   "op_LIST_EXTEND at the start of a block.\n\nThis could be "
                   "due to the use of a branch in a tuple unpacking statement.")
            raise errors.UnsupportedError(msg)

        # is last emitted statement a build_tuple?
        stmt = self.current_block.body[-1]
        ok = isinstance(stmt.value, ir.Expr) and stmt.value.op == "build_tuple"
        # check statements from self.current_block.body[-1] through to target,
        # make sure they are consts
        build_empty_list = None
        if ok:
            for stmt in reversed(self.current_block.body[:-1]):
                if not isinstance(stmt, ir.Assign):
                    ok = False
                    break
                # if its not a const, it needs to be the `build_list` for the
                # target, else it's something else we don't know about so just
                # bail
                if isinstance(stmt.value, ir.Const):
                    continue

                # it's not a const, check for target
                elif isinstance(stmt.value, ir.Expr) and stmt.target == target:
                    build_empty_list = stmt
                    # it's only ok to do this if the target has no initializer
                    # already
                    ok = not stmt.value.items
                    break
                else:
                    ok = False
                    break
        if ok:
            stmts = self.current_block.body
            build_tuple_asgn = self.current_block.body[-1]
            # move build list to last issued statement
            stmts.append(stmts.pop(stmts.index(build_empty_list)))
            # fix the build list
            build_tuple = build_tuple_asgn.value
            build_list = build_empty_list.value
            build_list.items = build_tuple.items
        else:
            # it's just a list extend with no static init, let it be
            extendattr = ir.Expr.getattr(target, 'extend', loc=self.loc)
            self.store(value=extendattr, name=extendvar)
            extendinst = ir.Expr.call(self.get(extendvar), (value,), (),
                                      loc=self.loc)
            self.store(value=extendinst, name=res)

    def op_LOAD_ASSERTION_ERROR(self, inst, res):
        gv_fn = ir.Global("AssertionError", AssertionError, loc=self.loc)
        self.store(value=gv_fn, name=res)

    # NOTE: The LOAD_METHOD opcode is implemented as a LOAD_ATTR for ease,
    # however this means a new object (the bound-method instance) could be
    # created. Conversely, using a pure LOAD_METHOD no intermediary is present
    # and it is essentially like a pointer grab and forward to CALL_METHOD. The
    # net outcome is that the implementation in Numba produces the same result,
    # but in object mode it may be that it runs more slowly than it would if
    # run in CPython.

    def op_LOAD_METHOD(self, *args, **kws):
        self.op_LOAD_ATTR(*args, **kws)

    def op_CALL_METHOD(self, *args, **kws):
        self.op_CALL_FUNCTION(*args, **kws)
