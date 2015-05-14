from __future__ import print_function, division, absolute_import

import collections
import dis
import sys

from numba import ir, controlflow, dataflow, utils
from numba.utils import builtins


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
            # destvar will change, invalidate all previously known simplifications
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


class YieldPoint(object):

    def __init__(self, block, inst):
        assert isinstance(block, ir.Block)
        assert isinstance(inst, ir.Yield)
        self.block = block
        self.inst = inst
        self.live_vars = None
        self.weak_live_vars = None


class GeneratorInfo(object):

    def __init__(self):
        # { index: YieldPoint }
        self.yield_points = {}
        # Ordered list of variable names
        self.state_vars = []

    def get_yield_points(self):
        """
        Return an iterable of YieldPoint instances.
        """
        return self.yield_points.values()


class Interpreter(object):
    """A bytecode interpreter that builds up the IR.
    """

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.scopes = []
        self.loc = ir.Loc(filename=bytecode.filename, line=1)
        self.arg_count = bytecode.arg_count
        self.arg_names = bytecode.arg_names
        # Control flow analysis
        self.cfa = controlflow.ControlFlowAnalysis(bytecode)
        self.cfa.run()
        # Data flow analysis
        self.dfa = dataflow.DataFlowAnalysis(self.cfa)
        self.dfa.run()

        global_scope = ir.Scope(parent=None, loc=self.loc)
        self._fill_global_scope(global_scope)
        self.scopes.append(global_scope)

        # { inst offset : ir.Block }
        self.blocks = {}
        # { name: value } of global variables used by the bytecode
        self.used_globals = {}
        # { name: [definitions] } of local variables
        self.definitions = collections.defaultdict(list)
        # { ir.Block: { variable names (potentially) alive at start of block } }
        self.block_entry_vars = {}

        if self.bytecode.is_generator:
            self.generator_info = GeneratorInfo()
        else:
            self.generator_info = None

        # Temp states during interpretation
        self.current_block = None
        self.current_block_offset = None
        self.syntax_blocks = []
        self.dfainfo = None

    def _fill_global_scope(self, scope):
        """TODO
        """
        pass

    def get_used_globals(self):
        """
        Return a dictionary of global variables used by the bytecode.
        """
        return self.used_globals

    def get_block_entry_vars(self, block):
        """
        Return a set of variable names possibly alive at the beginning of
        the block.
        """
        return self.block_entry_vars[block]

    def interpret(self):
        firstblk = min(self.cfa.blocks.keys())
        self.loc = ir.Loc(filename=self.bytecode.filename,
                          line=self.bytecode[firstblk].lineno)
        self.scopes.append(ir.Scope(parent=self.current_scope, loc=self.loc))
        # Interpret loop
        for inst, kws in self._iter_inst():
            self._dispatch(inst, kws)
        # Post-processing and analysis on generated IR
        self._insert_var_dels()
        self._compute_live_variables()
        if self.generator_info:
            self._compute_generator_info()

    def _compute_live_variables(self):
        """
        Compute the live variables at the beginning of each block
        and at each yield point.
        This must be done after insertion of ir.Dels.
        """
        # Scan for variable additions and deletions in each block
        block_var_adds = {}
        block_var_dels = {}
        cfg = self.cfa.graph

        for ir_block in self.blocks.values():
            self.block_entry_vars[ir_block] = set()
            adds = block_var_adds[ir_block] = set()
            dels = block_var_dels[ir_block] = set()
            for stmt in ir_block.body:
                if isinstance(stmt, ir.Assign):
                    name = stmt.target.name
                    adds.add(name)
                    dels.discard(name)
                elif isinstance(stmt, ir.Del):
                    name = stmt.value
                    adds.discard(name)
                    dels.add(name)

        # Propagate defined variables from every block to its successors.
        # (note the entry block automatically gets an empty set)
        changed = True
        while changed:
            # We iterate until the result stabilizes.  This is necessary
            # because of loops in the graph.
            changed = False
            for i in cfg.topo_order():
                ir_block = self.blocks[i]
                v = self.block_entry_vars[ir_block].copy()
                v |= block_var_adds[ir_block]
                v -= block_var_dels[ir_block]
                for j, _ in cfg.successors(i):
                    succ = self.blocks[j]
                    succ_entry_vars = self.block_entry_vars[succ]
                    if not succ_entry_vars >= v:
                        succ_entry_vars.update(v)
                        changed = True

    def _compute_generator_info(self):
        """
        Compute the generator's state variables as the union of live variables
        at all yield points.
        """
        gi = self.generator_info
        for yp in gi.get_yield_points():
            live_vars = set(self.block_entry_vars[yp.block])
            weak_live_vars = set()
            stmts = iter(yp.block.body)
            for stmt in stmts:
                if isinstance(stmt, ir.Assign):
                    if stmt.value is yp.inst:
                        break
                    live_vars.add(stmt.target.name)
                elif isinstance(stmt, ir.Del):
                    live_vars.remove(stmt.value)
            else:
                assert 0, "couldn't find yield point"
            # Try to optimize out any live vars that are deleted immediately
            # after the yield point.
            for stmt in stmts:
                if isinstance(stmt, ir.Del):
                    name = stmt.value
                    if name in live_vars:
                        live_vars.remove(name)
                        weak_live_vars.add(name)
                else:
                    break
            yp.live_vars = live_vars
            yp.weak_live_vars = weak_live_vars

        st = set()
        for yp in gi.get_yield_points():
            st |= yp.live_vars
            st |= yp.weak_live_vars
        gi.state_vars = sorted(st)

    def _insert_var_dels(self):
        """
        Insert ir.Del statements where necessary for the various
        variables defined within the IR.
        """
        # { var name -> { block offset -> stmt of last use } }
        var_use = collections.defaultdict(dict)
        cfg = self.cfa.graph

        for offset, ir_block in self.blocks.items():
            for stmt in ir_block.body:
                for var_name in stmt.list_vars():
                    if stmt.is_terminator and not stmt.is_exit:
                        # Move the "last use" at the beginning of successor
                        # blocks.
                        for succ, _ in cfg.successors(offset):
                            var_use[var_name.name].setdefault(succ, None)
                    else:
                        var_use[var_name.name][offset] = stmt

        for name in sorted(var_use):
            var_use_map = var_use[name]
            blocks = self._compute_var_disposal(name, var_use_map)
            self._patch_var_disposal(name, var_use_map, blocks)

    def _patch_var_disposal(self, var_name, use_map, disposal_blocks):
        """
        Insert ir.Del statements into each of the *disposal_blocks*
        for variable *var_name*, at the appropriate points in the blocks.
        """
        for b in disposal_blocks:
            # Either this is an original use point and we will insert after
            # that statement, or it's a computed one and we will insert
            # at the beginning of the block.
            stmt = use_map.get(b)
            ir_block = self.blocks[b]
            if stmt is None:
                ir_block.prepend(ir.Del(var_name, loc=ir_block.loc))
            else:
                # If the variable is used in an exiting statement
                # (e.g. raise or return), we don't insert anything.
                if not stmt.is_exit:
                    ir_block.insert_after(ir.Del(var_name, loc=stmt.loc), stmt)

    def _compute_var_disposal(self, var_name, use_map):
        """
        Compute a set of blocks in which *var_name* should be disposed of.
        """
        cfg = self.cfa.graph

        # Compute a set of blocks where we want the variable to be
        # kept alive.
        live_points = set()

        # Find the loops the variable is used in
        loops = set()
        # Each enclosing loop encloses *all* use points
        enclosing_loops = None
        for block in use_map:
            block_loops = cfg.in_loops(block)
            loops.update(block_loops)
            if enclosing_loops is None:
                enclosing_loops = set(block_loops)
            else:
                enclosing_loops.intersection_update(block_loops)

        spanned_loops = loops - enclosing_loops

        for loop in spanned_loops:
            # As there are blocks outside the loop, the variable must be
            # kept alive accross loop iterations => we only consider
            # loop exit points.
            # (otherwise it means the variable is *defined* inside the loop
            #  and need not be kept alive accross iterations)
            live_points.update(loop.exits)

        for block in use_map:
            live_points.add(block)

        # Now compute a set of *exclusive* blocks covering all paths
        # stemming from the live points. This ensures that each variable
        # is del'ed at most once per possible code path.
        tails = set()
        # We use the todo list as a stack, so we really iterate in
        # reverse topo order.
        todo = list(cfg.topo_sort(live_points))
        seen = set()
        infinite_loops = set()
        while todo:
            b = todo.pop()
            if b in seen:
                continue
            seen.add(b)
            block_loops = cfg.in_loops(b)
            if block_loops:
                loop = block_loops[0]
                if loop not in enclosing_loops:
                    if not loop.exits:
                        # Fix-up for infinite loops: we temporarily add
                        # the loop header to the tails, so that the variable
                        # isn't del'ed before the loop.
                        tails.add(loop.header)
                        infinite_loops.add(loop.header)
                    continue
            if cfg.descendents(b) & tails:
                for succ, _ in cfg.successors(b):
                    if succ not in tails:
                        todo.append(succ)
            else:
                tails.add(b)

        # Remove temporary entries for infinite loops
        tails -= infinite_loops

        return tails

    def init_first_block(self):
        # Define variables receiving the function arguments
        for index, name in enumerate(self.arg_names):
            val = ir.Arg(index=index, name=name, loc=self.loc)
            self.store(val, name)

    def _iter_inst(self):
        for blkct, block in enumerate(self.cfa.iterliveblocks()):
            firstinst = self.bytecode[block.body[0]]
            self._start_new_block(firstinst)
            if blkct == 0:
                # Is first block
                self.init_first_block()
            for offset, kws in self.dfainfo.insts:
                inst = self.bytecode[offset]
                self.loc = ir.Loc(filename=self.bytecode.filename,
                                  line=inst.lineno)
                yield inst, kws
            self._end_current_block()

    def _start_new_block(self, inst):
        self.loc = ir.Loc(filename=self.bytecode.filename, line=inst.lineno)
        oldblock = self.current_block
        self.insert_block(inst.offset)
        # Ensure the last block is terminated
        if oldblock is not None and not oldblock.is_terminated:
            jmp = ir.Jump(inst.offset, loc=self.loc)
            oldblock.append(jmp)
        # Get DFA block info
        self.dfainfo = self.dfa.infos[self.current_block_offset]
        self.assigner = Assigner()

    def _end_current_block(self):
        self._remove_unused_temporaries()
        self._insert_outgoing_phis()

    def _remove_unused_temporaries(self):
        """
        Remove assignments to unused temporary variables from the
        current block.
        """
        new_body = []
        for inst in self.current_block.body:
            if (isinstance(inst, ir.Assign)
                and inst.target.is_temp
                and inst.target.name in self.assigner.unused_dests):
                continue
            new_body.append(inst)
        self.current_block.body = new_body

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
            return utils.get_function_globals(self.bytecode.func)[name]
        except KeyError:
            return getattr(builtins, name, ir.UNDEFINED)

    def get_closure_value(self, index):
        """
        Get a value from the cell contained in this function's closure.
        """
        return self.bytecode.func.__closure__[index].cell_contents

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
            except ir.NotDefinedError as e:
                if e.loc is None:
                    e.loc = self.loc
                raise e

    def dump(self, file=None):
        file = file or sys.stdout
        for offset, block in sorted(self.blocks.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file=file)

    def dump_generator_info(self, file=None):
        file = file or sys.stdout
        gi = self.generator_info
        print("generator state variables:", sorted(gi.state_vars), file=file)
        for index, yp in sorted(gi.yield_points.items()):
            print("yield point #%d: live variables = %s, weak live variables = %s"
                  % (index, sorted(yp.live_vars), sorted(yp.weak_live_vars)),
                  file=file)


    # --- Scope operations ---

    def store(self, value, name, redefine=False):
        """
        Store *value* (a Expr or Var instance) into the variable named *name*
        (a str object).
        """
        if redefine or self.current_block_offset in self.cfa.backbone:
            target = self.current_scope.redefine(name, loc=self.loc)
        else:
            target = self.current_scope.get_or_define(name, loc=self.loc)
        if isinstance(value, ir.Var):
            value = self.assigner.assign(value, target)
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        self.current_block.append(stmt)
        self.definitions[name].append(value)

    def get(self, name):
        """
        Get the variable (a Var instance) with the given *name*.
        """
        # Try to simplify the variable lookup by returning an earlier
        # variable assigned to *name*.
        var = self.assigner.get_assignment_source(name)
        if var is None:
            var = self.current_scope.get(name)
        return var

    def get_definition(self, value):
        """
        Get the definition site for the given Var instance.
        A Expr instance is returned.
        """
        while True:
            if not isinstance(value, ir.Var):
                return value
            defs = self.definitions[value.name]
            if len(defs) == 0:
                raise KeyError("no definition for %r"
                               % (value.name,))
            if len(defs) > 1:
                raise KeyError("more than one definition for %r"
                               % (value.name,))
            value = defs[0]

    # --- Block operations ---

    def insert_block(self, offset, scope=None, loc=None):
        scope = scope or self.current_scope
        loc = loc or self.loc
        blk = ir.Block(scope=scope, loc=loc)
        self.blocks[offset] = blk
        self.current_block = blk
        self.current_block_offset = offset
        return blk

    def block_constains_opname(self, offset, opname):
        for offset in self.cfa.blocks[offset]:
            inst = self.bytecode[offset]
            if inst.opname == opname:
                return True
        return False

    # --- Bytecode handlers ---

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
                                          index=i, loc=self.loc)
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

    def op_LOAD_FAST(self, inst, res):
        srcname = self.code_locals[inst.arg]
        self.store(value=self.get(srcname), name=res)

    def op_STORE_FAST(self, inst, value):
        dstname = self.code_locals[inst.arg]
        value = self.get(value)
        self.store(value=value, name=dstname)

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
        const = ir.Const(value, loc=self.loc)
        self.store(const, res)

    def op_LOAD_GLOBAL(self, inst, res):
        name = self.code_names[inst.arg]
        value = self.get_global_value(name)
        self.used_globals[name] = value
        gl = ir.Global(name, value, loc=self.loc)
        self.store(gl, res)

    def op_LOAD_DEREF(self, inst, res):
        name = self.code_freevars[inst.arg]
        value = self.get_closure_value(inst.arg)
        gl = ir.FreeVar(inst.arg, name, value, loc=self.loc)
        self.store(gl, res)

    def op_SETUP_LOOP(self, inst):
        assert self.blocks[inst.offset] is self.current_block
        loop = ir.Loop(inst.offset, exit=(inst.next + inst.arg))
        self.syntax_blocks.append(loop)

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

    def op_BUILD_MAP(self, inst, size, res):
        expr = ir.Expr.build_map(size=size, loc=self.loc)
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
        lhs = self.get(lhs)
        rhs = self.get(rhs)
        expr = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
        self.store(expr, res)

    def _inplace_binop(self, op, lhs, rhs, res):
        lhs = self.get(lhs)
        rhs = self.get(rhs)
        expr = ir.Expr.inplace_binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
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

    def op_POP_BLOCK(self, inst):
        self.syntax_blocks.pop()

    def op_RETURN_VALUE(self, inst, retval, castval):
        self.store(ir.Expr.cast(self.get(retval), loc=self.loc), castval)
        ret = ir.Return(self.get(castval), loc=self.loc)
        self.current_block.append(ret)

    def op_COMPARE_OP(self, inst, lhs, rhs, res):
        op = dis.cmp_op[inst.arg]
        self._binop(op, lhs, rhs, res)

    def op_BREAK_LOOP(self, inst):
        loop = self.syntax_blocks[-1]
        assert isinstance(loop, ir.Loop)
        jmp = ir.Jump(target=loop.exit, loc=self.loc)
        self.current_block.append(jmp)

    def _op_JUMP_IF(self, inst, pred, iftrue):
        brs = {
            True: inst.get_jump_target(),
            False: inst.next,
        }
        truebr = brs[iftrue]
        falsebr = brs[not iftrue]
        bra = ir.Branch(cond=self.get(pred), truebr=truebr, falsebr=falsebr,
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

    def op_RAISE_VARARGS(self, inst, exc):
        if exc is not None:
            exc = self.get(exc)
        stmt = ir.Raise(exception=exc, loc=self.loc)
        self.current_block.append(stmt)

    def op_YIELD_VALUE(self, inst, value, res):
        dct = self.generator_info.yield_points
        index = len(dct) + 1
        inst = ir.Yield(value=self.get(value), index=index, loc=self.loc)
        yp = YieldPoint(self.current_block, inst)
        dct[index] = yp
        return self.store(inst, res)
