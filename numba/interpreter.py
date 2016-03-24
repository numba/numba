from __future__ import print_function, division, absolute_import

import collections
import dis
from functools import reduce
import operator
import sys

from . import ir, controlflow, dataflow, utils, errors, consts
from .utils import builtins


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
        self.loc = ir.Loc(filename=bytecode.filename, line=1)
        self.arg_count = bytecode.arg_count
        self.arg_names = bytecode.arg_names

        # { inst offset : ir.Block }
        self.blocks = {}
        # { name: value } of global variables used by the bytecode
        self.used_globals = {}
        # { name: [definitions] } of local variables
        self.definitions = collections.defaultdict(list)
        # { ir.Block: { variable names (potentially) alive at start of block } }
        self.block_entry_vars = {}

        self.reset()

    def reset(self):
        """
        Reset all internal state and release resources.
        """
        self.scopes = []
        global_scope = ir.Scope(parent=None, loc=self.loc)
        self.scopes.append(global_scope)

        # Control flow analysis
        self.cfa = controlflow.ControlFlowAnalysis(self.bytecode)
        self.cfa.run()
        # Data flow analysis
        self.dfa = dataflow.DataFlowAnalysis(self.cfa)
        self.dfa.run()
        # Constant inference
        self.consts = consts.ConstantInference(self)

        if self.bytecode.is_generator:
            self.generator_info = GeneratorInfo()
        else:
            self.generator_info = None

        # Temp states during interpretation
        self.current_block = None
        self.current_block_offset = None
        self.syntax_blocks = []
        self.dfainfo = None

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

    def infer_constant(self, name):
        """
        Try to infer the constant value of a given variable.
        """
        if isinstance(name, ir.Var):
            name = name.name
        return self.consts.infer_constant(name)

    def interpret(self):
        """
        Generate IR for this bytecode.
        """
        firstblk = min(self.cfa.blocks.keys())
        self.loc = ir.Loc(filename=self.bytecode.filename,
                          line=self.bytecode[firstblk].lineno)
        self.scopes.append(ir.Scope(parent=self.current_scope, loc=self.loc))
        # Interpret loop
        for inst, kws in self._iter_inst():
            self._dispatch(inst, kws)
        # Post-processing and analysis on generated IR
        var_def_map, var_dead_map = self._insert_var_dels()
        self._compute_live_variables(var_def_map, var_dead_map)
        if self.generator_info:
            self._compute_generator_info()

    def _compute_live_variables(self, var_def_map, var_dead_map):
        """
        Compute the live variables at the beginning of each block
        and at each yield point.
        The ``var_def_map`` and ``var_dead_map`` indicates the variable defined
        and deleted at each block, respectively.
        """
        # live var at the entry per block
        block_entry_vars = collections.defaultdict(set)

        def fix_point_progress():
            return tuple(map(len, block_entry_vars.values()))

        cfg = self.cfa.graph
        old_point = None
        new_point = fix_point_progress()

        # Propagate defined variables and still live the successors.
        # (note the entry block automatically gets an empty set)

        # Note: This is finding the actual available variables at the entry
        #       of each block. The algorithm in _compute_live_map() is finding
        #       the variable that must be available at the entry of each block.
        #       This is top-down in the dataflow.  The other one is bottom-up.
        while old_point != new_point:
            # We iterate until the result stabilizes.  This is necessary
            # because of loops in the graphself.
            for offset in self.blocks:
                # vars available + variable defined
                avail = block_entry_vars[offset] | var_def_map[offset]
                # substract variables deleted
                avail -= var_dead_map[offset]
                # add ``avail`` to each successors
                for succ, _data in cfg.successors(offset):
                    block_entry_vars[succ] |= avail

            old_point = new_point
            new_point = fix_point_progress()

        for offset, ir_block in self.blocks.items():
            self.block_entry_vars[ir_block] = block_entry_vars[offset]

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
        Insert del statements for each variable.
        Returns a 2-tuple of (variable definition map, variable deletion map)
        which indicates variables defined and deleted in each block.

        The algorithm avoids relying on explicit knowledge on loops and
        distinguish between variables that are defined locally vs variables that
        come from incoming blocks.
        We start with simple usage (variable reference) and definition (variable
        creation) maps on each block. Propagate the liveness info to predecessor
        blocks until it stabilize, at which point we know which variables must
        exist before entering each block. Then, we compute the end of variable
        lives and insert del statements accordingly. Variables are deleted after
        the last use. Variable referenced by terminators (e.g. conditional
        branch and return) are deleted by the successors or the caller.
        """
        var_use_map, var_def_map = self._compute_use_defs()
        live_map = self._compute_live_map(var_use_map, var_def_map)
        dead_maps = self._compute_dead_maps(live_map, var_def_map)
        internal_dead_map, escaping_dead_map = dead_maps
        self._patch_var_dels(internal_dead_map, escaping_dead_map)
        var_dead_map = dict((k, internal_dead_map[k] | escaping_dead_map[k])
                            for k in self.blocks)
        return var_def_map, var_dead_map

    def _compute_use_defs(self):
        """
        Find variable use/def per block.
        """
        var_use_map = {}   # { block offset -> set of vars }
        var_def_map = {}   # { block offset -> set of vars }
        for offset, ir_block in self.blocks.items():
            var_use_map[offset] = use_set = set()
            var_def_map[offset] = def_set = set()
            for stmt in ir_block.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Inst):
                        rhs_set = set(var.name
                                      for var in stmt.value.list_vars())
                    elif isinstance(stmt.value, ir.Var):
                        rhs_set = set([stmt.value.name])
                    elif isinstance(stmt.value, (ir.Arg, ir.Const, ir.Global,
                                                 ir.FreeVar)):
                        rhs_set = ()
                    else:
                        raise AssertionError('unreachable', type(stmt.value))
                    # If lhs not in rhs of the assignment
                    if stmt.target.name not in rhs_set:
                        def_set.add(stmt.target.name)

                for var in stmt.list_vars():
                    # do not include locally defined vars to use-map
                    if var.name not in def_set:
                        use_set.add(var.name)

        return var_use_map, var_def_map

    def _compute_live_map(self, var_use_map, var_def_map):
        """
        Find variables that must be alive at the ENTRY of each block.
        We use a simple fix-point algorithm that iterates until the set of
        live variables is unchanged for each block.
        """
        live_map = {}
        for offset in self.blocks.keys():
            live_map[offset] = var_use_map[offset]

        def fix_point_progress():
            return tuple(len(v) for v in live_map.values())

        old_point = None
        new_point = fix_point_progress()
        while old_point != new_point:
            for offset in live_map.keys():
                for inc_blk, _data in self.cfa.graph.predecessors(offset):
                    # substract all variables that are defined in
                    # the incoming block
                    live_map[inc_blk] |= live_map[offset] - var_def_map[inc_blk]
            old_point = new_point
            new_point = fix_point_progress()

        return live_map

    def _compute_dead_maps(self, live_map, var_def_map):
        """
        Compute the end-of-live information for variables.
        `live_map` contains a mapping of block offset to all the living
        variables at the ENTRY of the block.
        """
        # The following three dictionaries will be
        # { block offset -> set of variables to delete }
        # all vars that should be deleted at the start of the successors
        escaping_dead_map = collections.defaultdict(set)
        # all vars that should be deleted within this block
        internal_dead_map = collections.defaultdict(set)
        # all vars that should be delted after the function exit
        exit_dead_map = collections.defaultdict(set)

        for offset, ir_block in self.blocks.items():
            # live vars WITHIN the block will include all the locally
            # defined variables
            cur_live_set = live_map[offset] | var_def_map[offset]
            # vars alive alive in the outgoing blocks
            outgoing_live_map = dict((out_blk, live_map[out_blk])
                                     for out_blk, _data
                                     in self.cfa.graph.successors(offset))
            # vars to keep alive for the terminator
            terminator_liveset = set(v.name
                                     for v in ir_block.terminator.list_vars())
            # vars to keep alive in the successors
            combined_liveset = reduce(operator.or_, outgoing_live_map.values(),
                                      set())
            # include variables used in terminator
            combined_liveset |= terminator_liveset
            # vars that are dead within the block beacuse they are not
            # propagated to any outgoing blocks
            internal_set = cur_live_set - combined_liveset
            internal_dead_map[offset] = internal_set
            # vars that escape this block
            escaping_live_set = cur_live_set - internal_set
            for out_blk, new_live_set in outgoing_live_map.items():
                # successor should delete the unused escaped vars
                new_live_set = new_live_set | var_def_map[out_blk]
                escaping_dead_map[out_blk] |= escaping_live_set - new_live_set

            # if no outgoing blocks
            if not outgoing_live_map:
                # insert var used by terminator
                exit_dead_map[offset] = terminator_liveset

        # Verify that the dead maps cover all live variables
        all_vars = reduce(operator.or_, live_map.values(), set())
        internal_dead_vars = reduce(operator.or_, internal_dead_map.values(),
                                    set())
        escaping_dead_vars = reduce(operator.or_, escaping_dead_map.values(),
                                    set())
        exit_dead_vars = reduce(operator.or_, exit_dead_map.values(), set())
        dead_vars = (internal_dead_vars | escaping_dead_vars | exit_dead_vars)
        missing_vars = all_vars - dead_vars
        if missing_vars:
            # There are no exit points
            if not self.cfa.graph.exit_points():
                # We won't be able to verify this
                pass
            else:
                msg = 'liveness info missing for vars: {0}'.format(missing_vars)
                raise RuntimeError(msg)

        return internal_dead_map, escaping_dead_map

    def _patch_var_dels(self, internal_dead_map, escaping_dead_map):
        """
        Insert delete in each block
        """
        for offset, ir_block in self.blocks.items():
            # for each internal var, insert delete after the last use
            internal_dead_set = internal_dead_map[offset].copy()
            delete_pts = []
            # for each statement in reverse order
            for stmt in reversed(ir_block.body[:-1]):
                # internal vars that are used here
                live_set = set(v.name for v in stmt.list_vars())
                dead_set = live_set & internal_dead_set
                # used here but not afterwards
                delete_pts.append((stmt, dead_set))
                internal_dead_set -= dead_set

            # rewrite body and insert dels
            body = []
            for stmt, delete_set in reversed(delete_pts):
                body.append(stmt)
                # note: the reverse sort is not necessary for correctness
                #       it is just to minimize changes to test for now
                for var_name in sorted(delete_set, reverse=True):
                    body.append(ir.Del(var_name, loc=ir_block.loc))
            body.append(ir_block.body[-1])  # terminator
            ir_block.body = body

            # vars to delete at the start
            escape_dead_set = escaping_dead_map[offset]
            for var_name in sorted(escape_dead_set):
                ir_block.prepend(ir.Del(var_name, loc=ir_block.loc))

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
            except errors.NotDefinedError as e:
                if e.loc is None:
                    e.loc = self.loc
                raise e

    def dump(self, file=None):
        # Avoid early bind of sys.stdout as default value
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
        self.definitions[target.name].append(value)

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
        Get the definition site for the given variable name or instance.
        A Expr instance is returned.
        """
        while True:
            if isinstance(value, ir.Var):
                name = value.name
            elif isinstance(value, str):
                name = value
            else:
                return value
            defs = self.definitions[name]
            if len(defs) == 0:
                raise KeyError("no definition for %r"
                               % (name,))
            if len(defs) > 1:
                raise KeyError("more than one definition for %r"
                               % (name,))
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

    def op_BUILD_MAP(self, inst, items, size, res):
        items = [(self.get(k), self.get(v)) for k, v in items]
        expr = ir.Expr.build_map(items=items, size=size, loc=self.loc)
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
        expr = ir.Expr.inplace_binop(op + '=', op, lhs=lhs, rhs=rhs, loc=self.loc)
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
