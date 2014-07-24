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
        self.src_invalidate = collections.defaultdict(list)
        self.unused_dests = set()

    def assign(self, srcvar, destvar):
        """
        Assign *srcvar* to *destvar*. Return either *srcvar* or a possible
        simplified assignment source (earlier assigned to *srcvar*).
        """
        srcname = srcvar.name
        destname = destvar.name
        if srcname in self.src_invalidate:
            # Invalidate all previously known simplifications
            for d in self.src_invalidate.pop(srcname):
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


class Interpreter(object):
    """A bytecode interpreter that builds up the IR.
    """

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.scopes = []
        self.loc = ir.Loc(filename=bytecode.filename, line=1)
        self.argspec = bytecode.argspec
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
        self.syntax_info = []

        # Temp states during interpretation

        self.current_block = None
        self.current_block_offset = None
        self.syntax_blocks = []
        self.dfainfo = None
        self._block_actions = {}
        self.constants = {}

    def _fill_global_scope(self, scope):
        """TODO
        """
        pass

    def _fill_args_into_scope(self, scope):
        for arg in self.argspec.args:
            scope.define(name=arg, loc=self.loc)

    def interpret(self):
        firstblk = min(self.cfa.blocks.keys())
        self.loc = ir.Loc(filename=self.bytecode.filename,
                          line=self.bytecode[firstblk].lineno)
        self.scopes.append(ir.Scope(parent=self.current_scope, loc=self.loc))
        self._fill_args_into_scope(self.current_scope)
        # Interpret loop
        for inst, kws in self._iter_inst():
            self._dispatch(inst, kws)
            # Clean up
        self._remove_invalid_syntax_blocks()

    def _remove_invalid_syntax_blocks(self):
        self.syntax_info = [syn for syn in self.syntax_info if syn.valid()]

    def verify(self):
        for b in utils.dict_itervalues(self.blocks):
            b.verify()

    def init_first_block(self):
        # Duplicate arguments so that these values can be casted into different
        # types.
        for aname in self.argspec.args:
            aval = self.get(aname)
            self.store(aval, aname, redefine=True)

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
        # Notify listeners for the new block
        for fn in utils.dict_itervalues(self._block_actions):
            fn(self.current_block_offset, self.current_block)

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
            return utils.func_globals(self.bytecode.func)[name]
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
            return fn(inst, **kws)

    def dump(self, file=None):
        file = file or sys.stdout
        for offset, block in sorted(self.blocks.items()):
            print('label %d:' % offset, file=file)
            block.dump(file=file)

    # --- Scope operations ---

    def store(self, value, name, redefine=False):
        """
        Store *value* (a Var instance) into the variable named *name*
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

    def get(self, name):
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
        gl = ir.Global(name, value, loc=self.loc)
        self.store(gl, res)
        self.constants[res] = value

    def op_LOAD_DEREF(self, inst, res):
        name = self.code_freevars[inst.arg]
        value = self.get_closure_value(inst.arg)
        gl = ir.FreeVar(inst.arg, name, value, loc=self.loc)
        self.store(gl, res)

    def op_SETUP_LOOP(self, inst):
        assert self.blocks[inst.offset] is self.current_block
        loop = ir.Loop(inst.offset, exit=(inst.next + inst.arg))
        self.syntax_blocks.append(loop)
        self.syntax_info.append(loop)

    def op_CALL_FUNCTION(self, inst, func, args, kws, res):
        func = self.get(func)
        args = [self.get(x) for x in args]

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

        expr = ir.Expr.call(func, args, keyvalues, loc=self.loc)
        self.store(expr, res)

    def op_GET_ITER(self, inst, value, res):
        expr = ir.Expr.getiter(value=self.get(value), loc=self.loc)
        self.store(expr, res)

    def op_FOR_ITER(self, inst, iterator, pair, indval, pred):
        """
        Assign new block other this instruction.
        """
        assert inst.offset in self.blocks, "FOR_ITER must be block head"

        # Mark this block as the loop condition
        loop = self.syntax_blocks[-1]
        loop.condition = self.current_block_offset

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

        # Add event listener to mark the following blocks as loop body
        def mark_as_body(offset, block):
            loop.body.append(offset)

        self._block_actions[loop] = mark_as_body

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

    def op_POP_BLOCK(self, inst, delitems=()):
        blk = self.syntax_blocks.pop()
        for item in delitems:
            delete = ir.Del(item, loc=self.loc)
            self.current_block.append(delete)
        if blk in self._block_actions:
            del self._block_actions[blk]

    def op_RETURN_VALUE(self, inst, retval):
        ret = ir.Return(self.get(retval), loc=self.loc)
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
        # In a while loop?
        self._determine_while_condition((truebr, falsebr))

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
        stmt = ir.Raise(exception=self.constants[exc], loc=self.loc)
        self.current_block.append(stmt)

    def _determine_while_condition(self, branches):
        assert branches
        # There is a active syntax block
        if not self.syntax_blocks:
            return
            # TOS is a Loop instance
        loop = self.syntax_blocks[-1]
        if not isinstance(loop, ir.Loop):
            return
            # Its condition is not defined
        if loop.condition is not None:
            return
            # One of the branches goes to a POP_BLOCK
        for br in branches:
            if self.block_constains_opname(br, 'POP_BLOCK'):
                break
        else:
            return
            # Which is the exit of the loop
        if br not in self.cfa.blocks[loop.exit].incoming_jumps:
            return

        # Therefore, current block is a while loop condition
        loop.condition = self.current_block_offset
        # Add event listener to mark the following blocks as loop body
        def mark_as_body(offset, block):
            loop.body.append(offset)

        self._block_actions[loop] = mark_as_body
