from __future__ import print_function, division, absolute_import
try:
    import __builtin__ as builtins
except ImportError:
    import builtins
import sys
import dis
import inspect
from numba import ir, controlflow, dataflow, utils


class Interpreter(object):
    """A bytecode interpreter that builds up the IR.
    """
    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.scopes = []
        self.loc = ir.Loc(filename=bytecode.filename, line=1)
        self.argspec = inspect.getargspec(self.bytecode.func)
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

    def _fill_global_scope(self, scope):
        """TODO
        """
        pass

    def _fill_args_into_scope(self, scope):
        for arg in self.argspec.args:
            scope.define(name=arg, loc=self.loc)

    def interpret(self):
        self.loc = ir.Loc(filename=self.bytecode.filename,
                          line=self.bytecode[0].lineno)
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

    def _iter_inst(self):
        for block in self.cfa.iterliveblocks():
            firstinst = self.bytecode[block.body[0]]
            self._start_new_block(firstinst)
            for offset, kws in self.dfainfo.insts:
                inst = self.bytecode[offset]
                self.loc = ir.Loc(filename=self.bytecode.filename, line=inst.lineno)
                yield inst, kws

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
        # Insert PHI
        self._insert_phi()
        # Notify listeners for the new block
        for fn in utils.dict_itervalues(self._block_actions):
            fn(self.current_block_offset, self.current_block)

    def _insert_phi(self):
        if self.dfainfo.incomings:
            assert len(self.dfainfo.incomings) == 1
            incomings = self.cfa.blocks[self.current_block_offset].incoming
            phivar = self.dfainfo.incomings[0]
            if len(incomings) == 1:
                ib = utils.iter_next(iter(incomings))
                lingering = self.dfa.infos[ib].stack
                assert len(lingering) == 1
                iv = lingering[0]
                self.store(self.get(iv), phivar)
            else:
                # Invert the PHI node
                for ib in incomings:
                    lingering = self.dfa.infos[ib].stack
                    assert len(lingering) == 1
                    iv = lingering[0]

                    # Add assignment in incoming block to forward the value
                    target = self.current_scope.get_or_define('$phi' + phivar,
                                                              loc=self.loc)
                    stmt = ir.Assign(value=self.get(iv), target=target,
                                     loc=self.loc)
                    self.blocks[ib].insert_before_terminator(stmt)

                self.store(target, phivar)


    def get_global_value(self, name):
        """
        Get a global value from the func_global (first) or
        as a builtins (second).  If both failed, return a ir.UNDEFINED.
        """
        try:
            return utils.func_globals(self.bytecode.func)[name]
        except KeyError:
            return getattr(builtins, name, ir.UNDEFINED)

    @property
    def current_scope(self):
        return self.scopes[-1]

    @property
    def code_consts(self):
        return self.bytecode.code.co_consts

    @property
    def code_locals(self):
        return self.bytecode.code.co_varnames

    @property
    def code_names(self):
        return self.bytecode.code.co_names

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

    def store(self, value, name):
        if self.current_block_offset in self.cfa.backbone:
            target = self.current_scope.redefine(name, loc=self.loc)
        else:
            target = self.current_scope.get_or_define(name, loc=self.loc)
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        self.current_block.append(stmt)

    # def store_temp(self, value):
    #     target = self.current_scope.make_temp(loc=self.loc)
    #     stmt = ir.Assign(value=value, target=target, loc=self.loc)
    #     self.current_block.append(stmt)
    #     return target

    def get(self, name):
        return self.current_scope.get(name)

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

    def op_UNPACK_SEQUENCE(self, inst, sequence, stores, iterobj):
        sequence = self.get(sequence)
        getiter = ir.Expr.getiter(value=sequence, loc=self.loc)
        self.store(value=getiter, name=iterobj)

        for st in stores:
            iternext = ir.Expr.iternextsafe(value=self.get(iterobj),
                                            loc=self.loc)
            self.store(value=iternext, name=st)

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

    def op_SLICE_3(self, inst, base, start, stop, res):
        base = self.get(base)
        start = self.get(start)
        stop = self.get(stop)

        expr = ir.Expr.getslice(target=base, start=start, stop=stop,
                                loc=self.loc)
        self.store(value=expr, name=res)

    def op_STORE_FAST(self, inst, value):
        dstname = self.code_locals[inst.arg]
        value = self.get(value)
        self.store(value=value, name=dstname)

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

    def op_FOR_ITER(self, inst, iterator, indval, pred):
        """
        Assign new block other this instruction.
        """
        assert inst.offset in self.blocks, "FOR_ITER must be block head"

        # Mark this block as the loop condition
        loop = self.syntax_blocks[-1]
        loop.condition = self.current_block_offset

        # Emit code
        val = self.get(iterator)
        iternext = ir.Expr.iternext(value=val, loc=self.loc)
        self.store(iternext, indval)

        itervalid = ir.Expr.itervalid(value=val, loc=self.loc)
        self.store(itervalid, pred)

        # Conditional jump
        br = ir.Branch(cond=self.get(pred), truebr=inst.next,
                       falsebr=self.syntax_blocks[-1].exit,
                       loc=self.loc)
        self.current_block.append(br)

        # Add event listener to mark the following blocks as loop body
        def mark_as_body(offset, block):
            loop.body.append(offset)

        self._block_actions[loop] = mark_as_body

    def op_BINARY_SUBSCR(self, inst, target, index, res):
        index = self.get(index)
        target = self.get(target)
        expr = ir.Expr.getitem(target=target, index=index, loc=self.loc)
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

    def op_UNARY_NEGATIVE(self, inst, value, res):
        value = self.get(value)
        expr = ir.Expr.unary('-', value=value, loc=self.loc)
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

    _inplace_binop = _binop

    def op_INPLACE_ADD(self, inst, lhs, rhs, res):
        self._inplace_binop('+', lhs, rhs, res)

    def op_INPLACE_SUBSTRACT(self, inst, lhs, rhs, res):
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

    def op_POP_BLOCK(self, inst, delitem=None):
        blk = self.syntax_blocks.pop()
        if delitem is not None:
            delete = ir.Del(delitem, loc=self.loc)
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
            True:  inst.get_jump_target(),
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
        if br not in self.cfa.blocks[loop.exit].incoming:
            return

        # Therefore, current block is a while loop condition
        loop.condition = self.current_block_offset
        # Add event listener to mark the following blocks as loop body
        def mark_as_body(offset, block):
            loop.body.append(offset)

        self._block_actions[loop] = mark_as_body
