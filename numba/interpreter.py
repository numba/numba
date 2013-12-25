from __future__ import print_function
import sys
import dis
import inspect
from numba import ir


NEW_BLOCKERS = frozenset(['SETUP_LOOP'])


class Interpreter(object):
    """A bytecode interpreter that builds up the IR.
    """
    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.scopes = []
        self.loc = ir.Loc(line=1)

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
        self.stack = []

        # Internal states during interpretation
        self._force_new_block = False
        self._block_actions = {}

    def _fill_global_scope(self, scope):
        """TODO
        """
        pass

    def _fill_args_into_scope(self, scope):
        argspec = inspect.getargspec(self.bytecode.func)
        for arg in argspec.args:
            scope.define(name=arg, loc=self.loc)

    def interpret(self):
        self.loc = ir.Loc(line=self.bytecode[0].lineno)
        self.scopes.append(ir.Scope(parent=self.current_scope, loc=self.loc))
        self._fill_args_into_scope(self.current_scope)
        for inst in self._iter_inst():
            self._dispatch(inst)
        # Clean up
        self._remove_invalid_syntax_blocks()

    def _remove_invalid_syntax_blocks(self):
        self.syntax_info = [syn for syn in self.syntax_info if syn.valid()]

    def verify(self):
        for b in self.blocks.itervalues():
            b.verify()

    def _iter_inst(self):
        for inst in self.bytecode:
            if self._use_new_block(inst):
                self._start_new_block(inst)
            yield inst

    def _use_new_block(self, inst):
        if inst.offset in self.bytecode.labels:
            res = True
        elif inst.opname in NEW_BLOCKERS:
            res = True
        else:
            res = self._force_new_block

        self._force_new_block = False
        return res

    def _start_new_block(self, inst):
        self.loc = ir.Loc(line=inst.lineno)
        oldblock = self.current_block
        self.insert_block(inst.offset)
        # Ensure the last block is terminated
        if oldblock is not None and not oldblock.is_terminated:
            jmp = ir.Jump(inst.offset, loc=self.loc)
            oldblock.append(jmp)
        # Notify listeners for the new block
        for fn in self._block_actions.itervalues():
            fn(self.current_block_offset, self.current_block)

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

    def _dispatch(self, inst):
        assert self.current_block is not None
        fname = "op_%s" % inst.opname
        try:
            fn = getattr(self, fname)
        except AttributeError:
            raise NotImplementedError(inst)
        else:
            return fn(inst)

    def dump(self, file=sys.stdout):
        for offset, block in sorted(self.blocks.items()):
            print('label %d:' % offset, file=file)
            block.dump(file=file)

    # --- Scope operations ---

    def store_temp(self, value):
        var = self.current_scope.make_temp(loc=self.loc)
        stmt = ir.Assign(value=value, target=var, loc=self.loc)
        self.current_block.append(stmt)
        return var

    def store(self, value, name):
        var = self.current_scope.get_or_insert(name, loc=self.loc)
        stmt = ir.Assign(value=value, target=var, loc=self.loc)
        self.current_block.append(stmt)

    # --- Block operations ---

    def insert_block(self, offset, scope=None, loc=None):
        scope = scope or self.current_scope
        loc = loc or self.loc
        blk = ir.Block(scope=scope, loc=loc)
        self.blocks[offset] = blk
        self.current_block = blk
        self.current_block_offset = offset
        return blk

    # --- Stack operations ---

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    # --- Bytecode handlers ---

    def op_LOAD_CONST(self, inst):
        val = self.code_consts[inst.arg]
        k = ir.Const(value=val, loc=self.loc)
        tmp = self.store_temp(k)
        self.push(tmp)

    def op_LOAD_GLOBAL(self, inst):
        val = self.code_names[inst.arg]
        glb = ir.Global(value=val, loc=self.loc)
        tmp = self.store_temp(glb)
        self.push(tmp)

    def op_LOAD_FAST(self, inst):
        varname = self.code_locals[inst.arg]
        var = self.current_scope.refer(varname)
        self.push(var)

    def op_LOAD_ATTR(self, inst):
        name = self.code_names[inst.arg]
        expr = ir.Expr.getattr(value=self.pop(), attr=name, loc=self.loc)
        tmp = self.store_temp(expr)
        self.push(tmp)

    def op_STORE_FAST(self, inst):
        varname = self.code_locals[inst.arg]
        val = self.pop()
        self.store(val, varname)

    def op_SETUP_LOOP(self, inst):
        assert self.blocks[inst.offset] is self.current_block
        loop = ir.Loop(inst.offset, exit=(inst.next + inst.arg))
        self.syntax_blocks.append(loop)
        self.syntax_info.append(loop)

    def op_CALL_FUNCTION(self, inst):
        narg = inst.arg & 0xff
        nkws = (inst.arg >> 8) & 0xff

        def pop_kws():
            val = self.pop()
            key = self.pop()
            return key.value, val

        kws = list(reversed([pop_kws() for _ in range(nkws)]))
        args = list(reversed([self.pop() for _ in range(narg)]))
        func = self.pop()

        expr = ir.Expr.call(func=func, args=args, kws=kws, loc=self.loc)
        tmp = self.store_temp(expr)
        self.push(tmp)

    def op_GET_ITER(self, inst):
        expr = ir.Expr.getiter(value=self.pop(), loc=self.loc)
        tmp = self.store_temp(expr)
        self.push(tmp)

    def op_FOR_ITER(self, inst):
        """
        Assign new block other this instruction.
        """
        assert inst.offset in self.blocks, "FOR_ITER must be block head"

        # Mark this block as the loop condition
        loop = self.syntax_blocks[-1]
        loop.condition = self.current_block_offset

        # Emit code
        val = self.pop()
        iternext = ir.Expr.iternext(value=val, loc=self.loc)
        indval = self.store_temp(iternext)
        itervalid = ir.Expr.itervalid(value=indval, loc=self.loc)
        pred = self.store_temp(itervalid)
        self.push(indval)

        # Conditional jump
        br = ir.Branch(cond=pred, truebr=inst.next,
                       falsebr=self.syntax_blocks[-1].exit,
                       loc=self.loc)
        self.current_block.append(br)

        # Split the block for the next instruction
        self._force_new_block = True

        # Add event listener to mark the following blocks as loop body
        def mark_as_body(offset, block):
            loop.body.append(offset)

        self._block_actions[loop] = mark_as_body

    def op_BINARY_SUBSCR(self, inst):
        index = self.pop()
        target = self.pop()
        expr = ir.Expr.getitem(target=target, index=index, loc=self.loc)
        tmp = self.store_temp(expr)
        self.push(tmp)

    def op_STORE_SUBSCR(self, inst):
        index = self.pop()
        target = self.pop()
        value = self.pop()
        stmt = ir.SetItem(target=target, index=index, value=value,
                               loc=self.loc)
        self.current_block.append(stmt)

    def _binop(self, op, inst):
        rhs = self.pop()
        lhs = self.pop()
        expr = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
        tmp = self.store_temp(expr)
        self.push(tmp)

    def op_BINARY_ADD(self, inst):
        self._binop('+', inst)

    def op_BINARY_SUBTRACT(self, inst):
        self._binop('-', inst)

    def op_BINARY_MULTIPLY(self, inst):
        self._binop('*', inst)

    def op_BINARY_DIVIDE(self, inst):
        self._binop('/?', inst)

    def op_BINARY_TRUE_DIVIDE(self, inst):
        self._binop('/', inst)

    def op_BINARY_FLOOR_DIVIDE(self, inst):
        self._binop('//', inst)

    def op_BINARY_MODULO(self, inst):
        self._binop('%', inst)

    def _inplace_binop(self, op, inst):
        rhs = self.pop()
        lhs = self.pop()
        expr = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
        tmp = self.store_temp(expr)
        self.push(tmp)

    def op_INPLACE_ADD(self, inst):
        self._inplace_binop('+', inst)

    def op_INPLACE_SUBSTRACT(self, inst):
        self._inplace_binop('-', inst)

    def op_INPLACE_MULTIPLY(self, inst):
        self._inplace_binop('*', inst)

    def op_INPLACE_DIVIDE(self, inst):
        self._inplace_binop('/?', inst)

    def op_INPLACE_TRUE_DIVIDE(self, inst):
        self._inplace_binop('/', inst)

    def op_INPLACE_FLOOR_DIVIDE(self, inst):
        self._inplace_binop('//', inst)

    def op_JUMP_ABSOLUTE(self, inst):
        jmp = ir.Jump(inst.arg, loc=self.loc)
        self.current_block.append(jmp)

    def op_POP_BLOCK(self, inst):
        blk = self.syntax_blocks.pop()
        if blk in self._block_actions:
            del self._block_actions[blk]

    def op_RETURN_VALUE(self, inst):
        value = self.pop()
        ret = ir.Return(value, loc=self.loc)
        self.current_block.append(ret)

    def op_COMPARE_OP(self, inst):
        op = dis.cmp_op[inst.arg]
        self._binop(op, inst)

    def op_POP_JUMP_IF_FALSE(self, inst):
        target = inst.arg
        pred = self.pop()
        truebr = target
        falsebr = inst.next
        bra = ir.Branch(cond=pred, truebr=falsebr, falsebr=truebr,
                        loc=self.loc)
        self.current_block.append(bra)
        # Split block at next instruction
        self._force_new_block = True
        # In a while loop?
        self._determine_while_condition((truebr, falsebr))

    def op_POP_JUMP_IF_TRUE(self, inst):
        target = inst.arg
        pred = self.pop()
        truebr = target
        falsebr = inst.next
        bra = ir.Branch(cond=pred, truebr=truebr, falsebr=falsebr,
                        loc=self.loc)
        self.current_block.append(bra)
        # Split block at next instruction
        self._force_new_block = True
        # In a while loop?
        self._determine_while_condition((truebr, falsebr))

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
            if self.bytecode[br].opname == 'POP_BLOCK':
                break
        else:
            return
        # Which is the exit of the loop
        if loop.exit != br + 1:
            return

        # Therefore, current block is a while loop condition
        loop.condition = self.current_block_offset
        # Add event listener to mark the following blocks as loop body
        def mark_as_body(offset, block):
            loop.body.append(offset)

        self._block_actions[loop] = mark_as_body


