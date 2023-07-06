import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator, Union
from functools import reduce

from numba.core import (
    ir,
    bytecode,
    ir_utils,
    utils,
)
from numba.core.utils import (
    BINOPS_TO_OPERATORS,
    INPLACE_BINOPS_TO_OPERATORS,
)

from .bc2rvsdg import (
    build_rvsdg,
    SCFG,
    BasicBlock,
    RegionBlock,
    DDGBlock,
    DDGControlVariable,
    DDGBranch,
    DDGRegion,
    Op,
    ValueState,
    DEBUG_GRAPH,
)
from .regionpasses import RegionVisitor


def run_frontend(func):

    sig = utils.pySignature.from_callable(func)
    argnames = tuple(sig.parameters)
    rvsdg = build_rvsdg(func.__code__, argnames)

    func_id = bytecode.FunctionIdentity.from_function(func)
    func_ir = rvsdg_to_ir(func_id, rvsdg)

    return func_ir


def _get_first_bytecode(ops: list[Op]) -> dis.Instruction | None:
    for bc in (op.bc_inst for op in ops if op.bc_inst is not None):
        return bc


def _innermost_exiting(blk: RegionBlock) -> BasicBlock:
    while isinstance(blk, RegionBlock):
        blk = blk.subregion.graph[blk.exiting]
    return blk


_noop = {"var.incoming", "start"}


class RVSDG2IR(RegionVisitor):
    blocks: dict[int, ir.Block]
    func_id: bytecode.FunctionIdentity
    scope: ir.Scope
    global_scope: ir.Scope
    vsmap: dict[ValueState, ir.Var]
    _current_block: ir.Block | None
    last_block_label: int | None
    branch_predicate: ir.Var | None

    _label_map: dict[Union[str, int], int]

    _emit_debug_print = False

    def __init__(self, func_id):
        self.func_id = func_id
        self.loc = self.first_loc = ir.Loc.from_function_id(func_id)
        self.global_scope = ir.Scope(parent=None, loc=self.loc)
        self.scope = ir.Scope(parent=self.global_scope, loc=self.loc)
        self.blocks = {}
        self.vsmap = {}
        self._current_block = None
        self.last_block_label = None

        self._label_map = {}

    @property
    def current_block(self) -> ir.Block:
        out = self._current_block
        assert out is not None
        return out

    @property
    def last_block(self) -> ir.Block:
        return self.blocks[self.last_block_label]

    def _get_phi_name(self, varname: str, label: str) -> str:
        suffix = str(self._get_label(label))
        return f"$phi.{varname}.{suffix}"

    def _get_label(self, label: str) -> int:
        num = self._label_map.setdefault(label, len(self._label_map))
        return num

    def _get_temp_label(self) -> int:
        num = len(self._label_map)
        assert num not in self._label_map
        self._label_map[num] = num
        return num

    def initialize(self):
        self._label_map[None] = 0
        with self.set_block(0, ir.Block(scope=self.scope, loc=self.loc)):
            data = {}
            for i, k in enumerate(self.func_id.arg_names):
                val = ir.Arg(index=i, name=k, loc=self.loc)
                data[f"var.{k}"] = self.store(val, str(k))
            return data

    def finalize(self):
        if self.last_block_label is not None:
            last_block = self.blocks[self.last_block_label]
            if not last_block.is_terminated:
                last_block.append(
                    ir.StaticRaise(AssertionError, (), loc=self.loc)
                )

    def visit_block(self, block: BasicBlock, data):
        if isinstance(block, DDGBlock):
            # Prepare incoming variables
            for k, vs in block.in_vars.items():
                self.vsmap[vs] = data[k]

            # Emit instruction body
            ops = block.get_toposorted_ops()
            firstbc: dis.Instruction | None = _get_first_bytecode(ops)
            if firstbc is not None:
                assert firstbc.positions is not None
                self.loc = self.loc.with_lineno(
                    firstbc.positions.lineno,
                    firstbc.positions.col_offset,
                )
            with self.set_block(
                self._get_label(block.name),
                ir.Block(scope=self.scope, loc=self.loc),
            ):
                for op in ops:
                    if op.opname in _noop:
                        pass
                    elif op.bc_inst is not None:
                        self.interpret_bytecode(op)
                    elif op.opname in {"stack.export", "stack.incoming"}:
                        [arg] = op.inputs
                        [res] = op.outputs
                        self.vsmap[res] = self.store(
                            self.vsmap[arg], f"${res.name}"
                        )
                    else:
                        raise NotImplementedError(op.opname, op)
                if len(block._jump_targets) > 1:
                    assert self.branch_predicate is not None
                    truebr = self._get_label(block._jump_targets[1])
                    falsebr = self._get_label(block._jump_targets[0])
                    br = ir.Branch(
                        cond=self.branch_predicate,
                        truebr=truebr,
                        falsebr=falsebr,
                        loc=self.loc,
                    )
                    self.current_block.append(br)
            # Prepare outgoing variables
            data = {k: self.vsmap[vs] for k, vs in block.out_vars.items()}
            return data
        elif isinstance(block, DDGControlVariable):
            # Emit body
            with self.set_block(
                self._get_label(block.name),
                ir.Block(scope=self.scope, loc=self.loc),
            ):
                for cp, v in block.variable_assignment.items():
                    const = ir.Const(v, loc=self.loc, use_literal_type=False)
                    self.store(const, f"$.cp.{cp}", redefine=False)
            return data
        elif isinstance(block, DDGBranch):
            # Emit body
            if len(block.branch_value_table) == 2:
                with self.set_block(
                    self._get_label(block.name),
                    ir.Block(scope=self.scope, loc=self.loc),
                ):
                    # Handle simple two-way branch
                    assert set(block.branch_value_table.keys()) == {0, 1}
                    cp = block.variable
                    cpvar = self.scope.get_exact(f"$.cp.{cp}")
                    truebr = self._get_label(block.branch_value_table[1])
                    falsebr = self._get_label(block.branch_value_table[0])
                    br = ir.Branch(
                        cond=cpvar,
                        truebr=truebr,
                        falsebr=falsebr,
                        loc=self.loc,
                    )
                    self.current_block.append(br)
            else:
                # with self.set_block(
                #         self._get_label(block.name),
                #         ir.Block(scope=self.scope, loc=self.loc)):
                #     # Handle simple two-way branch
                #     # assert set(block.branch_value_table.keys()) == {0, 1}
                #     cp = block.variable
                #     cpvar = self.scope.get_exact(f"$.cp.{cp}")

                #     falsebr = self._get_label(block.branch_value_table[1])
                #     truebr = self._get_label(block.branch_value_table[0])

                #     const = self.store(ir.Const(0, loc=self.loc), "$.const")
                #     cmp = ir.Expr.binop(operator.eq, const, cpvar,
                #                         loc=self.loc)
                #     pred = self.store(cmp, "$.pred")
                #     br = ir.Branch(
                #         cond=pred,
                #         truebr=truebr,
                #         falsebr=falsebr,
                #         loc=self.loc,
                #     )
                #     self.current_block.append(br)
                bvt = block.branch_value_table
                cp = block.variable
                cpvar = self.scope.get_exact(f"$.cp.{cp}")
                labels = [(k, self._get_label(v)) for k, v in bvt.items()]

                blocks = []
                for _ in labels[:-1]:
                    blocks.append(
                        (
                            self._get_temp_label(),
                            ir.Block(scope=self.scope, loc=self.loc),
                        )
                    )

                # Jump into the first block
                with self.set_block(
                    self._get_label(block.name),
                    ir.Block(scope=self.scope, loc=self.loc),
                ):
                    self.current_block.append(
                        ir.Jump(blocks[-1][0], loc=self.loc)
                    )

                # Handle jump tree
                while blocks:
                    cp_expect, cp_label = labels.pop()
                    cur_label, cur_block = blocks.pop()
                    with self.set_block(cur_label, cur_block):
                        const = self.store(
                            ir.Const(cp_expect, loc=self.loc), "$.const"
                        )
                        cmp = ir.Expr.binop(
                            operator.eq, const, cpvar, loc=self.loc
                        )
                        pred = self.store(cmp, "$.cmp")

                        if not blocks:
                            _, falsebr = labels.pop()
                        else:
                            falsebr, _ = blocks[-1]
                        br = ir.Branch(
                            cond=pred,
                            truebr=cp_label,
                            falsebr=falsebr,
                            loc=self.loc,
                        )
                        self.current_block.append(br)
            return data
        else:
            raise NotImplementedError(block.name, type(block))

    def visit_loop(self, region: RegionBlock, data):
        assert isinstance(region, DDGRegion)
        # Prepare incoming states
        inner_data = {}
        for k in region.incoming_states:
            inner_data[k] = self.store(
                data[k],
                self._get_phi_name(k, region.name),
                redefine=False,
                block=self.last_block,
            )

        # Emit loop body
        out_data = self.visit_linear(region, inner_data)

        # Prepare outgoing states
        exit_data = {}
        for k in region.outgoing_states:
            exit_data[k] = self.store(
                out_data[k],
                self._get_phi_name(k, region.name),
                redefine=False,
                block=self.last_block,
            )

        return exit_data

    def visit_switch(self, region: RegionBlock, data):
        # Emit header
        header = region.header
        header_block = region.subregion[header]
        assert header_block.kind == "head"
        self.branch_predicate = None
        data_at_head = self.visit_linear(header_block, data)
        if not self.last_block.is_terminated:
            assert self.branch_predicate is not None

            # Jump-target 1 is when a the jump is taken.
            # Jump-target 0 is when the jump fallthrough
            truebr = self._get_label(header_block.jump_targets[1])
            falsebr = self._get_label(header_block.jump_targets[0])
            br = ir.Branch(
                self.branch_predicate, truebr, falsebr, loc=self.loc
            )
            self.last_block.append(br)

        # Emit branches
        data_for_branches = []
        branch_blocks = []
        for blk in region.subregion.graph.values():
            if blk.kind == "branch":
                branch_blocks.append(_innermost_exiting(blk))
                data_for_branches.append(self.visit_linear(blk, data_at_head))
                # Add jump to tail
                if not self.last_block.is_terminated:
                    [target] = blk.jump_targets
                    self.last_block.append(
                        ir.Jump(self._get_label(target), loc=self.loc)
                    )

        # handle outgoing values from the branches
        names = reduce(operator.or_, map(set, data_for_branches))
        for blk, branch_data in zip(
            branch_blocks, data_for_branches, strict=True
        ):
            for k in names:
                # Set undefined variable to None
                # (It should be a "zeroinitiailizer" but ir.Expr.null doesn't
                #  work)
                rhs = branch_data.get(k, ir.Const(None, loc=self.loc))
                # Insert stores to export
                phiname = self._get_phi_name(k, region.name)
                self.store(
                    rhs,
                    phiname,
                    redefine=False,
                    block=self.blocks[self._get_label(blk.name)],
                )
        data_after_branches = {
            k: self.scope.get_exact(self._get_phi_name(k, region.name))
            for k in names
        }

        # Emit tail
        exiting = region.exiting
        exiting_block = region.subregion[exiting]
        assert exiting_block.kind == "tail"
        data_at_tail = self.visit_linear(exiting_block, data_after_branches)

        return data_at_tail

    def visit_linear(self, region: RegionBlock, data):
        with self.set_block(
            self._get_label(region.name),
            ir.Block(scope=self.scope, loc=self.loc),
        ):
            # Ensures there's a block for all regions
            pass
        return super().visit_linear(region, data)

    @contextmanager
    def set_block(self, label: int, block: ir.Block) -> Iterator[ir.Block]:
        if self.last_block_label is not None:
            last_block = self.blocks[self.last_block_label]
            if not last_block.is_terminated:
                last_block.append(ir.Jump(label, loc=self.loc))

            if self._emit_debug_print:
                print("begin dump last blk".center(80, "-"))
                last_block.dump()
                print("end dump last blk".center(80, "="))

        self.blocks[label] = block
        old = self._current_block
        self._current_block = block
        try:
            yield block
        finally:
            self.last_block_label = label
            self._current_block = old
            # dump
            if self._emit_debug_print:
                print(f"begin dump blk: {label}".center(80, "-"))
                block.dump()
                print("end dump blk".center(80, "="))

    def store(self, value, name, *, redefine=True, block=None) -> ir.Var:
        target: ir.Var
        if redefine:
            target = self.scope.redefine(name, loc=self.loc)
        else:
            target = self.scope.get_or_define(name, loc=self.loc)
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        self.append(stmt, block=block)
        return target

    def store_vsmap(self, val, vs):
        self.vsmap[vs] = self.store(val, f"${vs.name}")

    def append(self, stmt: ir.Stmt, block=None):
        if block is None:
            block = self.current_block
        if block.is_terminated:
            block.insert_before_terminator(stmt)
        else:
            block.append(stmt)

    def get_global_value(self, name):
        """THIS IS COPIED from interpreter.py

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

    def debug_print(self, msg: str, *args):
        msg_const = self.store(ir.Const(msg, loc=self.loc), "$.debug.msg")
        fn = self.store(ir.Const(print, loc=self.loc), "$.debug.print")
        res = ir.Expr.call(fn, (msg_const, *args), (), loc=self.loc)
        self.store(res, "$.debug.res")

    def interpret_bytecode(self, op: Op):
        assert op.bc_inst is not None
        pos = op.bc_inst.positions
        assert pos is not None
        self.loc = self.loc.with_lineno(pos.lineno, pos.col_offset)
        # debug print
        if self._emit_debug_print:
            where = f"{op.bc_inst.offset:3}:({pos.lineno:3}:{pos.col_offset:3})"
            msg = f"[{where}] {op.bc_inst.opname}({op.bc_inst.argrepr}) "
            self.debug_print(msg)

            for k, vs in op.input_ports.items():
                val = self.vsmap.get(vs, None)
                if val is None:
                    self.debug_print(f"   in {k:>6}: <undef>")
                else:
                    self.debug_print(f"   in {k:>6}:", val)

        # dispatch
        fn = getattr(self, f"op_{op.bc_inst.opname}")
        fn(op, op.bc_inst)

        # debug print
        if self._emit_debug_print:
            for k, vs in op.output_ports.items():
                val = self.vsmap.get(vs, None)
                if val is None:
                    self.debug_print(f"  out {k:>6}: <undef>")
                else:
                    self.debug_print(f"  out {k:>6}:", val)

    def op_PUSH_NULL(self, op: Op, bc: dis.Instruction):
        pass

    def op_LOAD_CONST(self, op: Op, bc: dis.Instruction):
        assert not op.inputs
        [vs] = op.outputs
        # TODO: handle non scalar
        value = ir.Const(bc.argval, loc=self.loc)
        self.store_vsmap(value, vs)

    def op_LOAD_GLOBAL(self, op: Op, bc: dis.Instruction):
        # intentionally ignoring the nil
        [_nil, res] = op.outputs
        value = self.get_global_value(bc.argval)
        # TODO: handle non scalar
        const = ir.Const(value, loc=self.loc)
        self.store_vsmap(const, res)

    def op_LOAD_ATTR(self, op: Op, bc: dis.Instruction):
        [res] = op.outputs
        [item] = op.inputs
        getattr = ir.Expr.getattr(self.vsmap[item], bc.argval, loc=self.loc)
        self.store_vsmap(getattr, res)

    def op_LOAD_METHOD(self, op: Op, bc: dis.Instruction):
        [_nil, res] = op.outputs
        [item] = op.inputs
        getattr = ir.Expr.getattr(self.vsmap[item], bc.argval, loc=self.loc)
        self.store_vsmap(getattr, res)

    def op_LOAD_DEREF(self, op: Op, bc: dis.Instruction):
        [out] = op.outputs
        code = self.func_id.code
        name = bc.argval
        if name in code.co_cellvars:
            raise NotImplementedError
            gl = self.get(name)
        elif name in code.co_freevars:
            idx = code.co_freevars.index(name)
            value = self.get_closure_value(idx)
            gl = ir.FreeVar(idx, name, value, loc=self.loc)
            self.store_vsmap(gl, out)

    def op_STORE_FAST(self, op: Op, bc: dis.Instruction):
        [incvar] = op.inputs
        [res] = op.outputs
        var = self.vsmap[incvar]
        self.vsmap[res] = self.store(var, bc.argval)

    def op_KW_NAMES(self, op: Op, bc: dis.Instruction):
        pass  # do nothing

    def op_CALL(self, op: Op, bc: dis.Instruction):
        [_env, callee, arg0, *args] = op.inputs
        [_env, res] = op.outputs
        assert callee.name == "null"
        callee = arg0
        callee = self.vsmap[callee]

        if op.opname == "call.kw":
            kw_names_op = args[-1].parent
            assert kw_names_op.opname == "kw_names"
            args = args[:-1]
            names = self.func_id.code.co_consts[kw_names_op.bc_inst.arg]
            args = [self.vsmap[vs] for vs in args]
            kwargs = list(zip(names, args[-len(names) :]))
            args = args[: -len(names)]
        else:
            assert op.opname == "call"
            args = [self.vsmap[vs] for vs in args]
            kwargs = ()

        expr = ir.Expr.call(callee, args, kwargs, loc=self.loc)
        self.store_vsmap(expr, res)

    def op_COMPARE_OP(self, op: Op, bc: dis.Instruction):
        [_env, lhs, rhs] = op.inputs
        [_env, out] = op.outputs
        operator = bc.argrepr
        op = BINOPS_TO_OPERATORS[operator]
        lhs = self.vsmap[lhs]
        rhs = self.vsmap[rhs]
        expr = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
        self.store_vsmap(expr, out)

    def _binop(self, operator, op):
        [_env, lhs, rhs] = op.inputs
        [_env, out] = op.outputs

        if "=" in operator:
            operator = operator[:-1]
            immuop = BINOPS_TO_OPERATORS[operator]
            op = INPLACE_BINOPS_TO_OPERATORS[operator + "="]
            expr = ir.Expr.inplace_binop(
                op,
                immuop,
                lhs=self.vsmap[lhs],
                rhs=self.vsmap[rhs],
                loc=self.loc,
            )
        else:
            op = BINOPS_TO_OPERATORS[operator]
            lhs = self.vsmap[lhs]
            rhs = self.vsmap[rhs]
            expr = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
        self.store_vsmap(expr, out)

    def op_BINARY_OP(self, op: Op, bc: dis.Instruction):
        self._binop(bc.argrepr, op)

    def op_IS_OP(self, op: Op, bc: dis.Instruction):
        self._binop("is", op)

    def op_UNARY_NOT(self, op: Op, bc: dis.Instruction):
        [val] = op.inputs
        [out] = op.outputs
        expr = ir.Expr.unary("not", value=self.vsmap[val], loc=self.loc)
        self.store_vsmap(expr, out)

    def op_BINARY_SUBSCR(self, op: Op, bc: dis.Instruction):
        [_env, index, target] = op.inputs
        [_env, out] = op.outputs
        index = self.vsmap[index]
        target = self.vsmap[target]
        expr = ir.Expr.getitem(target, index=index, loc=self.loc)
        self.store_vsmap(expr, out)

    def op_STORE_SUBSCR(self, op: Op, bc: dis.Instruction):
        [_env, index, target, value] = op.inputs
        [_env] = op.outputs
        index = self.vsmap[index]
        target = self.vsmap[target]
        value = self.vsmap[value]
        stmt = ir.SetItem(
            target=target, index=index, value=value, loc=self.loc
        )
        self.append(stmt)

    def op_BUILD_TUPLE(self, op: Op, bc: dis.Instruction):
        items = op.inputs
        [out] = op.outputs
        expr = ir.Expr.build_tuple(
            items=[self.vsmap[it] for it in items], loc=self.loc
        )
        self.store_vsmap(expr, out)

    def op_BUILD_SLICE(self, op: Op, bc: dis.Instruction):
        args = tuple([self.vsmap[v] for v in op.inputs])
        [out] = op.outputs
        assert len(args) in (2, 3), "expected (start, stop, [step])"
        slicegv = ir.Global("slice", slice, loc=self.loc)
        slicevar = self.store(value=slicegv, name="$slicevar", redefine=True)
        sliceinst = ir.Expr.call(slicevar, args, (), loc=self.loc)
        self.store_vsmap(sliceinst, out)

    def op_GET_ITER(self, op: Op, bc: dis.Instruction):
        [arg] = op.inputs
        [res] = op.outputs
        expr = ir.Expr.getiter(value=self.vsmap[arg], loc=self.loc)
        self.store_vsmap(expr, res)

    def op_FOR_ITER(self, op: Op, bc: dis.Instruction):
        [iterator] = op.inputs
        [res] = op.outputs

        # Emit code
        pairval = ir.Expr.iternext(value=self.vsmap[iterator], loc=self.loc)
        pair = self.store(pairval, "$foriter")

        iternext = ir.Expr.pair_first(value=pair, loc=self.loc)
        indval = self.store(iternext, "$foriter.indval")
        self.vsmap[res] = indval

        isvalid = ir.Expr.pair_second(value=pair, loc=self.loc)
        pred = self.store(isvalid, "$foriter.isvalid")

        not_fn = ir.Const(operator.not_, loc=self.loc)
        res = ir.Expr.call(
            self.store(not_fn, "$not"), (pred,), (), loc=self.loc
        )
        self.branch_predicate = self.store(res, "$for_iter")

    def op_JUMP_IF_FALSE_OR_POP(self, op: Op, bc: dis.Instruction):
        [_env, pred] = op.inputs
        [_env] = op.outputs
        not_fn = ir.Const(operator.not_, loc=self.loc)
        res = ir.Expr.call(
            self.store(not_fn, "$not"), (self.vsmap[pred],), (), loc=self.loc
        )
        self.branch_predicate = self.store(res, "$jump_if")

    def op_JUMP_IF_TRUE_OR_POP(self, op: Op, bc: dis.Instruction):
        [_env, pred] = op.inputs
        [_env] = op.outputs
        self.branch_predicate = self.store(self.vsmap[pred], "$jump_if")

    def op_POP_JUMP_FORWARD_IF_FALSE(self, op: Op, bc: dis.Instruction):
        [_env, pred] = op.inputs
        [_env] = op.outputs
        not_fn = ir.Const(operator.not_, loc=self.loc)
        res = ir.Expr.call(
            self.store(not_fn, "$not"), (self.vsmap[pred],), (), loc=self.loc
        )
        self.branch_predicate = self.store(res, "$jump_if")

    def op_POP_JUMP_FORWARD_IF_TRUE(self, op: Op, bc: dis.Instruction):
        [_env, pred] = op.inputs
        [_env] = op.outputs
        self.branch_predicate = self.store(self.vsmap[pred], "$jump_if")

    def op_POP_JUMP_FORWARD_IF_NONE(self, op: Op, bc: dis.Instruction):
        [_env, pred] = op.inputs
        [_env] = op.outputs
        op = BINOPS_TO_OPERATORS["is"]
        none = self.store(
            value=ir.Const(None, loc=self.loc), name=f"$constNone{bc.offset}"
        )
        isnone = ir.Expr.binop(
            op, lhs=self.vsmap[pred], rhs=none, loc=self.loc
        )
        self.branch_predicate = self.store(isnone, "$jump_if")

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, op: Op, bc: dis.Instruction):
        [_env, pred] = op.inputs
        [_env] = op.outputs
        op = BINOPS_TO_OPERATORS["is not"]
        none = self.store(
            value=ir.Const(None, loc=self.loc), name=f"$constNone{bc.offset}"
        )
        isnotnone = ir.Expr.binop(
            op, lhs=self.vsmap[pred], rhs=none, loc=self.loc
        )
        self.branch_predicate = self.store(isnotnone, "$jump_if")

    op_POP_JUMP_BACKWARD_IF_TRUE = op_POP_JUMP_FORWARD_IF_TRUE

    def op_RETURN_VALUE(self, op: Op, bc: dis.Instruction):
        [_env, retval] = op.inputs
        self.append(ir.Return(self.vsmap[retval], loc=self.loc))
        assert self.current_block.is_terminated

    def op_RAISE_VARARGS(self, op: Op, bc: dis.Instruction):
        [_env, exc] = op.inputs
        # XXX: temporary implementation
        self.append(ir.Raise(exception=self.vsmap[exc], loc=self.loc))
        assert self.current_block.is_terminated


def rvsdg_to_ir(
    func_id: bytecode.FunctionIdentity, rvsdg: SCFG
) -> ir.FunctionIR:
    rvsdg2ir = RVSDG2IR(func_id)
    data = rvsdg2ir.initialize()
    rvsdg2ir.visit_graph(rvsdg, data)
    rvsdg2ir.finalize()

    for blk in rvsdg2ir.blocks.values():
        blk.verify()

    defs = ir_utils.build_definitions(rvsdg2ir.blocks)

    fir = ir.FunctionIR(
        blocks=rvsdg2ir.blocks,
        is_generator=False,
        func_id=func_id,
        loc=rvsdg2ir.first_loc,
        definitions=defs,
        arg_count=len(func_id.arg_names),
        arg_names=func_id.arg_names,
    )
    # fir.dump()
    if DEBUG_GRAPH:
        fir.render_dot().view()
    return fir
