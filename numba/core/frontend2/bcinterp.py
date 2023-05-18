import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce

from numba.core import (
    ir,
    bytecode,
    ir_utils,
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
    Label,
)
from .regionpasses import RegionVisitor


def run_frontend(func):


    rvsdg = build_rvsdg(func.__code__)

    func_id = bytecode.FunctionIdentity.from_function(func)
    func_ir = rvsdg_to_ir(func_id, rvsdg)

    return func_ir
    # bc = bytecode.ByteCode(func_id=func_id)
    # interp = bcinterp.Interpreter(func_id)
    # func_ir = interp.interpret(bc)
    # return func_ir


def _get_first_bytecode(ops: list[Op]) -> dis.Instruction|None:
    for bc in (op.bc_inst for op in ops if op.bc_inst is not None):
        return bc


# def _innermost_header(region: RegionBlock) -> BasicBlock:
#     while True:
#         [header] = region.headers
#         region = region.subregion.graph[header]
#         if not isinstance(region, RegionBlock):
#             return region


# def _innermost_exiting(region: RegionBlock) -> BasicBlock:
#     while True:
#         region = region.subregion.graph[region.exiting]
#         if not isinstance(region, RegionBlock):
#             return region

def _get_label(label: Label) -> int:
    # if isinstance(block, DDGBlock):
    #     ops = block.get_toposorted_ops()
    #     firstbc: dis.Instruction|None = _get_first_bytecode(ops)
    #     if firstbc is not None:
    #         return firstbc
    return id(label)


def _get_phi_name(varname: str, label: Label) -> str:
    return f"$phi.{varname}.{id(label)}"


_noop = {"var.incoming", "start"}

class RVSDG2IR(RegionVisitor):
    blocks: dict[int, ir.Block]
    func_id: bytecode.FunctionIdentity
    scope: ir.Scope
    global_scope: ir.Scope
    vsmap: dict[ValueState, ir.Var]
    _current_block: ir.Block | None
    last_block_label: int | None

    branch_predicate: ir.Var|None

    def __init__(self, func_id):
        self.func_id = func_id
        self.loc = self.first_loc = ir.Loc.from_function_id(func_id)
        self.global_scope = ir.Scope(parent=None, loc=self.loc)
        self.scope = ir.Scope(parent=self.global_scope, loc=self.loc)
        self.blocks = {}
        self.vsmap = {}
        self._current_block = None
        self.last_block_label = None

    @property
    def current_block(self) -> ir.Block:
        out = self._current_block
        assert out is not None
        return out

    @property
    def last_block(self) -> ir.Block:
        return self.blocks[self.last_block_label]

    def initialize(self):
        with self.set_block(0, ir.Block(scope=self.scope, loc=self.loc)):
            data = {}
            for i, k in enumerate(self.func_id.arg_names):
                val = ir.Arg(index=i, name=k, loc=self.loc)
                data[f"var.{k}"] = self.store(val, str(k))
            return data

    def visit_block(self, block: BasicBlock, data):
        if isinstance(block, DDGBlock):
            # Prepare incoming variables
            for k, vs in block.in_vars.items():
                self.vsmap[vs] = data[k]

            # Emit instruction body
            ops = block.get_toposorted_ops()
            firstbc: dis.Instruction|None = _get_first_bytecode(ops)
            if firstbc is not None:
                assert firstbc.positions is not None
                self.loc = self.loc.with_lineno(
                    firstbc.positions.lineno,
                    firstbc.positions.col_offset,
                )
            with self.set_block(
                        _get_label(block.label),
                        ir.Block(scope=self.scope, loc=self.loc)):
                for op in ops:
                    if op.opname in _noop:
                        pass
                    elif op.bc_inst is not None:
                        self.interpret_bytecode(op)
                    elif op.opname in {"stack.export", "stack.incoming"}:
                        [arg] = op.inputs
                        [res] = op.outputs
                        self.vsmap[res] = self.store(self.vsmap[arg], f"${res.name}")
                    else:
                        raise NotImplementedError(op.opname, op)

            # Prepare outgoing variables
            data = {k: self.vsmap[vs] for k, vs in block.out_vars.items()}
            return data
        elif isinstance(block, DDGControlVariable):
            # Emit body
            with self.set_block(
                        _get_label(block.label),
                        ir.Block(scope=self.scope, loc=self.loc)):
                for cp, v in block.variable_assignment.items():
                    const = ir.Const(v, loc=self.loc, use_literal_type=False)
                    self.store(const, f"$.cp.{cp}", redefine=False)
            return data
        elif isinstance(block, DDGBranch):
            # Emit body
            with self.set_block(
                        _get_label(block.label),
                        ir.Block(scope=self.scope, loc=self.loc)):

                assert len(block.branch_value_table) == 2, block.branch_value_table
                cp = block.variable
                cpvar = self.scope.get_exact(f"$.cp.{cp}")
                truebr = _get_label(block.branch_value_table[1])
                falsebr = _get_label(block.branch_value_table[0])
                br = ir.Branch(
                    cond=cpvar,
                    truebr=truebr,
                    falsebr=falsebr,
                    loc=self.loc,
                )
                self.current_block.append(br)

            return data
        else:
            raise NotImplementedError(block.label, type(block))

    def visit_loop(self, region: RegionBlock, data):
        assert isinstance(region, DDGRegion)
        # Prepare incoming states
        inner_data = {}
        for k in  region.incoming_states:
            inner_data[k] = self.store(
                data[k], _get_phi_name(k, region.label), redefine=False,
                block=self.last_block)

        # Emit loop body
        out_data = self.visit_linear(region, inner_data)

        # Prepare outgoing states
        exit_data = {}
        for k in  region.outgoing_states:
            exit_data[k] = self.store(
                out_data[k], _get_phi_name(k, region.label), redefine=False,
                block=self.last_block)

        return exit_data

    def visit_switch(self, region: RegionBlock, data):
        # Emit header
        [header] = region.headers
        header_block = region.subregion[header]
        assert header_block.kind == 'head'
        self.branch_predicate = None
        data_at_head = self.visit_linear(header_block, data)
        assert not self.last_block.is_terminated
        assert self.branch_predicate is not None

        # XXX: how to tell which target is which??
        truebr = _get_label(header_block._jump_targets[0])
        falsebr = _get_label(header_block._jump_targets[1])
        br = ir.Branch(self.branch_predicate, truebr, falsebr, loc=self.loc)
        self.last_block.append(br)

        # Emit branches
        data_for_branches = []
        branch_blocks = []
        for blk in region.subregion.graph.values():
            if blk.kind == "branch":
                branch_blocks.append(blk.subregion.graph[blk.exiting])
                data_for_branches.append(
                    self.visit_linear(blk, data_at_head)
                )
                # Add jump to tail
                [target] = blk.jump_targets
                assert not self.last_block.is_terminated
                self.last_block.append(ir.Jump(_get_label(target), loc=self.loc))

        # handle outgoing values from the branches
        names = reduce(operator.or_, map(set, data_for_branches))
        for blk, branch_data in zip(branch_blocks,
                                    data_for_branches, strict=True):
            for k in names:
                if k in branch_data:
                    # Insert stores to export
                    self.store(branch_data[k], _get_phi_name(k, region.label),
                               redefine=False,
                               block=self.blocks[_get_label(blk.label)])
        data_after_branches = {k: self.scope.get_exact(_get_phi_name(k, region.label))
                               for k in names}

        # Emit tail
        exiting = region.exiting
        exiting_block = region.subregion[exiting]
        assert exiting_block.kind == 'tail'
        data_at_tail = self.visit_linear(exiting_block, data_after_branches)

        return data_at_tail

    @contextmanager
    def set_block(self, label: int, block: ir.Block) -> Iterator[ir.Block]:
        if self.last_block_label is not None:
            last_block = self.blocks[self.last_block_label]
            if not last_block.is_terminated:
                last_block.append(ir.Jump(label, loc=self.loc))

            print("begin dump last blk".center(80, '-'))
            last_block.dump()
            print("end dump last blk".center(80, '='))

        self.blocks[label] = block
        old = self._current_block
        self._current_block = block
        try:
            yield block
        finally:
            self.last_block_label = label
            self._current_block = old
            # dump
            print(f"begin dump blk: {label}".center(80, '-'))
            block.dump()
            print("end dump blk".center(80, '='))

    def store(self, value, name, *, redefine=True, block=None) -> ir.Var:
        if redefine:
            target = self.scope.redefine(name, loc=self.loc)
        else:
            target = self.scope.get_or_define(name, loc=self.loc)
        if block is None:
            block = self.current_block
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        if block.is_terminated:
            block.insert_before_terminator(stmt)
        else:
            block.append(stmt)
        return target

    def get_global_value(self, name):
        """THIS IS COPIED from interpreter.py

        Get a global value from the func_global (first) or
        as a builtins (second).  If both failed, return a ir.UNDEFINED.
        """
        try:
            return self.func_id.func.__globals__[name]
        except KeyError:
            return getattr(builtins, name, ir.UNDEFINED)

    def interpret_bytecode(self, op: Op):
        assert op.bc_inst is not None
        pos = op.bc_inst.positions
        self.loc = self.loc.with_lineno(pos.lineno, pos.col_offset)
        # dispatch
        fn = getattr(self, f"op_{op.bc_inst.opname}")
        fn(op, op.bc_inst)

    def op_LOAD_CONST(self, op: Op, bc: dis.Instruction):
        assert not op.inputs
        [vs] = op.outputs
        # TODO: handle non scalar
        value = ir.Const(bc.argval, loc=self.loc)
        self.vsmap[vs] = self.store(value, f"${vs.name}")

    def op_LOAD_GLOBAL(self, op: Op, bc: dis.Instruction):
        # intentionally ignoring the nil
        [_nil, res] = op.outputs
        value = self.get_global_value(bc.argval)
        # TODO: handle non scalar
        const = ir.Const(value, loc=self.loc)
        self.vsmap[res] = self.store(const, f"${res.name}")

    def op_STORE_FAST(self, op: Op, bc: dis.Instruction):
        [incvar] = op.inputs
        [res] = op.outputs
        var = self.vsmap[incvar]
        self.vsmap[res] = self.store(var, bc.argval)

    def op_CALL(self, op: Op, bc: dis.Instruction):
        [_env, callee, arg0, *args] = op.inputs
        [_env, res] = op.outputs
        assert callee.name == "nil"
        callee = arg0
        callee = self.vsmap[callee]
        args = [self.vsmap[vs] for vs in args]
        kwargs = () # TODO
        expr = ir.Expr.call(callee, args, kwargs, loc=self.loc)
        self.vsmap[res] = self.store(expr, f"${res.name}")

    def op_BINARY_OP(self, op: Op, bc: dis.Instruction):
        [_env, lhs, rhs] = op.inputs
        [_env, out] = op.outputs
        operator = bc.argrepr
        if "=" in operator:
            operator = operator[:-1]
            immuop = BINOPS_TO_OPERATORS[operator]
            op = INPLACE_BINOPS_TO_OPERATORS[operator + '=']
            expr = ir.Expr.inplace_binop(
                op,
                immuop,
                lhs=self.vsmap[lhs],
                rhs=self.vsmap[rhs],
                loc=self.loc,
            )
            self.vsmap[out] = self.store(expr, f"${out.name}")
        else:
            raise NotImplementedError

    def op_GET_ITER(self, op: Op, bc: dis.Instruction):
        [arg] = op.inputs
        [res] = op.outputs
        expr = ir.Expr.getiter(value=self.vsmap[arg], loc=self.loc)
        self.vsmap[res] = self.store(expr, f"${res.name}")

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
        self.branch_predicate = pred

    def op_RETURN_VALUE(self, op: Op, bc: dis.Instruction):
        [_env, retval] = op.inputs
        self.current_block.append(ir.Return(self.vsmap[retval], loc=self.loc))


def rvsdg_to_ir(
        func_id: bytecode.FunctionIdentity,
        rvsdg: SCFG
    ) -> ir.FunctionIR:
    rvsdg2ir = RVSDG2IR(func_id)
    data = rvsdg2ir.initialize()
    rvsdg2ir.visit_graph(rvsdg, data)

    for blk in rvsdg2ir.blocks.values():
        blk.verify()

    defs = ir_utils.build_definitions(rvsdg2ir.blocks)

    fir = ir.FunctionIR(
        blocks=rvsdg2ir.blocks, is_generator=False, func_id=func_id,
        loc=rvsdg2ir.first_loc, definitions=defs,
        arg_count=len(func_id.arg_names), arg_names=func_id.arg_names)
    fir.dump()
    fir.render_dot().view()
    return fir