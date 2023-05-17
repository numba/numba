import dis
from contextlib import contextmanager
import builtins

from numba.core import (
    ir,
    bytecode,
)

from .bc2rvsdg import (
    build_rvsdg,
    SCFG,
    BasicBlock,
    RegionBlock,
    DDGBlock,
    Op,
    ValueState,
)
from .regionpasses import RegionVisitor


def run_frontend(func):


    rvsdg = build_rvsdg(func.__code__)

    func_id = bytecode.FunctionIdentity.from_function(func)
    func_ir = rvsdg_to_ir(func_id, rvsdg)

    return rvsdg
    # bc = bytecode.ByteCode(func_id=func_id)
    # interp = bcinterp.Interpreter(func_id)
    # func_ir = interp.interpret(bc)
    # return func_ir


def _get_first_bytecode(ops: list[Op]) -> dis.Instruction:
    return next(iter(op.bc_inst for op in ops if op.bc_inst is not None))


_noop = {"var.incoming", "start"}

class RVSDG2IR(RegionVisitor):
    blocks: dict[int, ir.Block]
    func_id: bytecode.FunctionIdentity
    scope: ir.Scope
    global_scope: ir.Scope
    vsmap: dict[ValueState, ir.Var]
    current_block: ir.Block | None

    def __init__(self, func_id):
        self.func_id = func_id
        self.loc = self.first_loc = ir.Loc.from_function_id(func_id)
        self.global_scope = ir.Scope(parent=None, loc=self.loc)
        self.scope = ir.Scope(parent=self.global_scope, loc=self.loc)
        self.blocks = {}
        self.vsmap = {}
        self.current_block = None

    def initialize(self):
        with self.set_block(ir.Block(scope=self.scope, loc=self.loc)):
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
            firstbc: dis.Instruction = _get_first_bytecode(ops)
            assert firstbc.positions is not None
            self.loc = self.loc.with_lineno(firstbc.positions.lineno, firstbc.positions.col_offset)
            with self.set_block(ir.Block(scope=self.scope, loc=self.loc)) as curblk:
                for op in ops:
                    if op.opname in _noop:
                        pass
                    elif op.bc_inst is not None:
                        self.interpret_bytecode(op)
                    elif op.opname == "stack.export":
                        [arg] = op.inputs
                        [res] = op.outputs
                        self.vsmap[res] = self.store(self.vsmap[arg], f"${res.name}")
                    else:
                        raise NotImplementedError(op.opname, op)
            data = {k: self.vsmap[vs] for k, vs in block.out_vars.items()}
            curblk.dump()
            # Prepare outgoing variables
            return data
        else:
            raise NotImplementedError(block.label, type(block))

    def visit_loop(self, region: RegionBlock, data):
        raise

    def visit_switch(self, region: RegionBlock, data):
        raise

    @contextmanager
    def set_block(self, block: ir.Block) -> ir.Block:
        old = self.current_block
        self.current_block = block
        try:
            yield block
        finally:
            self.current_block = old

    def store(self, value, name) -> ir.Var:
        target = self.scope.redefine(name, loc=self.loc)
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        self.current_block.append(stmt)
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

    def op_GET_ITER(self, op: Op, bc: dis.Instruction):
        [arg] = op.inputs
        [res] = op.outputs
        expr = ir.Expr.getiter(value=self.vsmap[arg], loc=self.loc)
        self.vsmap[res] = self.store(expr, f"${res.name}")

def rvsdg_to_ir(
        func_id: bytecode.FunctionIdentity,
        rvsdg: SCFG
    ):
    rvsdg2ir = RVSDG2IR(func_id)
    data = rvsdg2ir.initialize()
    rvsdg2ir.visit_graph(rvsdg, data)
