from dataclasses import dataclass, replace, field, fields
import dis
from contextlib import contextmanager
from typing import Optional, Iterator, Protocol, runtime_checkable
from collections import (
    deque,
    defaultdict,
)
from collections.abc import Mapping

from . import bcinterp

from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import (
    SCFG,
    ConcealedRegionView,
)
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    PythonBytecodeBlock,
    RegionBlock,
)
from numba_rvsdg.core.datastructures.labels import Label
from numba_rvsdg.rendering.rendering import ByteFlowRenderer

from .renderer import RvsdgRenderer



@dataclass(frozen=True)
class ValueState:
    parent: Optional["Op"]
    name: str
    out_index: int
    is_effect: bool = False

    def short_identity(self) -> str:
        return f"ValueState({id(self.parent):x}, {self.name}, {self.out_index})"


@dataclass(frozen=True)
class Op:
    opname: str
    bc_inst: Optional[dis.Instruction]
    _inputs: dict[str, ValueState] = field(default_factory=dict)
    _outputs: dict[str, ValueState] = field(default_factory=dict)

    def add_input(self, name, vs: ValueState):
        self._inputs[name] = vs

    def add_output(self, name: str, is_effect=False) -> ValueState:
        vs = ValueState(parent=self, name=name, out_index=len(self._outputs), is_effect=is_effect)
        self._outputs[name] = vs
        return vs

    def short_identity(self) -> str:
        return f"Op({self.opname}, {id(self):x})"

    @property
    def outputs(self):
        return list(self._outputs.values())


@runtime_checkable
class DDGProtocol(Protocol):
    in_vars: set[str]
    out_vars: set[str]


@dataclass(frozen=True)
class DDGRegion(RegionBlock):
    in_vars: set[str] = field(default_factory=set)
    out_vars: set[str] = field(default_factory=set)

    @contextmanager
    def render_rvsdg(self, renderer, digraph, label):
        with digraph.subgraph(name=f"cluster_rvsdg_{id(self)}") as subg:
            subg.attr(color="black", label="region", bgcolor="grey")
            subg.node(f"incoming_{id(self)}", label=f"{'|'.join([f'<{k}> {k}' for k in self.in_vars])}", shape='record', rank="min")
            subg.edge(f"incoming_{id(self)}", f"cluster_{label}", style="invis")
            yield subg
            subg.edge(f"cluster_{label}", f"outgoing_{id(self)}", style="invis")
            subg.node(f"outgoing_{id(self)}", label=f"{'|'.join([f'<{k}> {k}' for k in self.out_vars])}", shape='record', rank="max")


@dataclass(frozen=True)
class DDGBlock(BasicBlock):
    in_effect: ValueState | None = None
    out_effect: ValueState | None = None
    in_stackvars: list[ValueState] = field(default_factory=list)
    out_stackvars: list[ValueState] = field(default_factory=list)
    in_vars: dict[str, ValueState] = field(default_factory=dict)
    out_vars: dict[str, ValueState] = field(default_factory=dict)

    def render_rvsdg(self, renderer, digraph, label):
        with digraph.subgraph(name="cluster_"+str(label)) as g:
            g.attr(color='lightgrey')
            g.attr(label=str(label))
            # render body
            self.render_valuestate(renderer, g, self.in_effect)
            self.render_valuestate(renderer, g, self.out_effect)
            for vs in self.in_stackvars:
                self.render_valuestate(renderer, g, vs)
            for vs in self.out_stackvars:
                self.render_valuestate(renderer, g, vs)
            for vs in self.out_vars.values():
                self.render_valuestate(renderer, g, vs)
            # Fill incoming
            in_vars_fields = "incoming-vars|" + "|".join([f"<{x}> {x}" for x in self.in_vars])
            fields = "|" + in_vars_fields
            g.node(f"incoming_{id(self)}", shape="record", label=f"{fields}", rank="source")

            for vs in self.in_vars.values():
                self.add_vs_edge(renderer, f"incoming_{id(self)}:{vs.name}", vs.parent.short_identity())

            # Fill outgoing
            out_stackvars_fields = "outgoing-stack|" + "|".join([f"<{x.short_identity()}> {x.name}" for x in self.out_stackvars])
            out_vars_fields = "outgoing-vars|" + "|".join([f"<{x.name}> {x.name}" for x in self.out_vars.values()])
            fields = f"<{self.out_effect.short_identity()}> env" + "|" + out_stackvars_fields + "|" + out_vars_fields
            g.node(f"outgoing_{id(self)}", shape="record", label=f"{fields}")
            for vs in self.out_stackvars:
                self.add_vs_edge(renderer, vs, f"outgoing_{id(self)}:{vs.name}")
            for vs in self.out_vars.values():
                self.add_vs_edge(renderer, vs, f"outgoing_{id(self)}:{vs.name}")
            self.add_vs_edge(renderer, self.out_effect, f"outgoing_{id(self)}:{self.out_effect.short_identity()}")
            # Draw "head"
            g.node(str(label), shape="doublecircle", label="")

    def render_valuestate(self, renderer, digraph, vs: ValueState, *, follow=True):
        if vs.is_effect:
            digraph.node(vs.short_identity(), shape="circle", label=str(vs.name))
        else:
            digraph.node(vs.short_identity(), shape="rect", label=str(vs.name))
        if follow and vs.parent is not None:
            op = vs.parent
            self.render_op(renderer, digraph, op)

    def render_op(self, renderer, digraph, op: Op):
        op_anchor = op.short_identity()
        digraph.node(op_anchor, label=op_anchor,
                     shape="box", style="rounded")
        for vs in op._outputs.values():
            self.add_vs_edge(renderer, op_anchor, vs)
            self.render_valuestate(renderer, digraph, vs, follow=False)
        for vs in op._inputs.values():
            self.add_vs_edge(renderer, vs, op_anchor)
            self.render_valuestate(renderer, digraph, vs)

    def add_vs_edge(self, renderer, src, dst):
        is_effect = (isinstance(src, ValueState) and src.is_effect) or (isinstance(dst, ValueState) and dst.is_effect)
        if isinstance(src, ValueState):
            src = src.short_identity()
        if isinstance(dst, ValueState):
            dst = dst.short_identity()
        if is_effect:
            kwargs = dict(style="dotted")
        else:
            kwargs = {}
        renderer.add_edge(src, dst, **kwargs)



def render_scfg(byteflow):
    bfr = ByteFlowRenderer()
    bfr.bcmap_from_bytecode(byteflow.bc)
    bfr.render_scfg(byteflow.scfg).view("scfg")


def build_rvsdg(code):
    byteflow = ByteFlow.from_bytecode(code)
    byteflow = byteflow.restructure()
    rvsdg = convert_to_dataflow(byteflow)
    rvsdg = propagate_states(rvsdg)
    RvsdgRenderer().render_rvsdg(rvsdg).view("rvsdg")


def _compute_incoming_labels(graph: Mapping[Label, BasicBlock]) -> dict[Label, set[Label]]:
    jump_table: dict[Label, set[Label]] = {}
    blk: BasicBlock
    for k in graph:
        jump_table[k] = set()
    for blk in graph.values():
        for dst in blk.jump_targets:
            jump_table[dst].add(blk.label)
    return jump_table

def _flatten_full_graph(scfg: SCFG):
    from collections import ChainMap
    regions = [_flatten_full_graph(elem.subregion)
               for elem in scfg.graph.values()
               if isinstance(elem, RegionBlock)]
    out = ChainMap(*regions, scfg.graph)
    for blk in out.values():
        assert not isinstance(blk, RegionBlock), type(blk)
    return out

def view_toposorted_ddgblock_only(rvsdg: SCFG) -> list[list[DDGBlock]]:
    """Return toposorted nested list of DDGBlock
    """
    graph = _flatten_full_graph(rvsdg)
    incoming_labels = _compute_incoming_labels(graph)
    visited: set[Label] = set()
    toposorted: list[list[Label]] = []

    # Toposort
    while incoming_labels:
        level = []
        for k, vs in incoming_labels.items():
            if not (vs - visited):
                # all incoming visited
                level.append(k)
        for k in level:
            del incoming_labels[k]
        visited |= set(level)
        toposorted.append(level)

    # Filter
    output: list[list[DDGBlock]] = []
    for level in toposorted:
        filtered = [graph[k] for k in level if isinstance(graph[k], DDGBlock)]
        if filtered:
            output.append(filtered)

    return output



# def _traverse_tree(graph: Mapping[Label, BasicBlock]):
#     """BFS
#     """
#     def _find_head(incoming_labels):
#         for k, vs in incoming_labels.items():
#             if not vs:
#                 return k
#         raise Exception("unreachable")

#     incoming_labels = _compute_incoming_labels(graph)
#     pending = deque([graph[_find_head(incoming_labels)]])
#     visited = set()
#     while pending:
#         node = pending.popleft()
#         incomings = incoming_labels[node.label]
#         assert len(incomings - visited) == 0, "all incomings must be already visited"
#         yield incoming_labels[node.label], node
#         visited.add(node.label)
#         pending.extend([graph[x] for x in node.jump_targets if x not in visited])


def convert_to_dataflow(byteflow: ByteFlow) -> SCFG:
    bcmap = {inst.offset: inst for inst in byteflow.bc}
    rvsdg = convert_scfg_to_dataflow(byteflow.scfg, bcmap)
    return rvsdg

def propagate_states(rvsdg: SCFG) -> SCFG:
    propagate_states_ddgblock_only_inplace(rvsdg)
    propagate_states_to_parent_region_inplace(rvsdg)
    propagate_states_to_outgoing_inplace(rvsdg)
    return rvsdg

def propagate_states_ddgblock_only_inplace(rvsdg: SCFG):
    # Propagate the outgoing states
    topo_ddgblocks = view_toposorted_ddgblock_only(rvsdg)
    block_vars: dict[Label, set[str]] = {}
    live_vars: set[str] = set()
    for blklevel in topo_ddgblocks:
        new_vars: set[str] = set()
        for blk in blklevel:
            block_vars[blk.label] = live_vars.copy()
            new_vars |= set(blk.out_vars)
        live_vars |= new_vars

    # Apply changes
    for blklevel in topo_ddgblocks:
        for blk in blklevel:
            extra_vars = block_vars[blk.label] - set(blk.in_vars)
            for k in extra_vars:
                op = Op(opname="var.incoming", bc_inst=None)
                vs = op.add_output(k)
                blk.in_vars[k] = vs
                blk.out_vars[k] = vs


def _walk_all_regions(scfg: SCFG) -> Iterator[RegionBlock]:
    for blk in scfg.graph.values():
        if isinstance(blk, RegionBlock):
            yield from _walk_all_regions(blk.subregion)
            yield blk


def propagate_states_to_parent_region_inplace(rvsdg: SCFG):
    for reg in _walk_all_regions(rvsdg):
        assert isinstance(reg, DDGRegion)
        subregion: SCFG = reg.subregion
        head = subregion[subregion.find_head()]
        exit = subregion[reg.exiting]
        if isinstance(head, DDGProtocol):
            reg.in_vars.update(head.in_vars)
        if isinstance(exit, DDGProtocol):
            reg.out_vars.update(reg.in_vars)
            reg.out_vars.update(exit.out_vars)


def propagate_states_to_outgoing_inplace(rvsdg: SCFG):
    for src in rvsdg.graph.values():
        for dst_label in src.jump_targets:
            if dst_label in rvsdg.graph:
                dst = rvsdg.graph[dst_label]
                if isinstance(dst, DDGRegion):
                    dst.in_vars.update(src.out_vars)
                    dst.out_vars.update(dst.in_vars)
                    propagate_states_to_outgoing_inplace(dst.subregion)



def convert_scfg_to_dataflow(scfg, bcmap) -> SCFG:
    rvsdg = SCFG()
    for block in scfg.graph.values():
        # convert block
        if isinstance(block, PythonBytecodeBlock):
            ddg = convert_bc_to_ddg(block, bcmap)
            rvsdg.add_block(ddg)
        elif isinstance(block, RegionBlock):
            # Inside-out
            subregion = convert_scfg_to_dataflow(block.subregion, bcmap)
            newattrs = {fd.name: getattr(block, fd.name) for fd in fields(block)}
            newattrs.update(subregion=subregion)
            newblk = DDGRegion(**newattrs)
            rvsdg.add_block(newblk)
        else:
            rvsdg.add_block(block)

    return rvsdg


def convert_bc_to_ddg(block: PythonBytecodeBlock, bcmap: dict[int, dis.Bytecode]):
    instlist = block.get_instructions(bcmap)
    converter = BC2DDG()
    in_effect = converter.effect
    for inst in instlist:
        converter.convert(inst)
    blk = DDGBlock(
        label=block.label,
        _jump_targets=block._jump_targets,
        backedges=block.backedges,
        in_effect=in_effect,
        out_effect=converter.effect,
        in_stackvars=list(converter.incoming_stackvars),
        out_stackvars=list(converter.stack),
        in_vars=converter.incoming_vars.copy(),
        out_vars=converter.varmap.copy(),
    )

    return blk

class BC2DDG:
    def __init__(self):
        self.stack: list[ValueState] = []
        start_env = Op("start", bc_inst=None)
        self.effect = start_env.add_output("env", is_effect=True)
        self.varmap: dict[str, ValueState] = {}
        self.incoming_vars: dict[str, ValueState] = {}
        self.incoming_stackvars: list[ValueState] = []

    def push(self, val: ValueState):
        self.stack.append(val)

    def pop(self) -> ValueState:
        if not self.stack:
            op = Op(opname="stack.incoming", bc_inst=None)
            vs = op.add_output(f"stack[{len(self.incoming_stackvars)}]")
            self.stack.append(vs)
            self.incoming_stackvars.append(vs)
        return self.stack.pop()

    def store(self, varname: str, value: ValueState):
        self.varmap[varname] = value

    def load(self, varname: str) -> ValueState:
        if varname not in self.varmap:
            op = Op(opname="var.incoming", bc_inst=None)
            vs = op.add_output(varname)
            self.incoming_vars[varname] = vs
            self.varmap[varname] = vs

        return self.varmap[varname]

    def replace_effect(self, env: ValueState):
        assert env.is_effect
        self.effect = env

    def convert(self, inst: dis.Instruction):
        fn = getattr(self, f"op_{inst.opname}")
        fn(inst)

    def op_RESUME(self, inst: dis.Instruction):
        pass   # no-op

    def op_LOAD_GLOBAL(self, inst: dis.Instruction):
        op = Op(opname="global", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_output("value")
        self.push(op.add_output("out"))

    def op_LOAD_CONST(self, inst: dis.Instruction):
        op = Op(opname="const", bc_inst=inst)
        self.push(op.add_output("out"))

    def op_STORE_FAST(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname="store", bc_inst=inst)
        op.add_input("value", tos)
        self.store(inst.argval, op.add_output(inst.argval))

    def op_LOAD_FAST(self, inst: dis.Instruction):
        self.push(self.load(inst.argval))

    def op_PRECALL(self, inst: dis.Instruction):
        pass # no-op

    def op_CALL(self, inst: dis.Instruction):
        argc: int = inst.argval
        callable = self.pop()  # TODO
        arg0 = self.pop() # TODO
        # TODO: handle kwnames
        args = reversed([arg0, *[self.pop() for _ in range(argc)]])
        op = Op(opname="call", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("callee", callable)
        for i, arg in enumerate(args):
            op.add_input(f"arg.{i}", arg)
        self.replace_effect(op.add_output("env", is_effect=True))
        self.push(op.add_output("ret"))

    def op_GET_ITER(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname="getiter", bc_inst=inst)
        op.add_input("obj", tos)
        self.push(op.add_output("iter"))

    def op_FOR_ITER(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname="foriter", bc_inst=inst)
        op.add_input("iter", tos)
        self.push(op.add_output("indvar"))

    def op_BINARY_OP(self, inst: dis.Instruction):
        rhs = self.pop()
        lhs = self.pop()
        op = Op(opname="binaryop", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("lhs", lhs)
        op.add_input("rhs", rhs)
        self.replace_effect(op.add_output("env", is_effect=True))
        self.push(op.add_output("out"))

    def op_RETURN_VALUE(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname="ret", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("retval", tos)
        self.replace_effect(op.add_output("env", is_effect=True))

    def op_JUMP_BACKWARD(self, inst: dis.Instruction):
        pass # no-op


def run_frontend(func): #, inline_closures=False, emit_dels=False):
    # func_id = bytecode.FunctionIdentity.from_function(func)

    rvsdg = build_rvsdg(func.__code__)

    return rvsdg
    # bc = bytecode.ByteCode(func_id=func_id)
    # interp = bcinterp.Interpreter(func_id)
    # func_ir = interp.interpret(bc)
    # return func_ir
