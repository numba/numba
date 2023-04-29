
import logging
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    PythonBytecodeBlock,
    ControlVariableBlock,
    BranchBlock,
)
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.labels import (
    Label,
    PythonBytecodeLabel,
    ControlLabel,
)
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
import dis
from typing import Dict


class RvsdgRenderer(object):
    def __init__(self):
        from graphviz import Digraph

        self.g = Digraph()
        self.edges = {}

    def add_edge(self, src: str, dst: str, **attrs):
        self.edges[src, dst] = attrs

    def render_region_block(
        self, digraph: "Digraph", label: Label, regionblock: RegionBlock
    ):
        # render subgraph
        graph = regionblock.get_full_graph()
        with digraph.subgraph(name=f"cluster_{label}") as subg:
            color = "blue"
            if regionblock.kind == "branch":
                color = "green"
            if regionblock.kind == "tail":
                color = "purple"
            if regionblock.kind == "head":
                color = "red"
            subg.attr(color=color, label=regionblock.kind)
            for label, block in graph.items():
                self.render_block(subg, label, block)
        # render edges within this region
        self.render_edges(graph)

    def render_basic_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if isinstance(label, PythonBytecodeLabel):
            instlist = block.get_instructions(self.bcmap)
            body = label.__class__.__name__ + ": " + str(label.index) + "\l"
            body += "\l".join(
                [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
            )
        elif isinstance(label, ControlLabel):
            body = label.__class__.__name__ + ": " + str(label.index)
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_control_variable_block(
        self, digraph: "Digraph", label: Label, block: BasicBlock
    ):
        if isinstance(label, ControlLabel):
            body = label.__class__.__name__ + ": " + str(label.index) + "\l"
            body += "\l".join(
                (f"{k} = {v}" for k, v in block.variable_assignment.items())
            )
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_branching_block(
        self, digraph: "Digraph", label: Label, block: BasicBlock
    ):
        if isinstance(label, ControlLabel):

            def find_index(v):
                if hasattr(v, "offset"):
                    return v.offset
                if hasattr(v, "index"):
                    return v.index

            body = label.__class__.__name__ + ": " + str(label.index) + "\l"
            body += f"variable: {block.variable}\l"
            body += "\l".join(
                (f"{k}=>{find_index(v)}" for k, v in block.branch_value_table.items())
            )
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if type(block) == BasicBlock:
            self.render_basic_block(digraph, label, block)
        elif type(block) == ControlVariableBlock:
            self.render_control_variable_block(digraph, label, block)
        elif type(block) == BranchBlock:
            self.render_branching_block(digraph, label, block)
        elif type(block) == RegionBlock:
            self.render_region_block(digraph, label, block)
        else:
            block.render_rvsdg(self, digraph, label, block)

    def render_edges(self, blocks: Dict[Label, BasicBlock]):
        for label, block in blocks.items():
            for dst in block.jump_targets:
                if dst in blocks:
                    if type(block) == RegionBlock:
                        if block.exiting is not None:
                            self.g.edge(str(block.exiting), str(dst))
                        else:
                            self.g.edge(str(label), str(dst))
                    elif isinstance(block, BasicBlock):
                        self.g.edge(str(label), str(dst))
                    else:
                        raise Exception("unreachable")
            for dst in block.backedges:
                # assert dst in blocks
                self.g.edge(
                    str(label), str(dst), style="dashed", color="grey", constraint="0"
                )
        for (src, dst), attrs in self.edges.items():
            self.g.edge(src, dst, **attrs)
        self.edges.clear()

    def render_byteflow(self, byteflow: ByteFlow):
        self.bcmap_from_bytecode(byteflow.bc)

        # render nodes
        for label, block in byteflow.scfg.graph.items():
            self.render_block(self.g, label, block)
        self.render_edges(byteflow.scfg.graph)
        return self.g

    def render_scfg(self, scfg):
        # render nodes
        for label, block in scfg.graph.items():
            self.render_block(self.g, label, block)
        self.render_edges(scfg.graph)
        return self.g


def render_func(func):
    render_flow(ByteFlow.from_bytecode(func))

def render_flow(flow):
    ByteFlowRenderer().render_byteflow(flow).view("before")

    cflow = flow._join_returns()
    ByteFlowRenderer().render_byteflow(cflow).view("closed")

    lflow = cflow._restructure_loop()
    ByteFlowRenderer().render_byteflow(lflow).view("loop restructured")

    bflow = lflow._restructure_branch()
    ByteFlowRenderer().render_byteflow(bflow).view("branch restructured")

def render_scfg(scfg):
    ByteFlowRenderer().render_scfg(scfg).view("scfg")
