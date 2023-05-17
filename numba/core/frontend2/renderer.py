
import logging
import contextlib
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
from typing import Dict, Callable


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

        def render_subgraph(digraph, label):
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
                subg.attr(color=color, label=regionblock.kind, bgcolor="white")
                for label, block in graph.items():
                    self.render_block(subg, label, block)
            # render edges within this region
            self.render_edges(graph)

        if hasattr(regionblock, "render_rvsdg"):
            with regionblock.render_rvsdg(self, digraph, label) as digraph:
                render_subgraph(digraph, label)
        else:
            render_subgraph(digraph, label)

    def render_region_block(
        self, digraph: "Digraph", label: Label, regionblock: RegionBlock
    ):
        def render_subgraph(digraph, label):
            # render subgraph
            graph = regionblock.get_full_graph()
            with digraph.subgraph(name=f"cluster_{label}") as subg:
                subg.node(f"cluster_{label}", style="invis")
                color = "blue"
                if regionblock.kind == "branch":
                    color = "green"
                if regionblock.kind == "tail":
                    color = "purple"
                if regionblock.kind == "head":
                    color = "red"
                subg.attr(color=color, label=regionblock.kind, bgcolor="white")
                for label, block in graph.items():
                    self.render_block(subg, label, block)
            # render edges within this region
            self.render_edges(graph)

        if hasattr(regionblock, "render_rvsdg"):
                with regionblock.render_rvsdg(self, digraph, label) as digraph:
                  render_subgraph(digraph, label)
        else:
            render_subgraph(digraph, label)

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
        elif isinstance(block, ControlVariableBlock):
            with self._render_region_for_control_blocks(digraph, block, label) as subg:
                self.render_control_variable_block(subg, label, block)

        elif isinstance(block, BranchBlock):
            with self._render_region_for_control_blocks(digraph, block, label) as subg:
                self.render_branching_block(subg, label, block)
        elif isinstance(block, RegionBlock):
            self.render_region_block(digraph, label, block)
        else:
            block.render_rvsdg(self, digraph, label)

    @contextlib.contextmanager
    def _render_region_for_control_blocks(self, digraph, block, label):
        context: Callable

        if hasattr(block, "incoming_states"):
            @contextlib.contextmanager
            def context():
                with digraph.subgraph(name=f"cluster_controlregion_{id(self)}") as subg:
                    subg.attr(color='lightgrey', style='solid', label="")
                    yield subg
                    label_inc = f"incoming_{id(block)}"
                    label_out = f"outgoing_{id(block)}"
                    subg.node(label_inc, label=f"{'|'.join([f'<{k}> {k}' for k in block.incoming_states])}", shape='record')
                    subg.node(label_out, label=f"{'|'.join([f'<{k}> {k}' for k in block.outgoing_states])}", shape='record')
                    subg.edge(label_inc, str(label), style="invis", weight="1000")
                    subg.edge(str(label), label_out, style="invis", weight="1000")
        else:
            context = lambda: contextlib.nullcontext(digraph)
        with context() as subg:
            yield subg

    def render_edges(self, blocks: Dict[Label, BasicBlock]):
        for label, block in blocks.items():
            for dst in block.jump_targets:
                if dst in blocks:
                    if isinstance(block, RegionBlock):
                        if block.exiting is not None:

                            def get_inner_most_region(blk):
                                while isinstance(blk, RegionBlock):
                                    out = blk
                                    blk = blk.subregion.graph[blk.exiting]
                                return out

                            self.g.edge(str(get_inner_most_region(block).exiting), str(dst), color="blue")
                        else:
                            self.g.edge(str(label), str(dst), color="blue")
                    elif isinstance(block, BasicBlock):
                        self.g.edge(str(label), str(dst), color="blue")
                    else:
                        raise Exception("unreachable")
            for dst in block.backedges:
                # assert dst in blocks
                self.g.edge(
                    str(label), str(dst), style="dashed", color="grey", constraint="0"
                )
        # Render pending edges
        for (src, dst), attrs in self.edges.items():
            self.g.edge(src, dst, **attrs)
        self.edges.clear()

    # def render_byteflow(self, byteflow: ByteFlow):
    #     self.bcmap_from_bytecode(byteflow.bc)

    #     # render nodes
    #     for label, block in byteflow.scfg.graph.items():
    #         self.render_block(self.g, label, block)
    #     self.render_edges(byteflow.scfg.graph)
    #     return self.g

    def render_rvsdg(self, scfg):
        # render nodes
        for label, block in scfg.graph.items():
            self.render_block(self.g, label, block)
        self.render_edges(scfg.graph)
        # render inter-states
        self._render_inter_states(scfg)
        return self.g

    def _render_inter_states(self, scfg: SCFG):
        g = self.g
        for src in scfg.graph.values():
            if hasattr(src, "outgoing_states"):
                for label in src.jump_targets:
                    if label in scfg.graph:
                        dst = scfg.graph[label]
                        if hasattr(dst, "incoming_states"):
                            # Connect src outgoing to dst incoming
                            for name in src.outgoing_states:
                                g.edge(f"outgoing_{id(src)}:{name}",
                                       f"incoming_{id(dst)}:{name}")
                if isinstance(src, RegionBlock):
                    self._render_inter_states(src.subregion)
                    self._render_inter_states_in_region(src)

    def _render_inter_states_in_region(self, node: RegionBlock):
        g = self.g
        # Connect region incoming to the incoming of head
        head = node.subregion[node.subregion.find_head()]
        if hasattr(head, "incoming_states"):
            for name in head.incoming_states:
                g.edge(f"incoming_{id(node)}:{name}",
                        f"incoming_{id(head)}:{name}")
        # Connection outgoing of exit to region outgoing
        exit = node.subregion[node.exiting]
        if hasattr(exit, "outgoing_states"):
            for name in exit.outgoing_states:
                g.edge(f"outgoing_{id(exit)}:{name}",
                        f"outgoing_{id(node)}:{name}")


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
