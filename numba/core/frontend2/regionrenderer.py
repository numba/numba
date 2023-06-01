from typing import Any
from dataclasses import dataclass, replace, field
from itertools import groupby
from contextlib import contextmanager
from collections import defaultdict

from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
)
from .regionpasses import RegionVisitor

from .bc2rvsdg import (
    build_rvsdg,
    SCFG,
    BasicBlock,
    RegionBlock,
    DDGBlock,
    DDGControlVariable,
    DDGBranch,
    DDGRegion,
    DDGProtocol,
    Op,
    ValueState,
    DEBUG_GRAPH,
)

@dataclass(frozen=True)
class GraphNode:
    kind: str
    parent_regions: tuple[str, ...] = ()
    ports: tuple[str, ...] = ()

    data: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    src_port: str | None = None
    dst_port: str | None = None
    headlabel: str | None = None
    taillabel: str | None = None
    kind: str | None = None

@dataclass(frozen=True)
class GraphGroup:
    subgroups: dict[str, "GraphGroup"]
    nodes: set[str]

    @classmethod
    def make(cls):
        return cls(subgroups=defaultdict(GraphGroup.make), nodes=set())


class GraphBacking:
    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._groups: GraphGroup = GraphGroup.make()
        self._edges: set[GraphEdge] = set()

    def add_node(self, name: str, node: GraphNode):
        assert name not in self._nodes
        self._nodes[name] = node

        group = self._groups
        for p in node.parent_regions:
            group = group.subgroups[p]
        group.nodes.add(name)

    def add_edge(self, src: str, dst: str, **kwargs):
        self._edges.add(GraphEdge(src, dst, **kwargs))

    def verify(self):
        for edge in self._edges:
            if edge.src not in self._nodes:
                raise ValueError(f"missing node {edge.src!r}")
            if edge.dst not in self._nodes:
                raise ValueError(f"missing node {edge.dst!r}")
            if edge.src_port is not None:
                node = self._nodes[edge.src]
                if edge.src_port not in node.ports:
                    raise ValueError(f"missing port {edge.src_port!r} in node {edge.src!r}")
            if edge.dst_port is not None:
                node = self._nodes[edge.dst]
                if edge.dst_port not in node.ports:
                    raise ValueError(f"missing port {edge.dst_port!r} in node {edge.dst!r}")

    def render(self, renderer):
        self._render_group(renderer, self._groups)
        for edge in self._edges:
            renderer.render_edge(edge)

    def _render_group(self, renderer, group: GraphGroup):
        for k, subgroup in group.subgroups.items():
            with renderer.render_cluster(k) as subrenderer:
                self._render_group(subrenderer, subgroup)

        for k in group.nodes:
            node = self._nodes[k]
            renderer.render_node(k, node)



@dataclass(frozen=True)
class GraphNodeMaker:
    parent_path: tuple[str, ...]

    def subregion(self, name: str):
        cls = type(self)
        return cls(parent_path=(*self.parent_path, name))

    def make_node(self, **kwargs) -> GraphNode:
        return GraphNode(**kwargs, parent_regions=self.parent_path)


@dataclass(frozen=True)
class GraphBuilder:
    graph: GraphBacking
    node_maker: GraphNodeMaker

    @classmethod
    def make(cls):
        return cls(GraphBacking(), GraphNodeMaker(()))


class RegionGraphvizRenderer: # needs a abstract base class
    def __init__(self, g=None):
        from graphviz import Digraph

        self.digraph = Digraph() if g is None else g

    def render_node(self, k: str, node: GraphNode):
        if node.kind == "valuestate":
            self.digraph.node(k, label=node.data["body"], shape='rect')
        elif node.kind == "op":
            self.digraph.node(k, label=node.data["body"], shape='box', style='rounded')
        elif node.kind == "effect":
            self.digraph.node(k, label=node.data["body"], shape='circle')
        elif node.kind == "ports":
            ports = [f"<{x}> {x}" for x in node.ports]
            label = f"{node.data['body']} | {'|'.join(ports)}"
            self.digraph.node(k, label=label, shape="record")
        elif node.kind == "cfg":
            self.digraph.node(k, label=node.data["body"], shape="plain", fontcolor="blue")
        else:
            self.digraph.node(k, label=f"{k}\n{node.kind}\n{node.data.get('body', '')}", shape='rect')

    def render_edge(self, edge: GraphEdge):
        attrs = {}
        if edge.headlabel is not None:
            attrs["headlabel"] = edge.headlabel
        if edge.taillabel is not None:
            attrs["taillabel"] = edge.taillabel

        if edge.kind is not None:
            if edge.kind == "effect":
                attrs["style"] = "dotted"
            elif edge.kind == "meta":
                attrs["style"] = "invis"
            elif edge.kind == "cfg":
                attrs["style"] = "solid"
                attrs["color"] = "blue"
            else:
                raise ValueError(edge.kind)

        src = str(edge.src)
        dst = str(edge.dst)
        if edge.src_port:
            src += f":{edge.src_port}"
        if edge.dst_port:
            dst += f":{edge.dst_port}"

        self.digraph.edge(src, dst, **attrs)

    @contextmanager
    def render_cluster(self, name: str):
        with self.digraph.subgraph(name=f"cluster_{name}") as subg:
            attrs = dict(color="black", bgcolor="white")
            if name.startswith("regionouter"):
                attrs["bgcolor"] ="grey"
            elif name.startswith("loop_"):
                attrs["color"] = "blue"
            elif name.startswith("switch_"):
                attrs["color"] ="green"

            subg.attr(**attrs)
            yield type(self)(subg)


class RegionRenderer(RegionVisitor):

    def visit_block(self, block: BasicBlock, builder: GraphBuilder):
        nodename = block.name
        node_maker = builder.node_maker.subregion(f"metaregion_{nodename}")

        if isinstance(block, DDGBlock):
            node = node_maker.make_node(kind="cfg", data=dict(body=nodename))
            builder.graph.add_node(nodename, node)
            block.render_graph(replace(builder, node_maker=node_maker))
        else:
            body = "(tbd)"
            if isinstance(block, DDGBranch):
                body = f"branch_value_table:\n{block.branch_value_table}"
            elif isinstance(block, DDGControlVariable):
                body = f"variable_assignment:\n{block.variable_assignment}"
            node = node_maker.make_node(
                kind="cfg",
                data=dict(body=body),
            )
            builder.graph.add_node(nodename, node)

            if isinstance(block, DDGProtocol):
                # Insert incoming and outgoing
                self._add_inout_ports(
                    block, block,
                    replace(builder, node_maker=node_maker),
                )

        for dstnode in block.jump_targets:
            builder.graph.add_edge(nodename, dstnode, kind="cfg")
        return builder

    def _add_inout_ports(self, before_block, after_block, builder):
        # Make outgoing node
        outgoing_nodename = f"outgoing_{before_block.name}"
        outgoing_node = builder.node_maker.make_node(
            kind="ports",
            ports=list(before_block.outgoing_states),
            data=dict(body="outgoing"),
        )
        builder.graph.add_node(outgoing_nodename, outgoing_node)

        # Make incoming node
        incoming_nodename = f"incoming_{after_block.name}"
        incoming_node = builder.node_maker.make_node(
            kind="ports",
            ports=list(after_block.incoming_states),
            data=dict(body="incoming"),
        )
        builder.graph.add_node(incoming_nodename, incoming_node)

        # Add meta edge for implicit flow
        builder.graph.add_edge(incoming_nodename, outgoing_nodename, kind="meta")

    def visit_linear(self, region: RegionBlock, builder: GraphBuilder):
        nodename = region.name
        node_maker = builder.node_maker.subregion(f"regionouter_{nodename}")

        if isinstance(region, DDGProtocol):
            # Insert incoming and outgoing
            self._add_inout_ports(
                region, region,
                replace(builder, node_maker=node_maker),
            )

        subbuilder = replace(
            builder,
            node_maker=node_maker.subregion(f"{region.kind}_{nodename}"),
        )
        node = node_maker.make_node(kind="cfg", data=dict(body=nodename))
        builder.graph.add_node(region.name, node)
        super().visit_linear(region, subbuilder)

        self._connect_internal(region, builder)
        # connect cfg edge
        builder.graph.add_edge(region.name, region.header, kind="cfg")
        return builder

    def _connect_internal(self, region, builder):
        header = region.subregion[region.header]
        if isinstance(region, DDGProtocol) and isinstance(header, DDGProtocol):
            for k in region.incoming_states:
                builder.graph.add_edge(
                    f"incoming_{region.name}",
                    f"incoming_{header.name}",
                    src_port=k,
                    dst_port=k,
                )

        exiting = region.subregion[region.exiting]
        if isinstance(region, DDGProtocol) and isinstance(exiting, DDGProtocol):
            assert isinstance(region, RegionBlock)
            if "loop_region_0" == region.name and "head_region_0" == exiting.name:
                breakpoint()
            for k in region.outgoing_states & exiting.outgoing_states:
                builder.graph.add_edge(f"outgoing_{exiting.name}", f"outgoing_{region.name}", src_port=k, dst_port=k)

    def visit_graph(self, scfg: SCFG, builder):
        """Overriding
        """
        toposorted = self._toposort_graph(scfg)
        label: str
        last_label: str|None = None
        for lvl in toposorted:
            for label in lvl:
                builder = self.visit(scfg[label], builder)
                # Connect outgoing to incoming
                if last_label is not None:
                    last_node = scfg[last_label]
                    node = scfg[label]
                    self._connect_inout_ports(last_node, node, builder)
                last_label = label
        return builder

    def _connect_inout_ports(self, last_node, node, builder):
        if isinstance(last_node, DDGProtocol) and isinstance(node, DDGProtocol):
            for k in last_node.outgoing_states:
                builder.graph.add_edge(f"outgoing_{last_node.name}", f"incoming_{node.name}", src_port=k, dst_port=k)

    def visit_loop(self, region: RegionBlock, builder: GraphBuilder):
        return self.visit_linear(region, builder)

    def visit_switch(self, region: RegionBlock, builder: GraphBuilder):
        nodename = region.name
        node_maker = builder.node_maker.subregion(f"regionouter_{nodename}")
        if isinstance(region, DDGProtocol):
            # Insert incoming and outgoing
            self._add_inout_ports(
                region, region,
                replace(builder, node_maker=node_maker),
            )
        subbuilder = replace(
            builder,
            node_maker=node_maker.subregion(f"{region.kind}_{nodename}"),
        )

        head = region.subregion[region.header]
        tail = region.subregion[region.exiting]
        self.visit_linear(head, subbuilder)
        for blk in region.subregion.graph.values():
            if blk.kind == "branch":
                self._connect_inout_ports(head, blk, subbuilder)
                self.visit_linear(blk, subbuilder)
                self._connect_inout_ports(blk, tail, subbuilder)
        self.visit_linear(tail, subbuilder)

        self._connect_internal(region, builder)
        return builder

    def render(self, scfg):
        builder = GraphBuilder.make()
        self.visit_graph(scfg, builder)
        builder.graph.verify()
        return self.to_graphviz(builder.graph)

    def to_graphviz(self, graph: GraphBacking):
        rgr = RegionGraphvizRenderer()
        graph.render(rgr)
        return rgr.digraph
