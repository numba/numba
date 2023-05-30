from dataclasses import dataclass, replace
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
    Op,
    ValueState,
    DEBUG_GRAPH,
)

@dataclass(frozen=True)
class GraphNode:
    kind: str
    parent_regions: tuple[str, ...] = ()

@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str

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

    def add_edge(self, src: str, dst: str):
        self._edges.add(GraphEdge(src, dst))

    def verify(self):
        for edge in self._edges:
            if edge.src not in self._nodes:
                raise ValueError(f"missing node {edge.src!r}")
            if edge.dst not in self._nodes:
                raise ValueError(f"missing node {edge.dst!r}")

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

    def make_node(self, cls, *args, **kwargs) -> GraphNode:
        return cls(*args, **kwargs, parent_regions=self.parent_path)


@dataclass(frozen=True)
class GraphBuilder:
    graph: GraphBacking
    node_maker: GraphNodeMaker

    @classmethod
    def make(cls):
        return cls(GraphBacking(), GraphNodeMaker(()))


@dataclass(frozen=True)
class RVSDGGraphNode(GraphNode):
    pass


class RegionGraphvizRenderer: # needs a abstract base class
    def __init__(self, g=None):
        from graphviz import Digraph

        self.digraph = Digraph() if g is None else g

    def render_node(self, k: str, node: GraphNode):
        self.digraph.node(k, label=f"{k}\n{node.kind}", shape='rect')

    def render_edge(self, edge: GraphEdge):
        self.digraph.edge(edge.src, edge.dst)

    @contextmanager
    def render_cluster(self, name: str):
        with self.digraph.subgraph(name=f"cluster_{name}") as subg:
            yield type(self)(subg)


class RegionRenderer(RegionVisitor):

    def visit_block(self, block: BasicBlock, builder: GraphBuilder):
        nodename = self._id(block.name)
        node = builder.node_maker.make_node(RVSDGGraphNode, kind=f"{type(block).__name__}")
        builder.graph.add_node(nodename, node)
        for dstnode in block.jump_targets:
            builder.graph.add_edge(nodename, self._id(dstnode))
        return builder

    def visit_linear(self, region: RegionBlock, builder: GraphBuilder):
        subbuilder = replace(
            builder,
            node_maker=builder.node_maker.subregion(self._id(region.name)),
        )
        node = subbuilder.node_maker.make_node(RVSDGGraphNode, kind=f"{type(region).__name__}")
        subbuilder.graph.add_node(region.name, node)
        super().visit_linear(region, subbuilder)
        return builder

    def visit_loop(self, region: RegionBlock, builder: GraphBuilder):
        return self.visit_linear(region, builder)

    def visit_switch(self, region: RegionBlock, builder: GraphBuilder):
        self.visit_linear(region.subregion[region.header], builder)
        for blk in region.subregion.graph.values():
            if blk.kind == "branch":
                self.visit_linear(blk, builder)
        self.visit_linear(region.subregion[region.exiting], builder)
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


    def _id(self, nodename: str) -> str:
        return str(nodename)