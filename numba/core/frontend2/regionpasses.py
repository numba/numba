from collections.abc import Mapping
from numba_rvsdg.core.datastructures.scfg import (
    SCFG,
)
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
)


def _compute_incoming_labels(graph: Mapping[str, BasicBlock]) -> dict[str, set[str]]:
    jump_table: dict[str, set[str]] = {}
    blk: BasicBlock
    for k in graph:
        jump_table[k] = set()
    for blk in graph.values():
        for dst in blk.jump_targets:
            if dst in jump_table:
                jump_table[dst].add(blk.name)
    return jump_table


def toposort_graph(graph: Mapping[str, BasicBlock]) -> list[list[str]]:
    incoming_labels = _compute_incoming_labels(graph)
    visited: set[str] = set()
    toposorted: list[list[str]] = []
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
    return toposorted


class RegionVisitor:
    direction = "forward"

    def visit_block(self, block: BasicBlock, data):
        pass

    def visit_loop(self, region: RegionBlock, data):
        pass

    def visit_switch(self, region: RegionBlock, data):
        pass

    def visit_linear(self, region: RegionBlock, data):
        return self.visit_graph(region.subregion, data)

    def visit_graph(self, scfg: SCFG, data):
        toposorted = self._toposort_graph(scfg)
        label: str
        for lvl in toposorted:
            for label in lvl:
                data = self.visit(scfg[label], data)
        return data

    def _toposort_graph(self, scfg: SCFG):
        toposorted = toposort_graph(scfg.graph)
        if self.direction == "forward":
            return toposorted
        elif self.direction == "backward":
            return reversed(toposorted)
        else:
            assert False, f"invalid direction {self.direction!r}"

    def visit(self, block: BasicBlock, data):
        if isinstance(block, RegionBlock):
            if block.kind == "loop":
                fn = self.visit_loop
            elif block.kind == "switch":
                fn = self.visit_switch
            else:
                raise NotImplementedError('unreachable')
            data = fn(block, data)
        else:
            data = self.visit_block(block, data)
        return data


class RegionTransformer(RegionVisitor):

    def visit_block(self, parent: SCFG, block: BasicBlock, data):
        pass

    def visit_loop(self, parent: SCFG, region: RegionBlock, data):
        pass

    def visit_switch(self, parent: SCFG, region: RegionBlock, data):
        pass

    def visit_linear(self, parent: SCFG, region: RegionBlock, data):
        return self.visit_graph(region.subregion, data)

    def visit_graph(self, scfg: SCFG, data):
        toposorted = toposort_graph(scfg.graph)
        label: str
        for lvl in toposorted:
            for label in lvl:
                data = self.visit(scfg, scfg[label], data)
        return data

    def visit(self, parent: SCFG, block: BasicBlock, data):
        if isinstance(block, RegionBlock):
            if block.kind == "loop":
                fn = self.visit_loop
            elif block.kind == "switch":
                fn = self.visit_switch
            elif block.kind in {"head", "tail", "branch"}:
                fn = self.visit_linear
            else:
                raise NotImplementedError('unreachable', block.name, block.kind)
            data = fn(parent, block, data)
        else:
            data = self.visit_block(parent, block, data)
        return data
