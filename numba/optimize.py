# -*- coding: UTF-8 -*-

from itertools import imap

from numba import typesystem
from numba import visitors

properties = ("preload_data", "preload_shape", "preload_strides")

def update(phi_var):
    updated = False

    # Get incoming variables
    phi_node = phi_var.name_assignment
    incoming = [var for block, var in phi_node.find_incoming()]

    # Update incoming variables
    for property in properties:
        if getattr(phi_var, property):
            for v in incoming:
                updated |= not getattr(v, property)
                setattr(v, property, True)

    return updated

def propagate_properties(flow):
    """
    Propagate preloading properties through the def-use graph.

    If a phi node for an array variable needs preloading, all its
    incoming variables do too. This means a single property for a phi
    must propagate through an entire phi cycle:

        preload(v) = (initial_condition ∨ (∨ preload(p) for p ∈ incoming(v)))
    """
    changed = True
    while changed:
        changed = False
        for block in flow.blocks:
            for phi_node in block.phi_nodes:
                changed |= update(phi_node.variable)


class Preloader(visitors.NumbaTransformer):
    """
    Pre-load things in order to avoid a potential runtime load instruction.
    (We also use invariant loads and TBAA).
    """

    def visit_FunctionDef(self, node):
        # Set the initial preload conditions
        self.visitchildren(node)

        # Propagate initial conditions for variable merges to preceding
        # or post-ceding definitions (which are Name assignments or phis
        # themselves)
        propagate_properties(self.ast.flow)

        return node

    def visit_Subscript(self, node):
        if (node.value.type.is_array and not node.type.is_array and
                node.slice.type.is_int):
            array_variable = node.value.variable

            # Set the preload conditions
            array_variable.preload_data = True
            array_variable.preload_strides = True

        self.visitchildren(node)
        return node

    def visit_DataPointerNode(self, node):
        "This never runs, since we run before late specialization!"
        array_variable = node.node.variable

        # Set the preload conditions
        array_variable.preload_data = True
        array_variable.preload_strides = True

        self.visitchildren(node)
        return node
