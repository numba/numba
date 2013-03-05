# -*- coding: utf-8 -*-
# -*- coding: UTF-8 -*-

"""
Optimizations module.
"""
from __future__ import print_function, division, absolute_import

import ast
from itertools import imap

from numba import typesystem
from numba import visitors

#----------------------------------------------------------------------------
# NumPy Array Attribute Preloading
#----------------------------------------------------------------------------


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

    For each definition of an array variable, which is either a name assignment
    or a phi, we determine whether to pre-load the data pointer, the strides,
    and the shape information:

        array = np.array(...)
        for i in range(...):
            array[i]            # use array->data and array->strides[0]

    becomes

        array = np.array(...)

        temp_data = array->data
        temp_shape0 = array->shape[0]
        temp_stride0 = array->strides[0]

        for i in range(...):
            array[i]            # use pre-loaded temporaries
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
        is_full_index = node.value.type.is_array and not node.type.is_array
        # is_index = node.slice.type.is_int
        is_name = isinstance(node.value, ast.Name)
        maybe_null = ((is_name and node.value.cf_maybe_null) or
                      node.value.variable.uninitialized)

        # if maybe_null: print "maybe null"

        if is_full_index and is_name and not maybe_null:
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
