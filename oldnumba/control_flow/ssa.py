# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast
from itertools import chain

from numba import nodes
from .debug import logger

#------------------------------------------------------------------------
# Kill unused Phis
#------------------------------------------------------------------------

def kill_unused_phis(cfg):
    changed = True
    while changed:
        changed = _kill_unused_phis(cfg)

def kill_phi(block, phi):
    logger.debug("Killing phi: %s", phi)

    block.symtab.pop(phi.variable.renamed_name)

    for incoming_var in phi.incoming:
        # A single definition can reach a block multiple times,
        # remove all references
        refs = [ref for ref in incoming_var.cf_references
                        if ref.variable is not phi.variable]
        incoming_var.cf_references = refs

def kill_unused_phis(cfg):
    """
    Used before running type inference.

    Kill phis which are not referenced. We need to do this bottom-up,
    i.e. in reverse topological dominator-tree order, since in SSA
    a definition always lexically precedes a reference.

    This is important, since it kills any unnecessary promotions (e.g.
    ones to object, which LLVM wouldn't be able to optimize out).

    TODO: Kill phi cycles, or do reachability analysis before inserting phis.
    """
    changed = False

    for block in cfg.blocks[::-1]:
        phi_nodes = []

        for i, phi in enumerate(block.phi_nodes):
            if phi.variable.cf_references:
                # Used phi
                # logger.info("Used phi %s, %s" % (phi, phi.variable.cf_references))
                phi_nodes.append(phi)
            else:
                # Unused phi
                changed = True

                kill_phi(block, phi)

        block.phi_nodes = phi_nodes

    return changed

#------------------------------------------------------------------------
# Iterate over all phi nodes or variables
#------------------------------------------------------------------------

def iter_phis(flow):
    "Iterate over all phi nodes"
    return chain(*[block.phi_nodes for block in flow.blocks])

def iter_phi_vars(flow):
    "Iterate over all phi nodes"
    for phi_node in iter_phis(flow):
        yield phi_node.variable

#------------------------------------------------------------------------
# Specialization code for SSA
#------------------------------------------------------------------------

def specialize_ssa(funcdef):
    """
    Handle phi nodes:

        1) Handle incoming variables which are not initialized. Set
           incoming_variable.uninitialized_value to a constant 'bad'
           value (e.g. 0xbad for integers, NaN for floats, NULL for
           objects)

        2) Handle incoming variables which need promotions. An incoming
           variable needs a promotion if it has a different type than
           the the phi. The promotion happens in each ancestor block that
           defines the variable which reaches us.

           Promotions are set separately in the symbol table, since the
           ancestor may not be our immediate parent, we cannot introduce
           a rename and look up the latest version since there may be
           multiple different promotions. So during codegen, we first
           check whether incoming_type == phi_type, and otherwise we
           look up the promotion in the parent block or an ancestor.
    """
    for phi_node in iter_phis(funcdef.flow):
        specialize_phi(phi_node)

def specialize_phi(node):
    for parent_block, incoming_var in node.find_incoming():
        if incoming_var.type.is_uninitialized:
            incoming_type = incoming_var.type.base_type or node.type
            bad = nodes.badval(incoming_type)
            incoming_var.type.base_type = incoming_type
            incoming_var.uninitialized_value = bad
            # print incoming_var

        elif not incoming_var.type == node.type:
            # Create promotions for variables with phi nodes in successor
            # blocks.
            incoming_symtab = incoming_var.block.symtab
            if (incoming_var, node.type) not in node.block.promotions:
                # Make sure we only coerce once for each destination type and
                # each variable
                incoming_var.block.promotions.add((incoming_var, node.type))

                # Create promotion node
                name_node = nodes.Name(id=incoming_var.renamed_name,
                                       ctx=ast.Load())
                name_node.variable = incoming_var
                name_node.type = incoming_var.type
                coercion = name_node.coerce(node.type)
                promotion = nodes.PromotionNode(node=coercion)

                # Add promotion node to block body
                incoming_var.block.body.append(promotion)
                promotion.variable.block = incoming_var.block

                # Update symtab
                incoming_symtab.promotions[incoming_var.name,
                                           node.type] = promotion
            else:
                promotion = incoming_symtab.lookup_promotion(
                    incoming_var.name, node.type)

    return node

#------------------------------------------------------------------------
# Handle phis during code generation
#------------------------------------------------------------------------

def process_incoming(phi_node):
    """
    Add all incoming phis to the phi instruction.

    Handle promotions by using the promoted value from the incoming block.
    E.g.

        bb0: if C:
        bb1:     x = 2
             else:
        bb2:     x = 2.0

        bb3: x = phi(x_bb1, x_bb2)

    has a promotion for 'x' in basic block 1 (from int to float).
    """
    var = phi_node.variable
    phi = var.lvalue

    for parent_block, incoming_var in phi_node.find_incoming():
        if incoming_var.type.is_uninitialized:
            pass
        elif not incoming_var.type == phi_node.type:
            promotion = parent_block.symtab.lookup_promotion(var.name,
                                                             phi_node.type)
            incoming_var = promotion.variable

        assert incoming_var.lvalue, incoming_var
        assert parent_block.exit_block, parent_block

        phi.add_incoming(incoming_var.lvalue,
                         parent_block.exit_block)

def handle_phis(flow):
    """
    Update all our phi nodes after translation is done and all Variables
    have their llvm values set.
    """
    if flow is None:
        return

    for phi_node in iter_phis(flow):
        process_incoming(phi_node)
