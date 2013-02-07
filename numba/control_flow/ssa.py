from itertools import imap, chain

from numba import nodes
from debug import logger

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

        phi.add_incoming(incoming_var.lvalue,
                         parent_block.exit_block)

        if phi_node.type.is_array:
            nodes.update_preloaded_phi(phi_node.variable,
                                       incoming_var,
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
