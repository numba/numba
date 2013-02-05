from debug import logger

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