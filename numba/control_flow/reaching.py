from numba.control_flow.cfstats import *
from numba.control_flow.reporting import *

def compute_uninitialized_phis(flow):
    """
    Compute potentially uninitialized phi variables.
    """
    maybe_uninitialized = set()

    for phi_var, phi_node in flow.blocks[0].phis.iteritems():
        if phi_var.uninitialized:
            maybe_uninitialized.add(phi_var)

    for block in flow.blocks[1:]:
        for phi_var, phi_node in block.phis:
            if any(x in maybe_uninitialized for x in phi_node.incoming):
                maybe_uninitialized.add(phi_var)

    return maybe_uninitialized

def allow_null(node):
    return False

def check_definitions(flow, compiler_directives):
    flow.initialize()
    flow.reaching_definitions()

    # Track down state
    assignments = set()
    # Node to entry map
    references = {}
    assmt_nodes = set()

    for block in flow.blocks:
        i_state = block.i_input
        for stat in block.stats:
            i_assmts = flow.assmts[stat.entry]
            state = flow.map_one(i_state, stat.entry)
            if isinstance(stat, NameAssignment):
                stat.lhs.cf_state.update(state)
                assmt_nodes.add(stat.lhs)
                i_state = i_state & ~i_assmts.mask
                if stat.is_deletion:
                    i_state |= i_assmts.bit
                else:
                    i_state |= stat.bit
                assignments.add(stat)
                # if stat.rhs is not fake_rhs_expr:
                stat.entry.cf_assignments.append(stat)
            elif isinstance(stat, NameReference):
                references[stat.node] = stat.entry
                stat.entry.cf_references.append(stat)
                stat.node.cf_state.update(state)
                if not allow_null(stat.node):
                    i_state &= ~i_assmts.bit
                state.discard(Uninitialized)
                for assmt in state:
                    assmt.refs.add(stat)

    # Check variable usage
    warn_maybe_uninitialized = compiler_directives['warn.maybe_uninitialized']
    warn_unused_result = compiler_directives['warn.unused_result']
    warn_unused = compiler_directives['warn.unused']
    warn_unused_arg = compiler_directives['warn.unused_arg']

    messages = MessageCollection()

    # assignment hints
    for node in assmt_nodes:
        maybe_null = Uninitialized in node.cf_state
        node.cf_maybe_null = maybe_null
        node.cf_is_null = maybe_null and len(node.cf_state) == 1

    # Find uninitialized references and cf-hints
    for node, entry in references.iteritems():
        if Uninitialized in node.cf_state:
            node.cf_maybe_null = True
            from_closure = False # entry.from_closure
            if not from_closure and len(node.cf_state) == 1:
                node.cf_is_null = True
            if allow_null(node) or from_closure: # or entry.is_pyclass_attr:
                pass # Can be uninitialized here
            elif node.cf_is_null:
                is_object = True #entry.type.is_pyobject
                is_unspecified = False #entry.type.is_unspecified
                error_on_uninitialized = False #entry.error_on_uninitialized
                if entry.renameable and (is_object or is_unspecified or
                                         error_on_uninitialized):
                    messages.error(
                        node,
                        "local variable '%s' referenced before assignment"
                        % entry.name)
                else:
                    messages.warning(
                        node,
                        "local variable '%s' referenced before assignment"
                        % entry.name)
            elif warn_maybe_uninitialized:
                messages.warning(
                    node,
                    "local variable '%s' might be referenced before assignment"
                    % entry.name)
        else:
            node.cf_is_null = False
            node.cf_maybe_null = False

    # Unused result
    for assmt in assignments:
        if not assmt.refs: # and not assmt.entry.is_pyclass_attr
        # and not assmt.entry.in_closure):
            if assmt.entry.cf_references and warn_unused_result:
                if assmt.is_arg:
                    messages.warning(assmt, "Unused argument value '%s'" %
                                            assmt.entry.name)
                else:
                    messages.warning(assmt, "Unused result in '%s'" %
                                            assmt.entry.name)
            assmt.lhs.cf_used = False

    # Unused entries
    for entry in flow.entries:
        if (not entry.cf_references and not entry.is_cellvar and
            entry.renameable): # and not entry.is_pyclass_attr
            if entry.is_arg:
                if warn_unused_arg:
                    messages.warning(entry, "Unused argument '%s'" %
                                            entry.name)
            else:
                if warn_unused and entry.warn_unused and flow.is_tracked(entry):
                    messages.warning(entry, "Unused variable '%s'" %
                                            entry.name)
            entry.cf_used = False

    messages.report()

    for node in assmt_nodes:
        node.cf_state = None #ControlFlowState(node.cf_state)
    for node in references:
        node.cf_state = None #ControlFlowState(node.cf_state)
