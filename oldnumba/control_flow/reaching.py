# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import traits
from numba.control_flow.cfstats import (
    NameReference, NameAssignment, Uninitialized)

def allow_null(node):
    return False


def check_definitions(env, flow, warner):
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
            if not isinstance(stat, (NameAssignment, NameReference)):
                continue

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

    # assignment hints
    for node in assmt_nodes:
        maybe_null = Uninitialized in node.cf_state
        node.cf_maybe_null = maybe_null
        node.cf_is_null = maybe_null and len(node.cf_state) == 1

    warner.check_uninitialized(references)
    warner.warn_unused_result(assignments)
    warner.warn_unused_entries(flow)

    if warner.have_errors:
        warner.messages.report(post_mortem=False)

    for node in assmt_nodes:
        node.cf_state = None #ControlFlowState(node.cf_state)
    for node in references:
        node.cf_state = None #ControlFlowState(node.cf_state)

@traits.traits
class CFWarner(object):
    "Generate control flow related warnings."

    have_errors = traits.Delegate('messages')

    def __init__(self, message_collection, directives):
        self.messages = message_collection
        self.directives = directives

    def check_uninitialized(self, references):
        "Find uninitialized references and cf-hints"
        warn_maybe_uninitialized = self.directives['warn.maybe_uninitialized']

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
                        self.messages.error(
                            node,
                            "local variable '%s' referenced before assignment"
                            % entry.name)
                    else:
                        self.messages.warning(
                            node,
                            "local variable '%s' referenced before assignment"
                            % entry.name)
                elif warn_maybe_uninitialized:
                    self.messages.warning(
                        node,
                        "local variable '%s' might be referenced before assignment"
                        % entry.name)
            else:
                node.cf_is_null = False
                node.cf_maybe_null = False

    def warn_unused_entries(self, flow):
        """
        Generate warnings for unused variables or arguments. This is issues when
        an argument or variable is unused entirely in the function.
        """
        warn_unused = self.directives['warn.unused']
        warn_unused_arg = self.directives['warn.unused_arg']

        for entry in flow.entries:
            if (not entry.cf_references and not entry.is_cellvar and
                    entry.renameable): # and not entry.is_pyclass_attr
                if entry.is_arg:
                    if warn_unused_arg:
                        self.messages.warning(
                            entry, "Unused argument '%s'" % entry.name)
                else:
                    if (warn_unused and entry.warn_unused and
                        not entry.name.startswith('_') and
                        flow.is_tracked(entry)):
                        if getattr(entry, 'lineno', 1) > 0:
                            self.messages.warning(
                                entry, "Unused variable '%s'" % entry.name)
                entry.cf_used = False

    def warn_unused_result(self, assignments):
        """
        Warn about unused variable definitions. This is issued for individual
        definitions, e.g.

            i = 0   # this definition generates a warning
            i = 1
            print i
        """
        warn_unused_result = self.directives['warn.unused_result']
        for assmt in assignments:
            if not assmt.refs:
                if assmt.entry.cf_references and warn_unused_result:
                    if assmt.is_arg:
                        self.messages.warning(
                            assmt, "Unused argument value '%s'" %
                                                assmt.entry.name)
                    else:
                        self.messages.warning(
                            assmt, "Unused result in '%s'" %
                                                assmt.entry.name)
                assmt.lhs.cf_used = False

    def warn_unreachable(self, node):
        "Generate a warning for unreachable code"
        if hasattr(node, 'lineno'):
            self.messages.warning(node, "Unreachable code")
