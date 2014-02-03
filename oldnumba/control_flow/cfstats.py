# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import nodes
from numba.reporting import getpos

class StatementDescr(object):
    is_assignment = False

class LoopDescr(object):
    def __init__(self, next_block, loop_block):
        self.next_block = next_block
        self.loop_block = loop_block
        self.exceptions = []


class ExceptionDescr(object):
    """Exception handling helper.

    entry_point   ControlBlock Exception handling entry point
    finally_enter ControlBlock Normal finally clause entry point
    finally_exit  ControlBlock Normal finally clause exit point
    """

    def __init__(self, entry_point, finally_enter=None, finally_exit=None):
        self.entry_point = entry_point
        self.finally_enter = finally_enter
        self.finally_exit = finally_exit


class NameAssignment(object):

    is_assignment = True

    def __init__(self, lhs, rhs, entry, assignment_node, warn_unused=True):
        if not hasattr(lhs, 'cf_state'):
            lhs.cf_state = set()
        if not hasattr(lhs, 'cf_is_null'):
            lhs.cf_is_null = False

        self.lhs = lhs
        self.rhs = rhs
        self.assignment_node = assignment_node

        self.entry = entry
        self.pos = getpos(lhs)
        self.refs = set()
        self.is_arg = False
        self.is_deletion = False

        # NOTE: this is imperfect, since it means warnings are disabled for
        # *all* definitions in the function...
        self.entry.warn_unused = warn_unused

    def __repr__(self):
        return '%s(entry=%r)' % (self.__class__.__name__, self.entry)

    def infer_type(self, scope):
        return self.rhs.infer_type(scope)

    def type_dependencies(self, scope):
        return self.rhs.type_dependencies(scope)

class AttributeAssignment(object):
    """
    Assignment to some attribute. We need to detect assignments in the
    constructor of extension types.
    """

    def __init__(self, assmnt):
        self.assignment_node = assmnt
        self.lhs = assmnt.targets[0]
        self.rhs = assmnt.value

class Argument(NameAssignment):
    def __init__(self, lhs, rhs, entry):
        NameAssignment.__init__(self, lhs, rhs, entry)
        self.is_arg = True


class PhiNode(nodes.Node):

    def __init__(self, block, variable):
        self.block = block
        # Unrenamed variable. This will be replaced by the renamed version
        self.variable = variable
        self.type = None
        # self.incoming_blocks = []

        # Set of incoming variables
        self.incoming = set()
        self.phis = set()

        self.assignment_node = self

    @property
    def entry(self):
        return self.variable

    def add_incoming_block(self, block):
        self.incoming_blocks.append(block)

    def add(self, block, assmnt):
        if assmnt is not self:
            self.phis.add((block, assmnt))

    def __repr__(self):
        lhs = self.variable.name
        if self.variable.renamed_name:
            lhs = self.variable.unmangled_name
        incoming = ", ".join("var(%s, %s)" % (var_in.unmangled_name, var_in.type)
            for var_in in self.incoming)
        if self.variable.type:
            type = str(self.variable.type)
        else:
            type = ""
        return "%s %s = phi(%s)" % (type, lhs, incoming)

    def find_incoming(self):
        for parent_block in self.block.parents:
            name = self.variable.name
            incoming_var = parent_block.symtab.lookup_most_recent(name)
            yield parent_block, incoming_var


class NameDeletion(NameAssignment):
    def __init__(self, lhs, entry):
        NameAssignment.__init__(self, lhs, lhs, entry)
        self.is_deletion = True

class Uninitialized(object):
    pass

class NameReference(object):
    def __init__(self, node, entry):
        if not hasattr(node, 'cf_state'):
            node.cf_state = set()
        self.node = node
        self.entry = entry
        self.pos = getpos(node)

    def __repr__(self):
        return '%s(entry=%r)' % (self.__class__.__name__, self.entry)
