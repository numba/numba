# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast

from numba import visitors

class DeleteStatement(visitors.NumbaVisitor):
    """
    Delete a (compound) statement that contains basic blocks.
    The statement must be at the start of the entry block.

    idom: the immediate dominator of
    """

    def __init__(self, flow):
        self.flow = flow

    def visit_If(self, node):
        self.generic_visit(node)

        # Visit ControlBlocks
        self.visit(node.cond_block)
        self.visit(node.if_block)
        if node.orelse:
            self.visit(node.else_block)
        if node.exit_block:
            self.visit(node.exit_block)

    visit_While = visit_If

    def visit_For(self, node):
        self.generic_visit(node)

        # Visit ControlBlocks
        self.visit(node.cond_block)
        self.visit(node.if_block)
        if node.orelse:
            self.visit(node.else_block)
        if node.exit_block:
            self.visit(node.exit_block)


    def visit_ControlBlock(self, node):
        #print "deleting block", node
        for phi in node.phi_nodes:
            for incoming in phi.incoming:
                #print "deleting", incoming, phi
                incoming.cf_references.remove(phi)

        self.generic_visit(node)
        node.delete(self.flow)

    def visit_Name(self, node):
        references = node.variable.cf_references
        if isinstance(node.ctx, ast.Load) and node in references:
            references.remove(node)
