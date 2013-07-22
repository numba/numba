# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *

class TempNode(ExprNode): #, ast.Name):
    """
    Create a temporary to store values in. Does not perform reference counting.
    """

    temp_counter = 0

    def __init__(self, type, name=None, dst_variable=None):
        self.type = type
        self.name = name
        self.variable = Variable(type, name='___numba_%d' % self.temp_counter,
                                 is_local=True)
        TempNode.temp_counter += 1
        self.llvm_temp = None

        self.dst_variable = dst_variable
        self._tbaa_node = None

    def get_tbaa_node(self, tbaa):
        """
        TBAA metadata node unique to this temporary. This is valid
        since one cannot take the address of a temporary.
        """
        if self._tbaa_node is None:
            root = tbaa.get_metadata(char.pointer())
            self._tbaa_node = tbaa.make_unique_metadata(root)

        return self._tbaa_node

    def load(self, invariant=False):
        return TempLoadNode(temp=self, invariant=invariant)

    def store(self):
        return TempStoreNode(temp=self)

    def __repr__(self):
        if self.name:
            name = ", %s" % self.name
        else:
            name = ""
        return "temp(%s%s)" % (self.type, name)

class TempLoadNode(ExprNode):
    _fields = ['temp']

    def __init__(self, temp, invariant=False):
        self.temp = temp
        self.type = temp.type
        self.variable = temp.variable
        self.invariant = invariant

    def __repr__(self):
        return "load(%s)" % self.temp

class TempStoreNode(TempLoadNode):
    _fields = ['temp']

    def __repr__(self):
        return "store(%s)" % self.temp
