# -*- coding: utf-8 -*-
"""
Hold metadata for instructions.
"""
from __future__ import print_function, division, absolute_import

import llvm.core

from numba import *

def _typename(type):
    typename = str(type)
    typename = typename.replace("float", "float32")
    typename = typename.replace("double", "float64")
    typename = typename.replace("long double", "float128")
    return typename

def typename(type):
    "Get the TBAA type name"
    if type.is_tbaa:
        return type.name
    else:
        return _typename(type)

def is_tbaa_type(type):
    return (type.is_pointer or type.is_tbaa or
            type.is_object or type.is_array)

class TBAAMetadata(object):
    """
    Type Based Alias Analysis metadata. This defines a type tree where
    types in different branches are known not to alias each other. Only
    the ancestors of a type node alias that node.
    """

    def __init__(self, module):
        self.module = module
        self.metadata_cache = {}
        self.initialize()
        self.unique_number = 0

    def initialize(self):
        self.root = self.make_metadata("root", root=None)
        self.char_pointer = self.make_metadata("char *", root=self.root)
        self.metadata_cache[char.pointer()] = self.char_pointer

    def make_metadata(self, typename, root, is_constant=False):
        operands = [self.get_string(typename)]
        if root is not None:
            assert isinstance(root, llvm.core.MetaData)
            operands.append(root)

            if is_constant:
                constant = llvm.core.Constant.int(llvm.core.Type.int(64), 1)
                operands.append(constant)

        node = llvm.core.MetaData.get(self.module, operands)
        llvm.core.MetaData.add_named_operand(self.module, "tbaa", node)

        # print "made metadata", self.module.id, typename # , node, root
        return node

    def make_unique_metadata(self, root, is_constant=False):
        result = self.make_metadata("unique%d" % self.unique_number, root,
                                    is_constant)
        self.unique_number += 1
        return result

    def get_string(self, string):
        return llvm.core.MetaDataString.get(self.module, string)

    def find_root(self, type):
        """
        Find the metadata root of a type. E.g. if we have tbaa(sometype).pointer(),
        we want to return tbaa(sometype).root.pointer()
        """
        # Find TBAA base type
        n = 0
        while type.is_pointer:
            type = type.base_type
            n += 1

        if type.is_tbaa:
            # Return TBAA base type pointer composition
            root_type = type.root
            for i in range(n):
                root_type = root_type.pointer()

            # Define root type metadata
            root = self.get_metadata(root_type)
        else:
            # Not a TBAA root type, alias anything
            root = self.char_pointer

        return root

    def get_metadata(self, type, typeroot=None):
        if not is_tbaa_type(type):
            return None

        if type in self.metadata_cache:
            return self.metadata_cache[type]

        # Build metadata
        if typeroot:
            root = self.metadata_cache[typeroot]
        elif type.is_tbaa:
            if type.root in self.metadata_cache:
                root = self.metadata_cache[type.root]
            else:
                root = self.get_metadata(type.root)
        else:
            root = self.find_root(type)

        node = self.make_metadata(typename(type), root,
                                  ) #is_constant="const" in type.qualifiers)

        # Cache result
        self.metadata_cache[type] = node

        return node

    def set_metadata(self, load_instr, metadata):
        load_instr.set_metadata("tbaa", metadata)

    def set_tbaa(self, load_instr, type):
        md = self.get_metadata(type)
        if md is not None:
            self.set_metadata(load_instr, md)
