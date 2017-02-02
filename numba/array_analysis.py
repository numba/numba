from __future__ import print_function, division, absolute_import
from numba import ir
#from numba.annotations import type_annotations
from numba import types

class ArrayAnalysis(object):
    '''Analysis of array computations such as shapes and equivalence classes
    '''

    def __init__(self, func_ir, type_annotation):
        '''Constructor for the Rewrite class.
        '''
        self.func_ir = func_ir
        self.type_annotation = type_annotation
        self.next_eq_class = 1
        # equivalence classes for each dimension of each array are saved
        # string to tuple of class numbers
        # example: {'A':[1,2]}
        #          {1:[n,a],2:[k,m,3]}
        self.array_shape_classes = {}
        self.class_sizes = {}
        #print("ARRAY ANALYSIS")

    def run(self):
        # TODO: ignoring CFG for now
        for (key, block) in self.func_ir.blocks.items():
            self._analyze_block(block)
        print(self.array_shape_classes)
        #print("RUN ARRAY ANALYSIS")

    def _analyze_block(self, block):
        for inst in block.body:
            self._analyze_inst(inst)

    def _analyze_inst(self, inst):
        if isinstance(inst, ir.Assign):
            self._analyze_assign(inst)

    def _analyze_assign(self, assign):
        if isinstance(assign.value, ir.Arg) and self._isarray(assign.value.name):
            self._add_array_corr(assign.value.name)

    def _isarray(self, varname):
        return isinstance(self.type_annotation.typemap[varname],
                          types.npytypes.Array)

    def _add_array_corr(self, varname):
        assert varname not in self.array_shape_classes
        self.array_shape_classes[varname] = []
        arr_typ = self.type_annotation.typemap[varname]
        for i in range(0,arr_typ.ndim):
            new_class = self._get_next_class()
            self.array_shape_classes[varname].append(new_class)

    def _get_next_class(self):
        m = self.next_eq_class
        self.next_eq_class += 1
        return m
