from __future__ import print_function, division, absolute_import
from numba import ir
#from numba.annotations import type_annotations
from numba import types
from numba.typing import npydecl

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

        #if isinstance(assign.value, ir.Arg) and self._isarray(assign.value.name):
        #    self._add_array_corr(assign.value.name)
        # lhs is always var?
        assert isinstance(assign.target, ir.Var)
        lhs = assign.target.name
        if self._isarray(lhs):
            self.array_shape_classes[lhs] = self._get_rhs_class(assign.value)

    # lhs is array so rhs has to return array
    def _get_rhs_class(self, node):
        if isinstance(node, ir.Arg):
            assert self._isarray(node.name)
            return self._add_array_corr(node.name)
        if isinstance(node, ir.Expr):
            if node.op=='unary' and node.fn in UNARY_MAP_OP:
                assert isinstance(node.value, ir.Var)
                in_var = node.value.name
                assert self._isarray(in_var)
                return self.array_shape_classes[in_var]
            #elif node.op
        return [-1]

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
        return self.array_shape_classes[varname]

    def _get_next_class(self):
        m = self.next_eq_class
        self.next_eq_class += 1
        return m

UNARY_MAP_OP = npydecl.NumpyRulesUnaryArrayOperator._op_map.keys()
