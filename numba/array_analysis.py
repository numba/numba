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
        # class zero especial and is size 1 for constants
        # and added broadcast dimensions
        # -1 class means unknown
        self.class_sizes = {0:[1]}
        # keep a list of numpy Global variables to find numpy calls
        self.numpy_globals = []
        # keep numpy call variables with their call names
        self.numpy_calls = {}
        #print("ARRAY ANALYSIS")

    def run(self):
        # TODO: ignoring CFG for now
        for (key, block) in self.func_ir.blocks.items():
            self._analyze_block(block)
        print(self.array_shape_classes)
        print("numpy globals ", self.numpy_globals)
        print("numpy calls ", self.numpy_calls)
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
        rhs = assign.value
        if isinstance(rhs, ir.Global) and rhs.value.__name__=='numpy':
            self.numpy_globals.append(lhs)
        if isinstance(rhs, ir.Expr) and rhs.op=='getattr' and rhs.value.name in self.numpy_globals:
            self.numpy_calls[lhs] = rhs.attr

        #rhs_class_out = self._analyze_rhs_classes(rhs)
        if self._isarray(lhs):
            self.array_shape_classes[lhs] = self._analyze_rhs_classes(rhs)

    # lhs is array so rhs has to return array
    def _analyze_rhs_classes(self, node):
        if isinstance(node, ir.Arg):
            assert self._isarray(node.name)
            return self._add_array_corr(node.name)
        elif isinstance(node, ir.Var):
            return self.array_shape_classes[node.name]
        elif isinstance(node, ir.Expr):
            if node.op=='unary' and node.fn in UNARY_MAP_OP:
                assert isinstance(node.value, ir.Var)
                in_var = node.value.name
                assert self._isarray(in_var)
                return self.array_shape_classes[in_var]
            elif node.op=='binop' and node.fn in BINARY_MAP_OP:
                arg1 = node.lhs.name
                arg2 = node.rhs.name
                return self._broadcast_and_match_shapes(arg1, arg2)
            elif node.op=='inplace_binop' and node.immutable_fn in BINARY_MAP_OP:
                arg1 = node.lhs.name
                arg2 = node.rhs.name
                return self._broadcast_and_match_shapes(arg1, arg2)
            elif node.op=='cast':
                return self.array_shape_classes[node.value.name]
            elif node.op=='call' and node.func.name in self.numpy_calls.keys():
                return self._analyze_np_call(node.func.name, node.args)
            elif node.op=='getattr' and self._isarray(node.value.name):
                # matrix transpose
                if node.attr=='T':
                    out_eqs = self.array_shape_classes[node.value.name].copy()
                    out_eqs.reverse()
                    return out_eqs
            else:
                print("can't find shape classes for expr",node," of op",node.op)
        print("can't find shape classes for node",node," of type ",type(node))
        return []

    def _analyze_np_call(self, call_var, args):
        call_name = self.numpy_calls[call_var]
        if call_name=='dot':
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
            # for multi-dimensional arrays, last dimension of arg1 and second
            # to last dimension of arg2 should be equal since used in dot product.
            # if arg2 is 1D, its only dimension is used for dot product and
            # should be equal to second to last of arg1.
            assert len(args)==2 or len(args)==3
            in1 = args[0].name
            in2 = args[1].name
            ndims1 = self._get_ndims(in1)
            ndims2 = self._get_ndims(in2)
            c1 = self.array_shape_classes[in1][ndims1-1]
            c2 = 0

            if ndims2==1:
                c2 = self.array_shape_classes[in2][0]
            else:
                c2 = self.array_shape_classes[in2][ndims2-2]

            c_inner = self._merge_classes(c1,c2)
            self.array_shape_classes[in1][ndims1-1] = c_inner

            if ndims2==1:
                self.array_shape_classes[in2][0] = c_inner
            else:
                self.array_shape_classes[in2][ndims2-2] = c_inner

            c_out = []
            for i in range(0,ndims1-1):
                c_out.append(self.array_shape_classes[in1][i])
            for i in range(0,ndims2-2):
                c_out.append(self.array_shape_classes[in2][i])
            if ndims2>1:
                c_out.append(self.array_shape_classes[in2][ndims2-1])
            print("dot class ",c_out)
            return c_out
        elif call_name in UFUNC_MAP_OP:
            arg1 = arg2 = 'NULL'
            if len(args)>0: arg1 = args[0].name
            if len(args)>1: arg2 = args[1].name
            return self._broadcast_and_match_shapes(arg1, arg2)

        print("unknown numpy call:",call_name)
        return [0]

    def _merge_classes(self, c1, c2):
        return self._get_next_class()

    def _broadcast_and_match_shapes(self, arg1, arg2):
        """Infer shape equivalence of arguments based on Numpy broadcast rules
        and return shape of output
        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        """
        # one input has to be an array
        assert (arg1!='NULL' and self._isarray(arg1)) or (arg2!='NULL' and self._isarray(arg2))

        if arg1!='NULL' and self._isarray(arg1):
            eq1 = self.array_shape_classes[arg1]
        else:
            eq1 = [0]
        if arg2!='NULL' and self._isarray(arg2):
            eq2 = self.array_shape_classes[arg2]
        else:
            eq2 = [0]

        while len(eq1)<len(eq2):
            eq1.insert(0,0)
        while len(eq2)<len(eq1):
            eq2.insert(0,0)
        ndim = len(eq1)

        out_eq = [-1 for i in range(0,ndim)]
        for i in range(0,ndim):
            if eq1[i]==0:
                out_eq[i] = eq2[i]
            elif eq2[i]==0:
                out_eq[i] = eq1[i]
        return out_eq


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

    def _get_ndims(self, arr):
        return len(self.array_shape_classes[arr])

UNARY_MAP_OP = npydecl.NumpyRulesUnaryArrayOperator._op_map.keys()
BINARY_MAP_OP = npydecl.NumpyRulesArrayOperator._op_map.keys()
UFUNC_MAP_OP = [f.__name__ for f in npydecl.supported_ufuncs]
