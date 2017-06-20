#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import
import types as pytypes  # avoid confusion with numba.types
import numpy
from numba import ir, analysis, types, config
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir)
from numba.typing import npydecl
import collections
import copy

UNKNOWN_CLASS = -1
CONST_CLASS = 0
MAP_TYPES = [numpy.ufunc]


class ArrayAnalysis(object):
    """Analyzes Numpy array computations for properties such as shapes
    and equivalence classes.
    """

    def __init__(self, func_ir, typemap, calltypes):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.next_eq_class = 1
        # array_shape_classes saves dimension size information for all arrays
        # each array dimension is mapped to a size equivalence class
        # for example, {'A':[1,2], 'B':[3,2]} means second dimensions of
        #   A and B have the same size.
        # format: variable name to list of class numbers
        self.array_shape_classes = collections.OrderedDict()
        # class_sizes saves constants or variables that represent the size of
        #   an equivalence class
        # For example, {1:[n,a],2:[3]} means variabes n and a contain size for
        #   class 1, while size for class 2 is constant 3.
        # class CONST_CLASS is special and represents size 1 for constants
        #    and added broadcast dimensions
        # UNKNOWN_CLASS class means size might not be constant
        self.class_sizes = {CONST_CLASS: [1], UNKNOWN_CLASS: []}
        # size variable to use for each array dimension
        self.array_size_vars = {}
        # keep a list of numpy Global variables to find numpy calls
        self.numpy_globals = []
        # calls that are essentially maps like DUFunc
        self.map_calls = []
        # keep numpy call variables with their call names
        self.numpy_calls = {}
        # keep attr calls to arrays like t=A.sum() as {t:('sum',A)}
        self.array_attr_calls = {}
        # keep tuple builds like {'t':[a,b],}
        self.tuple_table = {}
        self.list_table = {}
        self.constant_table = {}

    def run(self):
        """run array shape analysis on the IR and save information in
        array_shape_classes, class_sizes, and array_size_vars (see __init__
        comments). May generate some array shape calls if necessary.
        """
        dprint_func_ir(self.func_ir, "starting array analysis")
        if config.DEBUG_ARRAY_OPT == 1:
            print("variable types: ", sorted(self.typemap.items()))
            print("call types: ", self.calltypes)
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            self._analyze_block(self.func_ir.blocks[label])

        self._merge_equivalent_classes()
        self._cleanup_analysis_data()

        if config.DEBUG_ARRAY_OPT == 1:
            self.dump()

    def dump(self):
        """dump save shape information and internals for debugging purposes.
        """
        print("array_shape_classes: ", self.array_shape_classes)
        print("class_sizes: ", self.class_sizes)
        print("array_size_vars: ", sorted(self.array_size_vars.items()))
        print("numpy globals ", self.numpy_globals)
        print("numpy calls ", sorted(self.numpy_calls.items()))
        print("array attr calls ", self.array_attr_calls)
        print("tuple table ", self.tuple_table)

    def _analyze_block(self, block):
        out_body = []
        for inst in block.body:
            # instructions can generate extra size calls to be appended.
            # if an array doesn't have a size variable for a dimension,
            # a size variable should be generated when the array is created
            generated_size_calls = self._analyze_inst(inst)
            out_body.append(inst)
            for node in generated_size_calls:
                out_body.append(node)
        block.body = out_body

    def _analyze_inst(self, inst):
        if isinstance(inst, ir.Assign):
            return self._analyze_assign(inst)
        return []

    def _analyze_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value
        if isinstance(rhs, ir.Global):
            for T in MAP_TYPES:
                if isinstance(rhs.value, T):
                    self.map_calls.append(lhs)
            if isinstance(
                    rhs.value,
                    pytypes.ModuleType) and rhs.value == numpy:
                self.numpy_globals.append(lhs)
        if isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
            if rhs.value.name in self.numpy_globals:
                self.numpy_calls[lhs] = rhs.attr
            elif rhs.value.name in self.numpy_calls:
                # numpy submodule call like np.random.ranf
                # we keep random.ranf as call name
                self.numpy_calls[lhs] = (self.numpy_calls[rhs.value.name]
                                         + '.' + rhs.attr)
            elif self._isarray(rhs.value.name):
                self.array_attr_calls[lhs] = (rhs.attr, rhs.value)
        if isinstance(rhs, ir.Expr) and rhs.op == 'build_tuple':
            self.tuple_table[lhs] = rhs.items
        if isinstance(rhs, ir.Expr) and rhs.op == 'build_list':
            self.list_table[lhs] = rhs.items
        if isinstance(rhs, ir.Const) and isinstance(rhs.value, tuple):
            self.tuple_table[lhs] = rhs.value
        if isinstance(rhs, ir.Const):  # and np.isscalar(rhs.value):
            self.constant_table[lhs] = rhs.value

        #rhs_class_out = self._analyze_rhs_classes(rhs)
        size_calls = []
        if self._isarray(lhs):
            analyze_out = self._analyze_rhs_classes(rhs)
            if analyze_out is None:
                rhs_corr = self._add_array_corr(lhs)
            else:
                rhs_corr = copy.copy(analyze_out)
            if lhs in self.array_shape_classes:
                # if shape already inferred in another basic block,
                # make sure this new inference is compatible
                if self.array_shape_classes[lhs] != rhs_corr:
                    self.array_shape_classes[lhs] = [-1] * self._get_ndims(lhs)
                    self.array_size_vars.pop(lhs, None)
                    if config.DEBUG_ARRAY_OPT == 1:
                        print("incompatible array shapes in control flow")
                    return []
            self.array_shape_classes[lhs] = rhs_corr
            self.array_size_vars[lhs] = [-1] * self._get_ndims(lhs)
            # make sure output lhs array has size variables for each dimension
            for (i, corr) in enumerate(rhs_corr):
                # if corr unknown or new
                if corr == -1 or corr not in self.class_sizes.keys():
                    # generate size call nodes for this dimension
                    nodes = self._gen_size_call(assign.target, i)
                    size_calls += nodes
                    assert isinstance(nodes[-1], ir.Assign)
                    size_var = nodes[-1].target
                    if corr != -1:
                        self.class_sizes[corr] = [size_var]
                    self.array_size_vars[lhs][i] = size_var
                else:
                    # reuse a size variable from this correlation
                    # TODO: consider CFG?
                    self.array_size_vars[lhs][i] = self.class_sizes[corr][0]

        # print(self.array_shape_classes)
        return size_calls

    def _gen_size_call(self, var, i):
        out = []
        ndims = self._get_ndims(var.name)
        # attr call: A_sh_attr = getattr(A, shape)
        shape_attr_call = ir.Expr.getattr(var, "shape", var.loc)
        attr_var = ir.Var(
            var.scope,
            mk_unique_var(
                var.name +
                "_sh_attr" +
                str(i)),
            var.loc)
        self.typemap[attr_var.name] = types.containers.UniTuple(
            types.intp, ndims)
        attr_assign = ir.Assign(shape_attr_call, attr_var, var.loc)
        out.append(attr_assign)
        # const var for dim: $constA0 = Const(0)
        const_node = ir.Const(i, var.loc)
        const_var = ir.Var(
            var.scope,
            mk_unique_var(
                "$const" +
                var.name +
                str(i)),
            var.loc)
        self.typemap[const_var.name] = types.intp
        const_assign = ir.Assign(const_node, const_var, var.loc)
        out.append(const_assign)
        # get size: Asize0 = A_sh_attr[0]
        size_var = ir.Var(
            var.scope,
            mk_unique_var(
                var.name +
                "size" +
                str(i)),
            var.loc)
        self.typemap[size_var.name] = types.intp
        getitem_node = ir.Expr.static_getitem(attr_var, i, const_var, var.loc)
        self.calltypes[getitem_node] = None
        getitem_assign = ir.Assign(getitem_node, size_var, var.loc)
        out.append(getitem_assign)
        return out

    # lhs is array so rhs has to return array
    def _analyze_rhs_classes(self, node):
        if isinstance(node, ir.Arg):
            return None
            # can't assume node.name is valid variable
            #assert self._isarray(node.name)
            # return self._add_array_corr(node.name)
        elif isinstance(node, ir.Var):
            return copy.copy(self.array_shape_classes[node.name])
        elif isinstance(node, (ir.Global, ir.FreeVar)):
            # XXX: currently, global variables are frozen in Numba (can change)
            if isinstance(node.value, numpy.ndarray):
                shape = node.value.shape
                out_eqs = []
                for c in shape:
                    new_class = self._get_next_class_with_size(c)
                    out_eqs.append(new_class)
                return out_eqs
        elif isinstance(node, ir.Expr):
            if node.op == 'unary' and node.fn in UNARY_MAP_OP:
                assert isinstance(node.value, ir.Var)
                in_var = node.value.name
                assert self._isarray(in_var)
                return copy.copy(self.array_shape_classes[in_var])
            elif node.op == 'binop' and node.fn in BINARY_MAP_OP:
                arg1 = node.lhs.name
                arg2 = node.rhs.name
                return self._broadcast_and_match_shapes([arg1, arg2])
            elif node.op == 'inplace_binop' and node.immutable_fn in BINARY_MAP_OP:
                arg1 = node.lhs.name
                arg2 = node.rhs.name
                return self._broadcast_and_match_shapes([arg1, arg2])
            elif node.op == 'arrayexpr':
                # set to remove duplicates
                args = {v.name for v in node.list_vars()}
                return self._broadcast_and_match_shapes(list(args))
            elif node.op == 'cast':
                return copy.copy(self.array_shape_classes[node.value.name])
            elif node.op == 'call':
                call_name = 'NULL'
                args = copy.copy(node.args)
                if node.func.name in self.map_calls:
                    return copy.copy(self.array_shape_classes[args[0].name])
                if node.func.name in self.numpy_calls.keys():
                    call_name = self.numpy_calls[node.func.name]
                elif node.func.name in self.array_attr_calls.keys():
                    call_name, arr = self.array_attr_calls[node.func.name]
                    args.insert(0, arr)
                if call_name is not 'NULL':
                    return self._analyze_np_call(
                        call_name, args, dict(node.kws))
                else:
                    if config.DEBUG_ARRAY_OPT == 1:
                        # no need to raise since this is not a failure and
                        # analysis can continue (might limit optimization
                        # later)
                        print("can't find shape for unknown call:", node)
                    return None
            elif node.op == 'getattr' and self._isarray(node.value.name):
                # numpy recarray, e.g. X.a
                val = node.value.name
                val_typ = self.typemap[val]
                if (isinstance(val_typ.dtype, types.npytypes.Record)
                        and node.attr in val_typ.dtype.fields):
                    return copy.copy(self.array_shape_classes[val])
                # matrix transpose
                if node.attr == 'T':
                    return self._analyze_np_call('transpose', [node.value],
                                                 dict())
            elif (node.op == 'getattr' and isinstance(
                    self.typemap[node.value.name], types.npytypes.Record)):
                # nested arrays in numpy records
                val = node.value.name
                val_typ = self.typemap[val]
                if (node.attr in val_typ.fields
                        and isinstance(val_typ.fields[node.attr][0],
                                       types.npytypes.NestedArray)):
                    shape = val_typ.fields[node.attr][0].shape
                    return self._get_classes_from_const_shape(shape)
            elif node.op == 'getitem' or node.op == 'static_getitem':
                # getitem where output is array is possibly accessing elements
                # of numpy records, e.g. X['a']
                val = node.value.name
                val_typ = self.typemap[val]
                if (self._isarray(val) and isinstance(val_typ.dtype,
                                                      types.npytypes.Record)
                        and node.index in val_typ.dtype.fields):
                    return copy.copy(self.array_shape_classes[val])
            else:
                if config.DEBUG_ARRAY_OPT == 1:
                    # no need to raise since this is not a failure and
                    # analysis can continue (might limit optimization later)
                    print(
                        "can't find shape classes for expr",
                        node,
                        " of op",
                        node.op)
        if config.DEBUG_ARRAY_OPT == 1:
            # no need to raise since this is not a failure and
            # analysis can continue (might limit optimization later)
            print(
                "can't find shape classes for node",
                node,
                " of type ",
                type(node))
        return None

    def _analyze_np_call(self, call_name, args, kws):
        #print("numpy call ",call_name,args)
        if call_name == 'transpose':
            out_eqs = copy.copy(self.array_shape_classes[args[0].name])
            out_eqs.reverse()
            return out_eqs
        elif call_name in ['empty', 'zeros', 'ones', 'full', 'random.ranf',
                           'random.random_sample', 'random.sample']:
            shape_arg = None
            if len(args) > 0:
                shape_arg = args[0]
            elif 'shape' in kws:
                shape_arg = kws['shape']
            else:
                return None
            return self._get_classes_from_shape(shape_arg)
        elif call_name in ['random.rand', 'random.randn']:
            # arguments are integers, not a tuple
            return self._get_classes_from_dim_args(args)
        elif call_name == 'eye':
            # if one input n, output is n*n
            # two inputs n,m, output is n*m
            # N is either positional or kw arg
            if 'N' in kws:
                assert len(args) == 0
                args.append(kws['N'])
            if 'M' in kws:
                assert len(args) == 1
                args.append(kws['M'])

            new_class1 = self._get_next_class_with_size(args[0].name)
            out_eqs = [new_class1]
            if len(args) > 1:
                new_class2 = self._get_next_class_with_size(args[1].name)
                out_eqs.append(new_class2)
            else:
                out_eqs.append(new_class1)
            return out_eqs
        elif call_name == 'identity':
            # input n, output is n*n
            new_class1 = self._get_next_class_with_size(args[0].name)
            return [new_class1, new_class1]
        elif call_name == 'diag':
            k = self._get_second_arg_or_kw(args, kws, 'k')
            # TODO: support k other than 0 (other diagonal smaller size than
            # main)
            if k == 0:
                in_arr = args[0].name
                in_class = self.array_shape_classes[in_arr][0]
                # if 1D input v, create 2D output with v on diagonal
                # if 2D input v, return v's diagonal
                if self._get_ndims(in_arr) == 1:
                    return [in_class, in_class]
                else:
                    self._get_ndims(in_arr) == 2
                    return [in_class]
        elif call_name in ['empty_like', 'zeros_like', 'ones_like', 'full_like',
                           'copy', 'asfortranarray']:
            # shape same as input
            if args[0].name in self.array_shape_classes:
                out_corrs = copy.copy(self.array_shape_classes[args[0].name])
            else:
                # array scalars: constant input results in 0-dim array
                assert not self._isarray(args[0].name)
                # TODO: make sure arg is scalar
                out_corrs = []
            # asfortranarray converts 0-d to 1-d automatically
            if out_corrs == [] and call_name == 'asfortranarray':
                out_corrs = [CONST_CLASS]
            return out_corrs
        elif call_name == 'reshape':
            #print("reshape args: ", args)
            # TODO: infer shape from length of args[0] in case of -1 input
            if len(args) == 2:
                # shape is either Int or tuple of Int
                return self._get_classes_from_shape(args[1])
            else:
                # a list integers for shape
                return self._get_classes_from_shape_list(args[1:])
        elif call_name == 'array':
            # only 1D list is supported, and not ndmin arg
            if args[0].name in self.list_table:
                l = self.list_table[args[0].name]
                new_class1 = self._get_next_class_with_size(len(l))
                return [new_class1]
        elif call_name == 'concatenate':
            # all dimensions of output are same as inputs, except axis
            axis = self._get_second_arg_or_kw(args, kws, 'axis')
            if axis == -1:  # don't know shape if axis is not constant
                return None
            arr_args = self._get_sequence_arrs(args[0].name)
            if len(arr_args) == 0:
                return None
            ndims = self._get_ndims(arr_args[0].name)
            if ndims <= axis:
                return None
            out_eqs = [-1] * ndims
            new_class1 = self._get_next_class()
            # TODO: set size to sum of input array's size along axis
            out_eqs[axis] = new_class1
            for i in range(ndims):
                if i == axis:
                    continue
                c = self.array_shape_classes[arr_args[0].name][i]
                for v in arr_args:
                    # all input arrays have equal dimensions, except on axis
                    c = self._merge_classes(
                        c, self.array_shape_classes[v.name][i])
                out_eqs[i] = c
            return out_eqs
        elif call_name == 'stack':
            # all dimensions of output are same as inputs, but extra on axis
            axis = self._get_second_arg_or_kw(args, kws, 'axis')
            if axis == -1:  # don't know shape if axis is not constant
                return None
            arr_args = self._get_sequence_arrs(args[0].name)
            if len(arr_args) == 0:
                return None
            ndims = self._get_ndims(arr_args[0].name)
            out_eqs = [-1] * ndims
            # all input arrays have equal dimensions
            for i in range(ndims):
                c = self.array_shape_classes[arr_args[0].name][i]
                for v in arr_args:
                    c = self._merge_classes(
                        c, self.array_shape_classes[v.name][i])
                out_eqs[i] = c
            # output has one extra dimension
            new_class1 = self._get_next_class_with_size(len(arr_args))
            out_eqs.insert(axis, new_class1)
            # TODO: set size to sum of input array's size along axis
            return out_eqs
        elif call_name == 'hstack':
            # hstack is same as concatenate with axis=1 for ndim>=2
            dummy_one_var = ir.Var(args[0].scope, "__dummy_1", args[0].loc)
            self.constant_table["__dummy_1"] = 1
            args.append(dummy_one_var)
            return self._analyze_np_call('concatenate', args, kws)
        elif call_name == 'dstack':
            # dstack is same as concatenate with axis=2, atleast_3d args
            args[0] = self.convert_seq_to_atleast_3d(args[0])
            dummy_two_var = ir.Var(args[0].scope, "__dummy_2", args[0].loc)
            self.constant_table["__dummy_2"] = 2
            args.append(dummy_two_var)
            return self._analyze_np_call('concatenate', args, kws)
        elif call_name == 'vstack':
            # vstack is same as concatenate with axis=0 if 2D input dims or more
            # TODO: set size to sum of input array's size for 1D
            arr_args = self._get_sequence_arrs(args[0].name)
            if len(arr_args) == 0:
                return None
            ndims = self._get_ndims(arr_args[0].name)
            if ndims >= 2:
                dummy_zero_var = ir.Var(
                    args[0].scope, "__dummy_0", args[0].loc)
                self.constant_table["__dummy_0"] = 0
                args.append(dummy_zero_var)
                return self._analyze_np_call('concatenate', args, kws)
        elif call_name == 'column_stack':
            # 1D arrays turn into columns of 2D array
            arr_args = self._get_sequence_arrs(args[0].name)
            if len(arr_args) == 0:
                return None
            c = self.array_shape_classes[arr_args[0].name][0]
            for v in arr_args:
                c = self._merge_classes(c, self.array_shape_classes[v.name][0])
            new_class = self._get_next_class_with_size(len(arr_args))
            return [c, new_class]
        elif call_name in ['cumsum', 'cumprod']:
            in_arr = args[0].name
            in_ndims = self._get_ndims(in_arr)
            # for 1D, output has same size
            # TODO: return flattened size for multi-dimensional input
            if in_ndims == 1:
                return copy.copy(self.array_shape_classes[in_arr])
        elif call_name == 'linspace':
            # default is 50, arg3 is size
            LINSPACE_DEFAULT_SIZE = 50
            size = LINSPACE_DEFAULT_SIZE
            if len(args) >= 3:
                size = args[2].name
            new_class = self._get_next_class_with_size(size)
            return [new_class]
        elif call_name == 'dot':
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
            # for multi-dimensional arrays, last dimension of arg1 and second
            # to last dimension of arg2 should be equal since used in dot product.
            # if arg2 is 1D, its only dimension is used for dot product and
            # should be equal to second to last of arg1.
            assert len(args) == 2 or len(args) == 3
            in1 = args[0].name
            in2 = args[1].name
            ndims1 = self._get_ndims(in1)
            ndims2 = self._get_ndims(in2)
            c1 = self.array_shape_classes[in1][ndims1 - 1]
            c2 = UNKNOWN_CLASS

            if ndims2 == 1:
                c2 = self.array_shape_classes[in2][0]
            else:
                c2 = self.array_shape_classes[in2][ndims2 - 2]

            c_inner = self._merge_classes(c1, c2)

            c_out = []
            for i in range(ndims1 - 1):
                c_out.append(self.array_shape_classes[in1][i])
            for i in range(ndims2 - 2):
                c_out.append(self.array_shape_classes[in2][i])
            if ndims2 > 1:
                c_out.append(self.array_shape_classes[in2][ndims2 - 1])
            return c_out
        elif call_name in UFUNC_MAP_OP:
            return self._broadcast_and_match_shapes([a.name for a in args])

        if config.DEBUG_ARRAY_OPT == 1:
            print("unknown numpy call:", call_name, " ", args)
        return None

    def _get_second_arg_or_kw(self, args, kws, kw_name):
        arg_var = None
        if len(args) > 1:
            arg_var = args[1].name
        elif kw_name in kws:
            arg_var = kws[kw_name].name
        if arg_var is None:
            return 0  # default value is 0 if no arg
        if arg_var in self.constant_table:
            return self.constant_table[arg_var]
        return -1  # arg var is not constant

    def _get_sequence_arrs(self, seq_arg):
        """get array sequence input to concatenate, stack etc."""
        arr_args = []
        # arrays sequence argument can be tuple or list
        if seq_arg in self.list_table:
            arr_args = self.list_table[seq_arg]
        if seq_arg in self.tuple_table:
            arr_args = self.tuple_table[seq_arg]
        return arr_args

    def convert_seq_to_atleast_3d(self, seq_arg_var):
        """convert array sequence to a sequence of arrays with at least 3d dims
        """
        arr_args = self._get_sequence_arrs(seq_arg_var.name)
        new_seq = []
        for arr in arr_args:
            curr_dims = self._get_ndims(arr.name)
            if curr_dims < 3:
                dummy_var = ir.Var(
                    arr.scope, mk_unique_var("__dummy_nd"), arr.loc)
                self.typemap[dummy_var.name] = self.typemap[arr.name].copy(
                    ndim=3)
                corrs = self.array_shape_classes[arr.name].copy()
                if curr_dims == 0:
                    corrs = [CONST_CLASS] * 3
                elif curr_dims == 1:  # Numpy adds both prepends and appends a dim
                    corrs = [CONST_CLASS] + corrs + [CONST_CLASS]
                elif curr_dims == 2:  # append a dim
                    corrs = corrs + [CONST_CLASS]
                self.array_shape_classes[dummy_var.name] = corrs
                new_seq.append(dummy_var)
            else:
                new_seq.append(arr)
        tup_name = mk_unique_var("__dummy_tup")
        self.tuple_table[tup_name] = new_seq
        return ir.Var(arr_args[0].scope, tup_name, arr_args[0].loc)

    def _get_classes_from_shape(self, shape_arg):
        # shape is either Int or tuple of Int
        arg_typ = self.typemap[shape_arg.name]
        if isinstance(arg_typ, types.scalars.Integer):
            new_class = self._get_next_class_with_size(shape_arg)
            return [new_class]
        # TODO: handle A.reshape(c.shape)
        if (not isinstance(arg_typ, types.containers.UniTuple) or
                shape_arg.name not in self.tuple_table):
            return None
        out_eqs = []
        for i in range(arg_typ.count):
            new_class = self._get_next_class_with_size(
                self.tuple_table[shape_arg.name][i])
            out_eqs.append(new_class)
        return out_eqs

    def _get_classes_from_shape_list(self, shape_list):
        assert isinstance(shape_list, list)
        out_eqs = []
        for shape_arg in shape_list:
            arg_typ = self.typemap[shape_arg.name]
            assert isinstance(arg_typ, types.scalars.Integer)
            new_class = self._get_next_class_with_size(shape_arg)
            out_eqs.append(new_class)
        return out_eqs

    def _get_classes_from_const_shape(self, shape):
        # shape is either int or tuple/list of ints
        if isinstance(shape, int):
            new_class = self._get_next_class_with_size(shape)
            return [new_class]
        assert isinstance(shape, (tuple, list))
        out_eqs = []
        for dim in shape:
            new_class = self._get_next_class_with_size(dim)
            out_eqs.append(new_class)
        return out_eqs

    def _get_classes_from_dim_args(self, args):
        out = []
        for arg in args:
            new_class = self._get_next_class_with_size(arg)
            out.append(new_class)
        return out

    def _merge_equivalent_classes(self):
        changed = True
        while changed:
            curr_sizes = self.class_sizes.copy()
            changed = False
            for c1, sizes1 in curr_sizes.items():
                for c2, sizes2 in curr_sizes.items():
                    if c1 != c2 and set(sizes1) & set(sizes2) != set():
                        changed = True
                        self._merge_classes(c1, c2)
        return

    def _merge_classes(self, c1, c2):
        # no need to merge if equal classes already
        if c1 == c2:
            return c1

        new_class = self._get_next_class()
        for l in self.array_shape_classes.values():
            for i in range(len(l)):
                if l[i] == c1 or l[i] == c2:
                    l[i] = new_class
        # merge lists of size vars and remove previous classes
        self.class_sizes[new_class] = (self.class_sizes.pop(c1, [])
                                       + self.class_sizes.pop(c2, []))
        return new_class

    def _cleanup_analysis_data(self):
        # delete unused classes
        all_used_class = set()
        for shape_corrs in self.array_shape_classes.values():
            all_used_class |= set(shape_corrs)
        curr_class_sizes = self.class_sizes.copy()
        for c in curr_class_sizes.keys():
            if c not in all_used_class:
                self.class_sizes.pop(c)

        # delete repeated size variables
        new_class_sizes = {}
        for c, var_list in self.class_sizes.items():
            const_sizes = [v for v in var_list if not isinstance(v, ir.Var)]
            name_var_table = {
                v.name: v for v in var_list if isinstance(
                    v, ir.Var)}
            v_set = {v.name for v in var_list if isinstance(v, ir.Var)}
            new_class_sizes[c] = [name_var_table[vname]
                                  for vname in v_set] + const_sizes
        self.class_sizes = new_class_sizes
        return

    def _broadcast_and_match_shapes(self, args):
        """Infer shape equivalence of arguments based on Numpy broadcast rules
        and return shape of output
        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        """
        # at least one input has to be array, rest are constants
        assert any([self._isarray(a) for a in args])
        # list of array equivalences
        eqs = []
        for a in args:
            if self._isarray(a):
                eqs.append(copy.copy(self.array_shape_classes[a]))
            else:
                eqs.append([CONST_CLASS])  # constant variable
        ndims = max([len(e) for e in eqs])
        for e in eqs:
            # prepend size 1 dims to match shapes (broadcast rules)
            while len(e) < ndims:
                e.insert(0, CONST_CLASS)
        out_eq = [-1 for i in range(ndims)]

        for i in range(ndims):
            c = eqs[0][i]
            for e in eqs:
                if e[i] != CONST_CLASS and e[i] != c:
                    if c == CONST_CLASS:
                        c = e[i]
                    else:
                        c = self._merge_classes(c, e[i])
            out_eq[i] = c

        return out_eq

    def _isarray(self, varname):
        # no SmartArrayType support yet (can't generate parfor, allocate, etc)
        return (isinstance(self.typemap[varname], types.npytypes.Array) and
                not isinstance(self.typemap[varname],
                               types.npytypes.SmartArrayType))

    def _add_array_corr(self, varname):
        assert varname not in self.array_shape_classes
        self.array_shape_classes[varname] = []
        arr_typ = self.typemap[varname]
        for i in range(arr_typ.ndim):
            new_class = self._get_next_class()
            self.array_shape_classes[varname].append(new_class)
        return self.array_shape_classes[varname]

    def _get_next_class_with_size(self, size):
        if isinstance(size, int) and size == 1:
            return CONST_CLASS
        if (isinstance(size, ir.Var) and size.name in self.constant_table
                and self.constant_table[size.name] == 1):
            return CONST_CLASS
        new_class = self._get_next_class()
        self.class_sizes[new_class] = [size]
        return new_class

    def _get_next_class(self):
        m = self.next_eq_class
        self.next_eq_class += 1
        return m

    def _get_ndims(self, arr):
        # return len(self.array_analysis.array_shape_classes[arr])
        return self.typemap[arr].ndim


def copy_propagate_update_analysis(stmt, var_dict, array_analysis):
    """update array analysis data during copy propagation.
    If an array is in defs of a statement, we update its size variables.
    """
    array_shape_classes = array_analysis.array_shape_classes
    class_sizes = array_analysis.class_sizes
    array_size_vars = array_analysis.array_size_vars
    # find defs of stmt
    def_set = set()
    if isinstance(stmt, ir.Assign):
        def_set.add(stmt.target.name)
    for T, def_func in analysis.ir_extension_usedefs.items():
        if isinstance(stmt, T):
            _, def_set = def_func(stmt)
    # update analysis for arrays in defs
    for var in def_set:
        if var in array_shape_classes:
            if var in array_size_vars:
                array_size_vars[var] = replace_vars_inner(array_size_vars[var],
                                                          var_dict)
            shape_corrs = array_shape_classes[var]
            for c in shape_corrs:
                if c != -1:
                    class_sizes[c] = replace_vars_inner(
                        class_sizes[c], var_dict)
    return


UNARY_MAP_OP = list(
    npydecl.NumpyRulesUnaryArrayOperator._op_map.keys()) + ['+']
BINARY_MAP_OP = npydecl.NumpyRulesArrayOperator._op_map.keys()
UFUNC_MAP_OP = [f.__name__ for f in npydecl.supported_ufuncs]
