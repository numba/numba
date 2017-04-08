from __future__ import print_function, division, absolute_import
import types as pytypes # avoid confusion with numba.types
import sys

from numba import ir, ir_utils, types, rewrites, config, analysis
from numba import array_analysis, postproc
from numba.ir_utils import *
from numba.analysis import *
from numba.controlflow import CFGraph
from numba.typing import npydecl
from numba.types.functions import Function
import numpy as np
import numba.parfor
# circular dependency: import numba.npyufunc.dufunc.DUFunc

_reduction_ops = {
  'sum'  : ('+=', '+', 0),
  'prod' : ('*=', '*', 1),
}

class LoopNest(object):
    '''The LoopNest class holds information of a single loop including
    the index variable (of a non-negative integer value), and the
    range variable, e.g. range(r) is 0 to r-1 with step size 1.
    '''
    def __init__(self, index_variable, range_variable, correlation=-1):
        self.index_variable = index_variable
        self.range_variable = range_variable
        self.correlation = correlation

    def __repr__(self):
        return ("LoopNest(index_variable=%s, " % self.index_variable
                + "range_variable=%s, " % self.range_variable
                + "correlation=%d)" % self.correlation)


class Parfor(ir.Expr, ir.Stmt):
    def __init__(self, loop_nests, init_block, loop_body, loc, array_analysis, index_var):
        super(Parfor, self).__init__(
            op   = 'parfor',
            loc  = loc
        )

        #self.input_info  = input_info
        #self.output_info = output_info
        self.loop_nests = loop_nests
        self.init_block = init_block
        self.loop_body = loop_body
        self.array_analysis = array_analysis
        self.index_var = index_var

    def __repr__(self):
        return repr(self.loop_nests) + repr(self.loop_body) + repr(self.index_var)

    def list_vars(self):
        """list variables used (read/written) in this parfor by
        traversing the body and combining block uses.
        """
        all_uses = []
        for l,b in self.loop_body.items():
            for stmt in b.body:
                all_uses += stmt.list_vars()

        for loop in self.loop_nests:
            all_uses.append(loop.index_variable)
            all_uses.append(loop.range_variable)

        for stmt in self.init_block.body:
            all_uses += stmt.list_vars()

        return all_uses

    def dump(self,  file=None):
        file = file or sys.stdout
        print(("begin parfor").center(20,'-'), file=file)
        for loopnest in self.loop_nests:
            print(loopnest, file=file)
        print("init block:", file=file)
        self.init_block.dump()
        for offset, block in sorted(self.loop_body.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file)
        print("index_var = ", self.index_var)
        print(("end parfor").center(20,'-'), file=file)


class ParforPass(object):
    """ParforPass class is responsible for converting Numpy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """
    def __init__(self, func_ir, typemap, calltypes, return_type):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.return_type = return_type
        self.array_analysis = array_analysis.ArrayAnalysis(func_ir, typemap,
            calltypes)
        ir_utils._max_label = max(func_ir.blocks.keys())

    def _has_known_shape(self, var):
        """Return True if the given variable has fully known shape in array_analysis.
        """
        if isinstance(var, ir.Var) and var.name in self.array_analysis.array_shape_classes:
            var_shapes = self.array_analysis.array_shape_classes[var.name]
            return not (-1 in var_shapes)
        return False

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR."""

        self.array_analysis.run()
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            block = self.func_ir.blocks[label]
            new_body = []
            for instr in block.body:
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    lhs = instr.target
                    # only translate C order since we can't allocate F
                    if self._has_known_shape(lhs) and self._is_C_order(lhs.name):
                        if self._is_supported_npycall(expr):
                            instr = self._numpy_to_parfor(lhs, expr)
                        elif isinstance(expr, ir.Expr) and expr.op == 'arrayexpr':
                            instr = self._arrayexpr_to_parfor(lhs, expr)
                    elif self._is_supported_npyreduction(expr):
                        instr = self._reduction_to_parfor(lhs, expr)
                new_body.append(instr)
            block.body = new_body

        # remove Del statements for easier optimization
        remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after parfor pass")
        # get copies in to blocks and out from blocks
        in_cps, out_cps = copy_propagate(self.func_ir.blocks)
        # table mapping variable names to ir.Var objects to help replacement
        name_var_table = get_name_var_table(self.func_ir.blocks)
        apply_copy_propagate(self.func_ir.blocks, in_cps, name_var_table,
            array_analysis.copy_propagate_update_analysis, self.array_analysis,
            self.typemap, self.calltypes)
        # remove dead code to enable fusion
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        #dprint_func_ir(self.func_ir, "after remove_dead")
        # reorder statements to maximize fusion
        maximize_fusion(self.func_ir.blocks)
        fuse_parfors(self.func_ir.blocks)
        # remove dead code after fusion to remove extra arrays and variables
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        #dprint_func_ir(self.func_ir, "after second remove_dead")
        # push function call variables inside parfors so gufunc function
        # wouldn't need function variables as argument
        push_call_vars(self.func_ir.blocks, {}, {})
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        # after optimization, some size variables are not available anymore
        remove_dead_class_sizes(self.func_ir.blocks, self.array_analysis)
        dprint_func_ir(self.func_ir, "after optimization")
        if config.DEBUG_ARRAY_OPT==1:
            print("variable types: ",self.typemap)
            print("call types: ", self.calltypes)
        # run post processor again to generate Del nodes
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()
        if self.func_ir.is_generator:
            fix_generator_types(self.func_ir.generator_info, self.return_type,
                self.typemap)
        lower_parfor_sequential(self.func_ir, self.typemap, self.calltypes)
        return

    def _is_C_order(self, arr_name):
        typ = self.typemap[arr_name]
        assert isinstance(typ, types.npytypes.Array)
        return typ.layout=='C'

    def _make_index_var(self, scope, index_vars, body_block):
        ndims = len(index_vars)
        if ndims > 1:
            loc = body_block.loc
            tuple_var = ir.Var(scope, mk_unique_var("$parfor_index_tuple_var"), loc)
            self.typemap[tuple_var.name] = types.containers.UniTuple(types.int64, ndims)
            tuple_call = ir.Expr.build_tuple(list(index_vars), loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            body_block.body.append(tuple_assign)
            return tuple_var, types.containers.UniTuple(types.int64, ndims)
        else:
            return index_vars[0], types.int64

    def _arrayexpr_to_parfor(self, lhs, arrayexpr):
        """generate parfor from arrayexpr node, which is essentially a
        map with recursive tree.
        """
        scope = lhs.scope
        loc = lhs.loc
        expr = arrayexpr.expr
        arr_typ = self.typemap[lhs.name]
        el_typ = arr_typ.dtype

        # generate loopnests and size variables from lhs correlations
        loopnests = []
        size_vars = []
        index_vars = []
        for this_dim in range(arr_typ.ndim):
            corr = self.array_analysis.array_shape_classes[lhs.name][this_dim]
            size_var = self.array_analysis.array_size_vars[lhs.name][this_dim]
            size_vars.append(size_var)
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
            index_vars.append(index_var)
            self.typemap[index_var.name] = types.int64
            loopnests.append( LoopNest(index_var, size_var, corr) )

        # generate init block and body
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(self.typemap, self.calltypes, lhs,
            tuple(size_vars), el_typ, scope, loc)
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var("$expr_out_var"), loc)
        self.typemap[expr_out_var.name] = el_typ

        index_var, index_var_typ = self._make_index_var(scope, index_vars, body_block)

        body_block.body.extend(_arrayexpr_tree_to_ir(self.typemap, self.calltypes,
            expr_out_var, expr, index_var, index_vars, self.array_analysis.array_shape_classes))

        parfor = Parfor(loopnests, init_block, {}, loc, self.array_analysis, index_var)

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        self.calltypes[setitem_node] = signature(types.none,
            self.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label:body_block}
        if config.DEBUG_ARRAY_OPT==1:
            print("generated parfor for arrayexpr:")
            parfor.dump()
        return parfor

    def _is_supported_npycall(self, expr):
        """check if we support parfor translation for
        this Numpy call.
        """
        #return False # turn off for now
        if not (isinstance(expr, ir.Expr) and expr.op == 'call'):
            return False
        if expr.func.name not in self.array_analysis.numpy_calls.keys():
            return False
        # TODO: add more calls
        if self.array_analysis.numpy_calls[expr.func.name]=='dot':
            #only translate matrix/vector and vector/vector multiply to parfor
            # (don't translate matrix/matrix multiply)
            if (self._get_ndims(expr.args[0].name)<=2 and
                    self._get_ndims(expr.args[1].name)==1):
                return True
        return False

    def _is_supported_npyreduction(self, expr):
        """check if we support parfor translation for
        this Numpy reduce call.
        """
        #return False # turn off for now
        if not (isinstance(expr, ir.Expr) and expr.op == 'call'):
            return False
        if expr.func.name not in self.array_analysis.numpy_calls.keys():
            return False
        # TODO: add more calls
        if self.array_analysis.numpy_calls[expr.func.name] in _reduction_ops:
            for arg in expr.args:
                if not self._has_known_shape(arg):
                    return False
            return True
        return False

    def _get_ndims(self, arr):
        #return len(self.array_analysis.array_shape_classes[arr])
        return self.typemap[arr].ndim

    def _numpy_to_parfor(self, lhs, expr):
        assert isinstance(expr, ir.Expr) and expr.op == 'call'
        call_name = self.array_analysis.numpy_calls[expr.func.name]
        args = expr.args
        kws = dict(expr.kws)
        if call_name=='dot':
            assert len(args)==2 or len(args)==3
            # if 3 args, output is allocated already
            out = None
            if len(args)==3:
                out = args[2]
            if 'out' in kws:
                out = kws['out']

            in1 = args[0]
            in2 = args[1]
            el_typ = self.typemap[lhs.name].dtype
            assert self._get_ndims(in1.name)<=2 and self._get_ndims(in2.name)==1
            # loop range correlation is same as first dimention of 1st input
            corr = self.array_analysis.array_shape_classes[in1.name][0]
            size_var = self.array_analysis.array_size_vars[in1.name][0]
            scope = lhs.scope
            loc = expr.loc
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), lhs.loc)
            self.typemap[index_var.name] = types.int64
            loopnests = [ LoopNest(index_var, size_var, corr) ]
            init_block = ir.Block(scope, loc)
            parfor = Parfor(loopnests, init_block, {}, loc, self.array_analysis, index_var)
            if self._get_ndims(in1.name)==2:
                # for 2D input, there is an inner loop
                # correlation of inner dimension
                inner_size_var = self.array_analysis.array_size_vars[in1.name][1]
                # loop structure: range block, header block, body

                range_label = next_label()
                header_label = next_label()
                body_label = next_label()
                out_label = next_label()

                if out==None:
                    alloc_nodes = mk_alloc(self.typemap, self.calltypes, lhs,
                        size_var, el_typ, scope, loc)
                    init_block.body = alloc_nodes
                else:
                    out_assign = ir.Assign(out, lhs, loc)
                    init_block.body = [out_assign]
                init_block.body.extend(_gen_dotmv_check(self.typemap,
                    self.calltypes, in1, in2, lhs, scope, loc))
                # sum_var = 0
                const_node = ir.Const(0, loc)
                const_var = ir.Var(scope, mk_unique_var("$const"), loc)
                self.typemap[const_var.name] = el_typ
                const_assign = ir.Assign(const_node, const_var, loc)
                sum_var = ir.Var(scope, mk_unique_var("$sum_var"), loc)
                self.typemap[sum_var.name] = el_typ
                sum_assign = ir.Assign(const_var, sum_var, loc)

                range_block = mk_range_block(self.typemap, inner_size_var,
                    self.calltypes, scope, loc)
                range_block.body = [const_assign, sum_assign] + range_block.body
                range_block.body[-1].target = header_label # fix jump target
                phi_var = range_block.body[-2].target

                header_block = mk_loop_header(self.typemap, phi_var,
                    self.calltypes, scope, loc)
                header_block.body[-1].truebr = body_label
                header_block.body[-1].falsebr = out_label
                phi_b_var = header_block.body[-2].target

                body_block = _mk_mvdot_body(self.typemap, self.calltypes,
                    phi_b_var, index_var, in1, in2,
                    sum_var, scope, loc, el_typ)
                body_block.body[-1].target = header_label

                out_block = ir.Block(scope, loc)
                # lhs[parfor_index] = sum_var
                setitem_node = ir.SetItem(lhs, index_var, sum_var, loc)
                self.calltypes[setitem_node] = signature(types.none,
                    self.typemap[lhs.name], types.int64, el_typ)
                out_block.body = [setitem_node]
                parfor.loop_body = {range_label:range_block,
                    header_label:header_block, body_label:body_block,
                    out_label:out_block}
            else: # self._get_ndims(in1.name)==1 (reduction)
                NotImplementedError("no reduction for dot() "+expr)
            if config.DEBUG_ARRAY_OPT==1:
                print("generated parfor for numpy call:")
                parfor.dump()
            return parfor
        # return error if we couldn't handle it (avoid rewrite infinite loop)
        raise NotImplementedError("parfor translation failed for ", expr)

    def _reduction_to_parfor(self, lhs, expr):
        assert isinstance(expr, ir.Expr) and expr.op == 'call'
        call_name = self.array_analysis.numpy_calls[expr.func.name]
        args = expr.args
        kws = dict(expr.kws)
        if call_name in _reduction_ops:
            acc_op, im_op, init_val = _reduction_ops[call_name]
            assert len(args)==1
            in1 = args[0]
            arr_typ = self.typemap[in1.name]
            in_typ = arr_typ.dtype
            im_op_func_typ = find_op_typ(im_op, [in_typ, in_typ])
            el_typ = im_op_func_typ.return_type
            ndims = arr_typ.ndim

            # For full reduction, loop range correlation is same as 1st input
            corrs = self.array_analysis.array_shape_classes[in1.name]
            sizes = self.array_analysis.array_size_vars[in1.name]
            assert ndims == len(sizes) and ndims == len(corrs)
            scope = lhs.scope
            loc = expr.loc
            loopnests = []
            parfor_index = []
            for i in range(ndims):
                index_var = ir.Var(scope, mk_unique_var("$parfor_index" + str(i)), loc)
                self.typemap[index_var.name] = types.int64
                parfor_index.append(index_var)
                loopnests.append(LoopNest(index_var, sizes[i], corrs[i]))

            acc_var = lhs

            # init value
            init_const = ir.Const(el_typ(init_val), loc)

            # init block has to init the reduction variable
            init_block = ir.Block(scope, loc)
            init_block.body.append(ir.Assign(init_const, acc_var, loc))

            # loop body accumulates acc_var
            acc_block = ir.Block(scope, loc)
            tmp_var = ir.Var(scope, mk_unique_var("$val"), loc)
            self.typemap[tmp_var.name] = in_typ
            index_var, index_var_type = self._make_index_var(scope, parfor_index, acc_block)
            getitem_call = ir.Expr.getitem(in1, index_var, loc)
            self.calltypes[getitem_call] = signature(in_typ, arr_typ, index_var_type)
            acc_block.body.append(ir.Assign(getitem_call, tmp_var, loc))
            acc_call = ir.Expr.inplace_binop(acc_op, im_op, acc_var, tmp_var, loc)
            # for some reason, type template of += returns None,
            # so type template of + should be used
            self.calltypes[acc_call] = im_op_func_typ
            acc_block.body.append(ir.Assign(acc_call, acc_var, loc))
            loop_body = { next_label() : acc_block }

            # parfor
            parfor = Parfor(loopnests, init_block, loop_body, loc,
                self.array_analysis, index_var)
            return parfor
        # return error if we couldn't handle it (avoid rewrite infinite loop)
        raise NotImplementedError("parfor translation failed for ", expr)

def _gen_dotmv_check(typemap, calltypes, in1, in2, out, scope, loc):
    """compile dot() check from linalg module and insert a call to it"""
    # save max_label since pipeline is called recursively
    saved_max_label = ir_utils._max_label
    from numba import njit
    from numba.targets.linalg import dot_3_mv_check_args
    check_func = njit(dot_3_mv_check_args)
    # g_var = Global(dot_3_mv_check_args)
    g_var = ir.Var(scope, mk_unique_var("$check_mv"), loc)
    func_typ = types.functions.Dispatcher(check_func)
    typemap[g_var.name] = func_typ
    g_obj = ir.Global("dot_3_mv_check_args", check_func, loc)
    g_assign = ir.Assign(g_obj, g_var, loc)
    # dummy_var = call g_var(in1, in2, out)
    call_node = ir.Expr.call(g_var, [in1,in2,out], (), loc)
    calltypes[call_node] = func_typ.get_call_type(typing.Context(),
        [typemap[in1.name], typemap[in2.name], typemap[out.name]], {})
    dummy_var = ir.Var(scope, mk_unique_var("$call_out_dummy"), loc)
    typemap[dummy_var.name] = types.none
    call_assign = ir.Assign(call_node, dummy_var, loc)
    ir_utils._max_label = saved_max_label
    return [g_assign, call_assign]

def _mk_mvdot_body(typemap, calltypes, phi_b_var, index_var, in1, in2, sum_var,
        scope, loc, el_typ):
    """generate array inner product (X[p,:], v[:]) for parfor of np.dot(X,v)"""
    body_block = ir.Block(scope, loc)
    # inner_index = phi_b_var
    inner_index = ir.Var(scope, mk_unique_var("$inner_index"), loc)
    typemap[inner_index.name] = types.int64
    inner_index_assign = ir.Assign(phi_b_var, inner_index, loc)
    # tuple_var = build_tuple(index_var, inner_index)
    tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
    typemap[tuple_var.name] = types.containers.UniTuple(types.int64, 2)
    tuple_call = ir.Expr.build_tuple([index_var, inner_index], loc)
    tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
    # X_val = getitem(X, tuple_var)
    X_val = ir.Var(scope, mk_unique_var("$"+in1.name+"_val"), loc)
    typemap[X_val.name] = el_typ
    getitem_call = ir.Expr.getitem(in1, tuple_var, loc)
    calltypes[getitem_call] = signature(el_typ, typemap[in1.name],
        typemap[tuple_var.name])
    getitem_assign = ir.Assign(getitem_call, X_val, loc)
    # v_val = getitem(V, inner_index)
    v_val = ir.Var(scope, mk_unique_var("$"+in2.name+"_val"), loc)
    typemap[v_val.name] = el_typ
    v_getitem_call = ir.Expr.getitem(in2, inner_index, loc)
    calltypes[v_getitem_call] = signature(el_typ, typemap[in2.name], types.int64)
    v_getitem_assign = ir.Assign(v_getitem_call, v_val, loc)
    # add_var = X_val * v_val
    add_var = ir.Var(scope, mk_unique_var("$add_var"), loc)
    typemap[add_var.name] = el_typ
    add_call = ir.Expr.binop('*', X_val, v_val, loc)
    calltypes[add_call] = signature(el_typ, el_typ, el_typ)
    add_assign = ir.Assign(add_call, add_var, loc)
    # acc_var = sum_var + add_var
    acc_var = ir.Var(scope, mk_unique_var("$acc_var"), loc)
    typemap[acc_var.name] = el_typ
    acc_call = ir.Expr.inplace_binop('+=', '+', sum_var, add_var, loc)
    calltypes[acc_call] = signature(el_typ, el_typ, el_typ)
    acc_assign = ir.Assign(acc_call, acc_var, loc)
    # sum_var = acc_var
    final_assign = ir.Assign(acc_var, sum_var, loc)
    # jump to header
    b_jump_header = ir.Jump(-1, loc)
    body_block.body = [inner_index_assign, tuple_assign,
        getitem_assign, v_getitem_assign, add_assign, acc_assign,
        final_assign, b_jump_header]
    return body_block

def _arrayexpr_tree_to_ir(typemap, calltypes, expr_out_var, expr,
        parfor_index_tuple_var, all_parfor_indices, array_shape_classes):
    """generate IR from array_expr's expr tree recursively. Assign output to
    expr_out_var and returns the whole IR as a list of Assign nodes.
    """
    el_typ = typemap[expr_out_var.name]
    scope = expr_out_var.scope
    loc = expr_out_var.loc
    out_ir = []

    if isinstance(expr, tuple):
        op, arr_expr_args = expr
        arg_vars = []
        for arg in arr_expr_args:
            arg_out_var = ir.Var(scope, mk_unique_var("$arg_out_var"), loc)
            typemap[arg_out_var.name] = el_typ
            out_ir += _arrayexpr_tree_to_ir(typemap, calltypes,
                arg_out_var, arg, parfor_index_tuple_var, all_parfor_indices,
                    array_shape_classes)
            arg_vars.append(arg_out_var)
        if op in npydecl.supported_array_operators:
            el_typ1 = typemap[arg_vars[0].name]
            if len(arg_vars)==2:
                el_typ2 = typemap[arg_vars[1].name]
                func_typ = find_op_typ(op, [el_typ1, el_typ2])
                ir_expr = ir.Expr.binop(op, arg_vars[0], arg_vars[1], loc)
                if op=='/':
                    func_typ, ir_expr = _gen_np_divide(arg_vars[0], arg_vars[1], out_ir, typemap)
            else:
                func_typ = find_op_typ(op, [el_typ1])
                ir_expr = ir.Expr.unary(op, arg_vars[0], loc)
            calltypes[ir_expr] = func_typ
            el_typ = func_typ.return_type
            out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
        for T in array_analysis.MAP_TYPES:
            if isinstance(op, T):
                # elif isinstance(op, (np.ufunc, DUFunc)):
                # function calls are stored in variables which are not removed
                # op is typing_key to the variables type
                func_var = ir.Var(scope, _find_func_var(typemap, op), loc)
                ir_expr = ir.Expr.call(func_var, arg_vars, (), loc)
                call_typ = typemap[func_var.name].get_call_type(
                    typing.Context(), [el_typ]*len(arg_vars), {})
                calltypes[ir_expr] = call_typ
                el_typ = call_typ.return_type
                #signature(el_typ, el_typ)
                out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Var):
        var_typ = typemap[expr.name]
        if isinstance(var_typ, types.Array):
            el_typ = var_typ.dtype
            ir_expr = _gen_arrayexpr_getitem(expr, parfor_index_tuple_var,
                all_parfor_indices, el_typ, calltypes, typemap,
                array_shape_classes, out_ir)
        else:
            # assert typemap[expr.name]==el_typ
            el_typ = var_typ
            ir_expr = expr
        out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Const):
        el_typ = typing.Context().resolve_value_type(expr.value)
        out_ir.append(ir.Assign(expr, expr_out_var, loc))

    if len(out_ir)==0:
        raise NotImplementedError(
            "Don't know how to translate array expression '%r'" % (expr,))
    typemap.pop(expr_out_var.name, None)
    typemap[expr_out_var.name] = el_typ
    return out_ir

def _gen_np_divide(arg1, arg2, out_ir, typemap):
    """generate np.divide() instead of / for array_expr to get numpy error model
    like inf for division by zero (test_division_by_zero).
    """
    scope = arg1.scope
    loc = arg1.loc
    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: div_attr = getattr(g_np_var, divide)
    div_attr_call = ir.Expr.getattr(g_np_var, "divide", loc)
    attr_var = ir.Var(scope, mk_unique_var("$div_attr"), loc)
    func_var_typ = get_np_ufunc_typ(np.divide)
    typemap[attr_var.name] = func_var_typ
    attr_assign = ir.Assign(div_attr_call, attr_var, loc)
    # divide call:  div_attr(arg1, arg2)
    div_call = ir.Expr.call(attr_var, [arg1, arg2], (), loc)
    func_typ = func_var_typ.get_call_type(typing.Context(),
        [typemap[arg1.name], typemap[arg2.name]], {})
    out_ir.extend([g_np_assign, attr_assign])
    return func_typ, div_call

def _gen_arrayexpr_getitem(var, parfor_index_tuple_var, all_parfor_indices,
        el_typ, calltypes, typemap, array_shape_classes, out_ir):
    """if there is implicit dimension broadcast, generate proper access variable
    for getitem. For example, if indices are (i1,i2,i3) but shape correlations
    are (c1,0,c3), generate a tuple with (i1,0,i3) for access.
    Another example: for (i1,i2,i3) and (c1,c2) generate (i2,i3).
    """
    loc = var.loc
    index_var = parfor_index_tuple_var
    shape_corrs = copy.copy(array_shape_classes[var.name])
    ndims = typemap[var.name].ndim
    num_indices = len(all_parfor_indices)
    if 0 in shape_corrs or ndims<num_indices:
        if ndims==1:
            # broadcast prepends dimensions, so use last index for 1D arrays
            index_var = all_parfor_indices[-1]
        else:
            # broadcast prepends dimensions so ignore indices from beginning
            ind_offset = num_indices-ndims
            tuple_var = ir.Var(var.scope,
                mk_unique_var("$parfor_index_tuple_var_bcast"), loc)
            typemap[tuple_var.name] = types.containers.UniTuple(types.int64,
                ndims)
            # const var for size 1 dim access index: $const0 = Const(0)
            const_node = ir.Const(0, var.loc)
            const_var = ir.Var(var.scope, mk_unique_var("$const_ind_0"), loc)
            typemap[const_var.name] = types.int64
            const_assign = ir.Assign(const_node, const_var, loc)
            out_ir.append(const_assign)
            index_vars = []
            for i in reversed(range(ndims)):
                if shape_corrs[i]==0:
                    index_vars.append(const_var)
                else:
                    index_vars.append(all_parfor_indices[ind_offset+i])
            index_vars = list(reversed(index_vars))
            tuple_call = ir.Expr.build_tuple(index_vars, loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            out_ir.append(tuple_assign)
            index_var = tuple_var

    ir_expr =  ir.Expr.getitem(var, index_var, loc)
    calltypes[ir_expr] = signature(el_typ, typemap[var.name],
        typemap[index_var.name])
    return ir_expr

def _find_func_var(typemap, func):
    """find variable in typemap which represents the function func.
    """
    for k,v in typemap.items():
        # Function types store actual functions in typing_key.
        if isinstance(v, Function) and v.typing_key==func:
            return k
    raise RuntimeError("ufunc call variable not found")

def lower_parfor_sequential(func_ir, typemap, calltypes):
    parfor_found = False
    new_blocks = {}
    for (block_label, block) in func_ir.blocks.items():
        scope = block.scope
        i = _find_first_parfor(block.body)
        while i!=-1:
            parfor_found = True
            inst = block.body[i]
            loc = inst.init_block.loc
            # split block across parfor
            prev_block = ir.Block(scope, loc)
            prev_block.body = block.body[:i]
            block.body = block.body[i+1:]
            # previous block jump to parfor init block
            init_label = next_label()
            prev_block.body.append(ir.Jump(init_label, loc))
            new_blocks[init_label] = inst.init_block
            new_blocks[block_label] = prev_block
            block_label = next_label()

            ndims = len(inst.loop_nests)
            for i in range(ndims):
                loopnest = inst.loop_nests[i]
                # create range block for loop
                range_label = next_label()
                header_label = next_label()
                range_block = mk_range_block(typemap, loopnest.range_variable,
                    calltypes, scope, loc)
                range_block.body[-1].target = header_label # fix jump target
                phi_var = range_block.body[-2].target
                new_blocks[range_label] = range_block
                header_block = mk_loop_header(typemap, phi_var, calltypes,
                    scope, loc)
                header_block.body[-2].target = loopnest.index_variable
                new_blocks[header_label] = header_block
                # jump to this new inner loop
                if i==0:
                    inst.init_block.body.append(ir.Jump(range_label, loc))
                    header_block.body[-1].falsebr = block_label
                else:
                    new_blocks[prev_header_label].body[-1].truebr = range_label
                    header_block.body[-1].falsebr = prev_header_label
                prev_header_label = header_label # to set truebr next loop

            # last body block jump to inner most header
            body_last_label = max(inst.loop_body.keys())
            inst.loop_body[body_last_label].body.append(
                ir.Jump(header_label, loc))
            # inner most header jumps to first body block
            body_first_label = min(inst.loop_body.keys())
            header_block.body[-1].truebr = body_first_label
            # add parfor body to blocks
            for (l, b) in inst.loop_body.items():
                new_blocks[l] = b
            i = _find_first_parfor(block.body)

        # old block stays either way
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    # rename only if parfor found and replaced (avoid test_flow_control error)
    if parfor_found:
        func_ir.blocks = _rename_labels(func_ir.blocks)
    dprint_func_ir(func_ir, "after parfor sequential lowering")
    return

def _find_first_parfor(body):
    for (i, inst) in enumerate(body):
        if isinstance(inst, Parfor):
            return i
    return -1

def _rename_labels(blocks):
    """rename labels of function body blocks according to topological sort.
    lowering requires this order.
    """
    topo_order = find_topo_order(blocks)

    # make a block with return last if available (just for readability)
    return_label = -1
    for l,b in blocks.items():
        if isinstance(b.body[-1], ir.Return):
            return_label = l
    # some cases like generators can have no return blocks
    if return_label!=-1:
        topo_order.remove(return_label)
        topo_order.append(return_label)

    label_map = {}
    new_label = 0
    for label in topo_order:
        label_map[label] = new_label
        new_label += 1
    # update target labels in jumps/branches
    for b in blocks.values():
        term = b.terminator
        if isinstance(term, ir.Jump):
            term.target = label_map[term.target]
        if isinstance(term, ir.Branch):
            term.truebr = label_map[term.truebr]
            term.falsebr = label_map[term.falsebr]
    # update blocks dictionary keys
    new_blocks = {}
    for k, b in blocks.items():
        new_label = label_map[k]
        new_blocks[new_label] = b

    return new_blocks

def get_parfor_params(parfor):
    """find variables used in body of parfor from outside.
    computed as live variables at entry of first block.
    """
    blocks = wrap_parfor_blocks(parfor)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    unwrap_parfor_blocks(parfor)
    keylist = sorted(live_map.keys())
    first_non_init_block = keylist[1]

    # remove parfor index variables since they are not input
    for l in parfor.loop_nests:
        live_map[first_non_init_block] -= {l.index_variable.name}

    return sorted(live_map[first_non_init_block])

def get_parfor_outputs(parfor):
    """get arrays that are written to inside the parfor and need to be passed
    as parameters to gufunc.
    """
    # FIXME: The following assumes the target of all SetItem are outputs, which is wrong!
    last_label = max(parfor.loop_body.keys())
    outputs = []
    for blk in parfor.loop_body.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.SetItem):
                if stmt.index.name==parfor.index_var.name:
                    outputs.append(stmt.target.name)
    parfor_params = get_parfor_params(parfor)
    # make sure these written arrays are in parfor parameters (live coming in)
    outputs = list(set(outputs) & set(parfor_params))
    return sorted(outputs)

def get_parfor_reductions(parfor):
    """get variables that are accumulated using inplace_binop inside the parfor
    and need to be passed as reduction parameters to gufunc.
    """
    last_label = max(parfor.loop_body.keys())
    reductions = {}
    names = []
    parfor_params = get_parfor_params(parfor)
    for blk in parfor.loop_body.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and stmt.value.op == "inplace_binop":
                name = stmt.value.lhs.name
                if name in parfor_params:
                    names.append(name)
                    reductions[name] = (stmt.value.fn, stmt.value.immutable_fn)
    return sorted(names), reductions

def visit_vars_parfor(parfor, callback, cbdata):
    if config.DEBUG_ARRAY_OPT==1:
        print("visiting parfor vars for:",parfor)
        print("cbdata: ", sorted(cbdata.items()))
    for l in parfor.loop_nests:
        l.index_variable = visit_vars_inner(l.index_variable, callback, cbdata)
        l.range_variable = visit_vars_inner(l.range_variable, callback, cbdata)
    visit_vars({-1:parfor.init_block}, callback, cbdata)
    visit_vars(parfor.loop_body, callback, cbdata)
    return

# add call to visit parfor variable
ir_utils.visit_vars_extensions[Parfor] = visit_vars_parfor

def parfor_defs(parfor):
    """list variables written in this parfor by recursively
    calling compute_use_defs() on body and combining block defs.
    """
    all_defs = set()
    # index variables are sematically defined here
    for l in parfor.loop_nests:
        all_defs.add(l.index_variable.name)

    # all defs of body blocks
    for l,b in parfor.loop_body.items():
        for stmt in b.body:
            if isinstance(stmt, ir.Assign):
                all_defs.add(stmt.target.name)
            elif isinstance(stmt, Parfor):
                all_defs.update(parfor_defs(stmt))

    # all defs of init block
    for stmt in parfor.init_block.body:
        if isinstance(stmt, ir.Assign):
            all_defs.add(stmt.target.name)
        elif isinstance(stmt, Parfor):
            all_defs.update(parfor_defs(stmt))

    return all_defs

analysis.ir_extension_defs[Parfor] = parfor_defs

def parfor_insert_dels(parfor, curr_dead_set):
    """insert dels in parfor. input: dead variable set right after parfor.
    returns the variables for which del was inserted.
    """
    blocks = wrap_parfor_blocks(parfor)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    dead_map = compute_dead_maps(cfg, blocks, live_map, usedefs.defmap)
    # treat loop variables and size variables as live
    loop_vars = {l.range_variable.name for l in parfor.loop_nests}
    loop_vars |= {l.index_variable.name for l in parfor.loop_nests}
    for var_list in parfor.array_analysis.array_size_vars.values():
        loop_vars |= {v.name for v in var_list if isinstance(v, ir.Var)}
    dead_set = set()
    # TODO: handle escaping deads
    escaping_dead = {}
    for label in blocks.keys():
        # only kill vars that are actually dead at the parfor's block
        dead_map.internal[label] &= curr_dead_set
        dead_map.internal[label] -= loop_vars
        dead_set |= dead_map.internal[label]
        escaping_dead[label] = set()
    # dummy class to replace func_ir. _patch_var_dels only accesses blocks
    class DummyFuncIR(object):
        def __init__(self, blocks):
            self.blocks = blocks
    post_proc = postproc.PostProcessor(DummyFuncIR(blocks))
    post_proc._patch_var_dels(dead_map.internal, escaping_dead)
    unwrap_parfor_blocks(parfor)
    return dead_set

postproc.ir_extension_insert_dels[Parfor] = parfor_insert_dels

# reorder statements to maximize fusion
def maximize_fusion(blocks):
    for block in blocks.values():
        order_changed = True
        while order_changed:
            order_changed = False
            i = 0
            while i<len(block.body)-2:
                stmt = block.body[i]
                next_stmt = block.body[i+1]
                # swap only parfors with non-parfors
                # if stmt hasn't written to any variable that next_stmt uses,
                # they can be swapped
                if isinstance(stmt, Parfor) and not isinstance(next_stmt, Parfor):
                    stmt_writes = get_parfor_writes(stmt)
                    next_accesses = {v.name for v in next_stmt.list_vars()}
                    if stmt_writes & next_accesses == set():
                        block.body[i] = next_stmt
                        block.body[i+1] = stmt
                        order_changed = True
                i += 1
    return

def get_parfor_writes(parfor):
    assert isinstance(parfor, Parfor)
    writes = set()
    blocks = parfor.loop_body.copy()
    blocks[-1] = parfor.init_block
    for block in blocks.values():
        for stmt in block.body:
            if isinstance(stmt, (ir.Assign, ir.SetItem, ir.StaticSetItem)):
                writes.add(stmt.target.name)
            elif isinstance(stmt, Parfor):
                writes.update(get_parfor_writes(stmt))
    return writes

def fuse_parfors(blocks):
    for block in blocks.values():
        fusion_happened = True
        while fusion_happened:
            fusion_happened = False
            new_body = []
            i = 0
            while i<len(block.body)-1:
                stmt = block.body[i]
                next_stmt = block.body[i+1]
                if isinstance(stmt, Parfor) and isinstance(next_stmt, Parfor):
                    fused_node = try_fuse(stmt, next_stmt)
                    if fused_node is not None:
                        fusion_happened = True
                        new_body.append(fused_node)
                        i += 2
                        continue
                new_body.append(stmt)
                i += 1
            new_body.append(block.body[-1])
            block.body = new_body
    return

def try_fuse(parfor1, parfor2):
    """try to fuse parfors and return a fused parfor, otherwise return None
    """
    dprint("try_fuse trying to fuse \n",parfor1,"\n",parfor2)

    # fusion of parfors with different dimensions not supported yet
    if len(parfor1.loop_nests)!=len(parfor1.loop_nests):
        dprint("try_fuse parfors number of dimensions mismatch")
        return None

    ndims = len(parfor1.loop_nests)
    # all loops should be equal length
    for i in range(ndims):
        if parfor1.loop_nests[i].correlation!=parfor2.loop_nests[i].correlation:
            dprint("try_fuse parfor dimension correlation mismatch", i)
            return None

    # TODO: make sure parfor1's reduction output is not used in parfor2
    # only data parallel loops
    if has_cross_iter_dep(parfor1) or has_cross_iter_dep(parfor2):
        dprint("try_fuse parfor cross iteration dependency found")
        return None

    # make sure parfor2's init block isn't using any output of parfor1
    parfor1_body_usedefs = compute_use_defs(parfor1.loop_body)
    parfor1_body_vardefs = set()
    for defs in parfor1_body_usedefs.defmap.values():
        parfor1_body_vardefs |= defs
    init2_uses = compute_use_defs({0:parfor2.init_block}).usemap[0]
    if not parfor1_body_vardefs.isdisjoint(init2_uses):
        dprint("try_fuse parfor2 init block depends on parfor1 body")
        return None

    return fuse_parfors_inner(parfor1, parfor2)

def fuse_parfors_inner(parfor1, parfor2):
    # fuse parfor2 into parfor1
    # append parfor2's init block on parfor1's
    parfor1.init_block.body.extend(parfor2.init_block.body)

    # append parfor2's first block to parfor1's last block
    parfor2_first_label = min(parfor2.loop_body.keys())
    parfor2_first_block = parfor2.loop_body[parfor2_first_label].body
    parfor1_last_label = max(parfor1.loop_body.keys())
    parfor1.loop_body[parfor1_last_label].body.extend(parfor2_first_block)

    # add parfor2 body blocks to parfor1's except first
    parfor1.loop_body.update(parfor2.loop_body)
    parfor1.loop_body.pop(parfor2_first_label)

    # replace parfor2 indices with parfor1's
    ndims = len(parfor1.loop_nests)
    index_dict = {}
    for i in range(ndims):
        index_dict[parfor2.loop_nests[i].index_variable.name] = parfor1.loop_nests[i].index_variable
    replace_vars(parfor1.loop_body, index_dict)

    return parfor1

def has_cross_iter_dep(parfor):
    # we consevatively assume there is cross iteration dependency when
    # the parfor index is used in any expression since the expression could
    # be used for indexing arrays
    # TODO: make it more accurate using ud-chains
    indices = { l.index_variable for l in parfor.loop_nests }
    for b in parfor.loop_body.values():
        for stmt in b.body:
            # GetItem/SetItem nodes are fine since can't have expression inside
            # and only simple indices are possible
            if isinstance(stmt, (ir.SetItem, ir.StaticSetItem)):
                continue
            # tuples are immutable so no expression on parfor possible
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                op = stmt.value.op
                if op in ['build_tuple', 'getitem', 'static_getitem']:
                    continue
            # other statements can have potential violations
            if not indices.isdisjoint(stmt.list_vars()):
                dprint("has_cross_iter_dep found", indices, stmt)
                return True
    return False

def dprint(*s):
    if config.DEBUG_ARRAY_OPT==1:
        print(*s)

def remove_dead_parfor(parfor, lives, args):
    # remove dead get/sets in last block
    # FIXME: I think that "in the last block" is not sufficient in general.  We might need to
    # remove from any block.
    last_label = max(parfor.loop_body.keys())
    last_block = parfor.loop_body[last_label]

    # save array values set to replace getitems
    saved_values = {}
    new_body = []
    for stmt in last_block.body:
        if (isinstance(stmt, ir.SetItem) and stmt.index.name==parfor.index_var.name
                and stmt.target.name not in lives):
            saved_values[stmt.target.name] = stmt.value
            continue
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
            rhs = stmt.value
            if rhs.op=='getitem' and rhs.index.name==parfor.index_var.name:
                # replace getitem if value saved
                stmt.value = saved_values.get(rhs.value.name, rhs)
        new_body.append(stmt)
    last_block.body = new_body
    # process parfor body recursively
    remove_dead_parfor_recursive(parfor, lives, args)
    return

ir_utils.remove_dead_extensions[Parfor] = remove_dead_parfor

def remove_dead_parfor_recursive(parfor, lives, args):
    """create a dummy function from parfor and call remove dead recursively
    """
    blocks = parfor.loop_body.copy() # shallow copy is enough
    first_body_block = min(blocks.keys())
    assert first_body_block > 0 # we are using 0 for init block here
    last_label = max(blocks.keys())
    if len(blocks[last_label].body) == 0:
        return
    loc = blocks[last_label].body[-1].loc
    scope = blocks[last_label].scope

    # add dummy jump in init_block for CFG to work
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(first_body_block, loc))
    # add lives in a dummpy return to last block to avoid their removal
    tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
    live_vars = [ ir.Var(scope,v,loc) for v in lives ]
    tuple_call = ir.Expr.build_tuple(live_vars, loc)
    blocks[last_label].body.append(ir.Assign(tuple_call, tuple_var, loc))
    blocks[last_label].body.append(ir.Return(tuple_var,loc))
    remove_dead(blocks, args)
    blocks[0].body.pop() # remove dummy jump
    blocks[last_label].body.pop() # remove dummy return
    blocks[last_label].body.pop() # remove dummy tupple
    return

def remove_dead_class_sizes(blocks, array_analysis):
    usedefs = compute_use_defs(blocks)
    all_defs = set()
    # all available variables after dead code elimination
    for defs in usedefs.defmap.values():
        all_defs.update(defs)

    # remove dead size variables in class sizes
    for varlist in array_analysis.class_sizes.values():
        vars_to_remove = []
        for v in varlist:
            # don't remove constants
            if isinstance(v, ir.Var) and v.name not in all_defs:
                vars_to_remove.append(v)
        for v in vars_to_remove:
            varlist.remove(v)

    # replace array size variables with their class size variable if available
    for var, dim_sizes in array_analysis.array_size_vars.items():
        ndims = len(dim_sizes)
        shape_classes = array_analysis.array_shape_classes[var]
        for i in range(ndims):
            corr = shape_classes[i]
            if corr!=-1 and len(array_analysis.class_sizes[corr])>0:
                class_size = array_analysis.class_sizes[corr][0]
                dim_sizes[i] = class_size
            # delete dead size vars
            if isinstance(dim_sizes[i], ir.Var) and dim_sizes[i].name not in all_defs:
                dim_sizes[i] = None
    return

def wrap_parfor_blocks(parfor):
    """wrap parfor blocks for analysis/optimization like CFG"""
    blocks = parfor.loop_body.copy() # shallow copy is enough
    first_body_block = min(blocks.keys())
    assert first_body_block > 0 # we are using 0 for init block here
    last_label = max(blocks.keys())
    loc = blocks[last_label].body[-1].loc

    # add dummy jump in init_block for CFG to work
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(first_body_block, loc))
    blocks[last_label].body.append(ir.Jump(first_body_block,loc))
    return blocks

def unwrap_parfor_blocks(parfor):
    last_label = max(parfor.loop_body.keys())
    parfor.init_block.body.pop() # remove dummy jump
    parfor.loop_body[last_label].body.pop() # remove dummy return
    return

def get_copies_parfor(parfor):
    """find copies generated/killed by parfor"""
    blocks = wrap_parfor_blocks(parfor)
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks)
    in_gen_copies, in_extra_kill = get_block_copies(blocks)
    unwrap_parfor_blocks(parfor)

    # parfor's extra kill is all possible gens and kills of it's loop
    kill_set = in_extra_kill[0]
    for label in parfor.loop_body.keys():
        kill_set |= { l for l,r in in_gen_copies[label] }
    last_label = max(parfor.loop_body.keys())
    if config.DEBUG_ARRAY_OPT==1:
        print("copy propagate parfor out_copies:",
            out_copies_parfor[last_label], "kill_set",kill_set)
    return out_copies_parfor[last_label], kill_set

copy_propagate_extensions[Parfor] = get_copies_parfor

def apply_copies_parfor(parfor, var_dict, name_var_table, ext_func, ext_data,
        typemap, calltypes):
    """apply copy propagate recursively in parfor"""
    blocks = wrap_parfor_blocks(parfor)
    # add dummy assigns for each copy
    assign_list = []
    for lhs_name, rhs in var_dict.items():
        assign_list.append(ir.Assign(rhs, name_var_table[lhs_name],
            ir.Loc("dummy",-1)))
    blocks[0].body = assign_list+blocks[0].body
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks)
    apply_copy_propagate(blocks, in_copies_parfor, name_var_table, ext_func,
        ext_data, typemap, calltypes)
    unwrap_parfor_blocks(parfor)
    # remove dummy assignments
    blocks[0].body = blocks[0].body[len(assign_list):]
    return

apply_copy_propagate_extensions[Parfor] = apply_copies_parfor

def push_call_vars(blocks, saved_globals, saved_getattrs):
    """push call variables to right before their call site.
    assuming one global/getattr is created for each call site and control flow
    doesn't change it.
    """
    for block in blocks.values():
        new_body = []
        # global/attr variables that are defined in this block already,
        #   no need to reassign them
        block_defs = set()
        for stmt in block.body:
            if isinstance(stmt, ir.Assign):
                rhs = stmt.value
                lhs = stmt.target
                if (isinstance(rhs, ir.Global)):
                        #and isinstance(rhs.value, pytypes.ModuleType)):
                    saved_globals[lhs.name] = stmt
                    block_defs.add(lhs.name)
                    #continue
                elif isinstance(rhs, ir.Expr) and rhs.op=='getattr':
                    if (rhs.value.name in saved_globals
                            or rhs.value.name in saved_getattrs):
                        saved_getattrs[lhs.name] = stmt
                        block_defs.add(lhs.name)
                        #continue
            elif isinstance(stmt, Parfor):
                pblocks = stmt.loop_body.copy()
                pblocks[-1] = stmt.init_block
                push_call_vars(pblocks, saved_globals, saved_getattrs)
                new_body.append(stmt)
                continue
            for v in stmt.list_vars():
                new_body += _get_saved_call_nodes(v.name, saved_globals,
                    saved_getattrs, block_defs)
            new_body.append(stmt)
        block.body = new_body

    return

def _get_saved_call_nodes(fname, saved_globals, saved_getattrs, block_defs):
    nodes = []
    while (fname not in block_defs and (fname in saved_globals
                or fname in saved_getattrs)):
        if fname in saved_globals:
            nodes.append(saved_globals[fname])
            block_defs.add(saved_globals[fname].target.name)
            fname = '_PA_DONE'
        elif fname in saved_getattrs:
            up_name = saved_getattrs[fname].value.value.name
            nodes.append(saved_getattrs[fname])
            block_defs.add(saved_getattrs[fname].target.name)
            fname = up_name
    nodes.reverse()
    return nodes

def fix_generator_types(generator_info, return_type, typemap):
    """postproc updates generator_info with live variables after transformations
    but generator variables have types in return_type that are updated here.
    """
    new_state_types = []
    for v in generator_info.state_vars:
        new_state_types.append(typemap[v])
    return_type.state_types = tuple(new_state_types)
    return
