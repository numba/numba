from __future__ import print_function, division, absolute_import

from numba import ir, ir_utils, types, rewrites, config
from numba.ir_utils import *
from numba.analysis import compute_cfg_from_blocks
from numba.controlflow import CFGraph
from numba.typing import npydecl
from numba.types.functions import Function
import numpy as np
import numba.parfor2
# circular dependency: import numba.npyufunc.dufunc.DUFunc

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


class Parfor2(ir.Expr, ir.Stmt):
    def __init__(self, loop_nests, init_block, loop_body, loc):
        super(Parfor2, self).__init__(
            op   = 'parfor2',
            loc  = loc
        )

        #self.input_info  = input_info
        #self.output_info = output_info
        self.loop_nests = loop_nests
        self.init_block = init_block
        self.loop_body = loop_body

    def __repr__(self):
        return repr(self.loop_nests) + repr(self.loop_body)

    def dump(self):
        for loopnest in self.loop_nests:
            print(loopnest)
        print("init block:")
        self.init_block.dump()
        for offset, block in sorted(self.loop_body.items()):
            print('label %s:' % (offset,))
            block.dump()


class ParforPass(object):
    """ParforPass class is responsible for converting Numpy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """
    def __init__(self, func_ir, typemap, calltypes, array_analysis):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.array_analysis = array_analysis
        ir_utils._max_label = max(func_ir.blocks.keys())

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible."""

        for (key, block) in self.func_ir.blocks.items():
            new_body = []
            for instr in block.body:
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    lhs = instr.target
                    if self._is_supported_npycall(expr):
                        instr = self._numpy_to_parfor(lhs, expr)
                    if isinstance(expr, ir.Expr) and expr.op == 'arrayexpr':
                        instr = self._arrayexpr_to_parfor(lhs, expr)
                new_body.append(instr)
            block.body = new_body

        return

    def _arrayexpr_to_parfor(self, lhs, arrayexpr):
        """generate parfor from arrayexpr node, which is essentially a
        map with recursive tree.
        """
        expr = arrayexpr.expr
        arr_typ = self.typemap[lhs.name]
        # TODO: support mutilple dimensions
        assert arr_typ.ndim==1
        el_typ = arr_typ.dtype
        corr = self.array_analysis.array_shape_classes[lhs.name][0]
        size_var = self.array_analysis.array_size_vars[lhs.name][0]
        scope = lhs.scope
        loc = lhs.loc
        index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
        self.typemap[index_var.name] = INT_TYPE
        loopnests = [ LoopNest(index_var, size_var, corr) ]
        init_block = ir.Block(scope, loc)
        parfor = Parfor2(loopnests, init_block, {}, loc)

        init_block.body = mk_alloc(self.typemap, self.calltypes, lhs,
            size_var, el_typ, scope, loc)
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var("$expr_out_var"), loc)
        self.typemap[expr_out_var.name] = el_typ
        body_block.body = _arrayexpr_tree_to_ir(self.typemap, self.calltypes,
            expr_out_var, expr, index_var)
        # lhs[parfor_index] = expr_out_var
        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        self.calltypes[setitem_node] = signature(types.none,
            self.typemap[lhs.name], INT_TYPE, el_typ)
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

    def _get_ndims(self, arr):
        return len(self.array_analysis.array_shape_classes[arr])

    def _numpy_to_parfor(self, lhs, expr):
        assert isinstance(expr, ir.Expr) and expr.op == 'call'
        call_name = self.array_analysis.numpy_calls[expr.func.name]
        args = expr.args
        if call_name=='dot':
            assert len(args)==2 #TODO: or len(args)==3
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
            self.typemap[index_var.name] = INT_TYPE
            loopnests = [ LoopNest(index_var, size_var, corr) ]
            init_block = ir.Block(scope, loc)
            parfor = Parfor2(loopnests, init_block, {}, loc)
            if self._get_ndims(in1.name)==2:
                # for 2D input, there is an inner loop
                # correlation of inner dimension
                inner_size_var = self.array_analysis.array_size_vars[in1.name][1]
                # loop structure: range block, header block, body

                range_label = next_label()
                header_label = next_label()
                body_label = next_label()
                out_label = next_label()

                alloc_nodes = mk_alloc(self.typemap, self.calltypes, lhs,
                    size_var, el_typ, scope, loc)
                init_block.body = alloc_nodes

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
                    self.typemap[lhs.name], INT_TYPE, el_typ)
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

def _mk_mvdot_body(typemap, calltypes, phi_b_var, index_var, in1, in2, sum_var, scope,
        loc, el_typ):
    body_block = ir.Block(scope, loc)
    # inner_index = phi_b_var
    inner_index = ir.Var(scope, mk_unique_var("$inner_index"), loc)
    typemap[inner_index.name] = INT_TYPE
    inner_index_assign = ir.Assign(phi_b_var, inner_index, loc)
    # tuple_var = build_tuple(index_var, inner_index)
    tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
    typemap[tuple_var.name] = types.containers.UniTuple(INT_TYPE, 2)
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
    calltypes[v_getitem_call] = signature(el_typ, typemap[in2.name], INT_TYPE)
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

def _arrayexpr_tree_to_ir(typemap, calltypes, expr_out_var, expr, parfor_index):
    """generate IR from array_expr's expr tree recursively. Assign output to
    expr_out_var and returne the whole IR as a list of Assign nodes.
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
                arg_out_var, arg, parfor_index)
            arg_vars.append(arg_out_var)
        if op in npydecl.supported_array_operators:
            if len(arg_vars)==2:
                ir_expr = ir.Expr.binop(op, arg_vars[0], arg_vars[1], loc)
                calltypes[ir_expr] = signature(el_typ, el_typ, el_typ)
            else:
                ir_expr = ir.Expr.unary(op, arg_vars[0], loc)
                calltypes[ir_expr] = signature(el_typ, el_typ)
            out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
        elif isinstance(op, np.ufunc):
            # elif isinstance(op, (np.ufunc, DUFunc)):
            # function calls are stored in variables which are not removed
            # op is typing_key to the variables type
            func_var = ir.Var(scope, _find_func_var(typemap, op), loc)
            ir_expr = ir.Expr.call(func_var, arg_vars, (), loc)
            calltypes[ir_expr] = typemap[func_var.name].get_call_type(
                typing.Context(), [el_typ], {})
            #signature(el_typ, el_typ)
            out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Var):
        if isinstance(typemap[expr.name], types.Array):
            # TODO: support multi-dimensional arrays
            assert typemap[expr.name].ndim==1
            ir_expr = ir.Expr.getitem(expr, parfor_index, loc)
            calltypes[ir_expr] = signature(el_typ, typemap[expr.name],
                INT_TYPE)
        else:
            assert typemap[expr.name]==el_typ
            ir_expr = expr
        out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Const):
        out_ir.append(ir.Assign(expr, expr_out_var, loc))

    if len(out_ir)==0:
        raise NotImplementedError(
            "Don't know how to translate array expression '%r'" % (expr,))
    return out_ir

def _find_func_var(typemap, func):
    """find variable in typemap which represents the function func.
    """
    for k,v in typemap.items():
        # Function types store actual functions in typing_key.
        if isinstance(v, Function) and v.typing_key==func:
            return k
    raise RuntimeError("ufunc call variable not found")

def lower_parfor2(func_ir, typemap, calltypes):
    """lower parfor to sequential or parallel Numba IR.
    """
    if "lower_parfor2_parallel" in dir(numba.parfor2):
        lower_parfor2_parallel(func_ir, typemap, calltypes)
    else:
        lower_parfor2_sequential(func_ir, typemap, calltypes)

def lower_parfor2_sequential(func_ir, typemap, calltypes):
    new_blocks = {}
    for (block_label, block) in func_ir.blocks.items():
        scope = block.scope
        i = _find_first_parfor(block.body)
        while i!=-1:
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
            if len(inst.loop_nests)>1:
                raise NotImplementedError("multi-dimensional parfor")
            loopnest = inst.loop_nests[0]
            # create range block for loop
            range_label = next_label()
            inst.init_block.body.append(ir.Jump(range_label, loc))
            header_label = next_label()
            range_block = mk_range_block(typemap, loopnest.range_variable,
                calltypes, scope, loc)
            range_block.body[-1].target = header_label # fix jump target
            phi_var = range_block.body[-2].target
            new_blocks[range_label] = range_block
            header_block = mk_loop_header(typemap, phi_var, calltypes,
                scope, loc)
            # first body block to jump to
            body_first_label = min(inst.loop_body.keys())
            header_block.body[-1].truebr = body_first_label
            header_block.body[-1].falsebr = block_label
            header_block.body[-2].target = loopnest.index_variable
            new_blocks[header_label] = header_block
            # last block jump to header
            body_last_label = max(inst.loop_body.keys())
            inst.loop_body[body_last_label].body.append(
                ir.Jump(header_label, loc))
            # add parfor body to blocks
            for (l, b) in inst.loop_body.items():
                new_blocks[l] = b
            i = _find_first_parfor(block.body)

        # old block stays either way
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    func_ir.blocks = _rename_labels(func_ir.blocks)
    if config.DEBUG_ARRAY_OPT==1:
        print("function after parfor lowering:")
        func_ir.dump()
    return

def _find_first_parfor(body):
    for (i, inst) in enumerate(body):
        if isinstance(inst, Parfor2):
            return i
    return -1

def _rename_labels(blocks):
    """rename labels of function body blocks according to topological sort.
    lowering requires this order.
    """
    cfg = compute_cfg_from_blocks(blocks)
    topo_order = cfg.topo_order()
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
