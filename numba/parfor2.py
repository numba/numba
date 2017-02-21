from __future__ import print_function, division, absolute_import

from numba import ir, ir_utils, types, rewrites
from numba.ir_utils import *

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


@rewrites.register_rewrite('after-inference')
class RewriteParfor2(rewrites.Rewrite):
    """The RewriteParforExtra class is responsible for converting Numpy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """
    def __init__(self, pipeline, *args, **kws):
        super(RewriteParfor2, self).__init__(pipeline, *args, **kws)
        self.array_analysis = pipeline.array_analysis
        ir_utils._max_label = max(pipeline.func_ir.blocks.keys())
        # Install a lowering hook if we are using this rewrite.
        special_ops = self.pipeline.targetctx.special_ops
        #if 'parfor2' not in special_ops:
        #    special_ops['parfor2'] = _lower_parfor2

    def match(self, interp, block, typemap, calltypes):
        """Match Numpy calls.
        Return True when one or more matches were found, False otherwise.
        """
        # We can trivially reject everything if there are no
        # calls in the type results.
        if len(calltypes) == 0:
            return False

        self.current_block = block
        self.typemap = typemap
        self.calltypes = calltypes

        assignments = block.find_insts(ir.Assign)
        for instr in assignments:
            expr = instr.value
            # is it a Numpy call?
            if self._is_supported_npycall(expr):
                return True
        return False

    def apply(self):
        """When we've found Numpy calls in a basic block, replace Numpy calls
        with Parfors when possible.
        """
        result = ir.Block(self.current_block.scope, self.current_block.loc)
        block = self.current_block
        for instr in block.body:
            if isinstance(instr, ir.Assign):
                expr = instr.value
                lhs = instr.target
                if self._is_supported_npycall(expr):
                    instr = self._numpy_to_parfor(lhs, expr)
            result.append(instr)
        return result

    def _is_supported_npycall(self, expr):
        """check if we support parfor translation for
        this Numpy call.
        """
        return False # turn off for now
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
            in1 = args[0].name
            in2 = args[1].name
            el_typ = self.typemap[lhs.name].dtype
            assert self._get_ndims(in1)<=2 and self._get_ndims(in2)==1
            # loop range correlation is same as first dimention of 1st input
            corr = self.array_analysis.array_shape_classes[in1][0]
            size_var = self.array_analysis.array_size_vars[in1][0]
            scope = self.current_block.scope
            loc = expr.loc
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), lhs.loc)
            self.typemap[index_var.name] = INT_TYPE
            loopnests = [ LoopNest(index_var, size_var, corr) ]
            init_block = ir.Block(scope, loc)
            parfor = Parfor2(loopnests, init_block, {}, loc)
            if self._get_ndims(in1)==2:
                # for 2D input, there is an inner loop
                # correlation of inner dimension
                inner_size_var = self.array_analysis.array_size_vars[in1][1]
                # loop structure: range block, header block, body

                range_label = next_label()
                header_label = next_label()
                body_label = next_label()
                out_label = next_label()

                # sum_var = 0
                const_node = ir.Const(0, loc)
                const_var = ir.Var(scope, mk_unique_var("$const"), loc)
                self.typemap[const_var.name] = el_typ
                const_assign = ir.Assign(const_node, const_var, loc)
                sum_var = ir.Var(scope, mk_unique_var("$sum_var"), loc)
                self.typemap[sum_var.name] = el_typ
                sum_assign = ir.Assign(const_var, sum_var, loc)
                alloc_nodes = mk_alloc(self.typemap, lhs, size_var, el_typ, scope,
                    loc)
                init_block.body = alloc_nodes + [const_assign, sum_assign]

                range_block = mk_range_block(self.typemap, inner_size_var,
                    self.calltypes, scope, loc)
                range_block.body[-1].target = header_label # fix jump target
                phi_var = range_block.body[-2].target

                header_block = mk_loop_header(self.typemap, phi_var,
                    self.calltypes, scope, loc)
                header_block.body[-1].truebr = body_label
                header_block.body[-1].falsebr = out_label
                phi_b_var = header_block.body[-2].target

                body_block = _mk_mvdot_body(self.typemap, phi_b_var, index_var,
                    in1, in2, sum_var, scope, loc, el_typ)
                body_block.body[-1].target = header_label

                out_block = ir.Block(scope, loc)
                # lhs[parfor_index] = sum_var
                setitem_node = ir.SetItem(lhs, index_var, sum_var, loc)
                out_block.body = [setitem_node]
                parfor.loop_body = {range_label:range_block,
                    header_label:header_block, body_label:body_block,
                    out_label:out_block}
            else: # self._get_ndims(in1)==1 (reduction)
                NotImplementedError("no reduction for dot() "+expr)
            parfor.dump()
            return parfor
        # return error if we couldn't handle it (avoid rewrite infinite loop)
        raise NotImplementedError("parfor translation failed for ", expr)

def _mk_mvdot_body(typemap, phi_b_var, index_var, in1, in2, sum_var, scope,
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
    X_val = ir.Var(scope, mk_unique_var("$"+in1+"_val"), loc)
    typemap[X_val.name] = el_typ
    getitem_call = ir.Expr.getitem(in1, tuple_var, loc)
    getitem_assign = ir.Assign(getitem_call, X_val, loc)
    # v_val = getitem(V, inner_index)
    v_val = ir.Var(scope, mk_unique_var("$"+in2+"_val"), loc)
    typemap[v_val.name] = el_typ
    v_getitem_call = ir.Expr.getitem(in2, inner_index, loc)
    v_getitem_assign = ir.Assign(v_getitem_call, v_val, loc)
    # add_var = X_val + v_val
    add_var = ir.Var(scope, mk_unique_var("$add_var"), loc)
    typemap[add_var.name] = el_typ
    add_call = ir.Expr.binop('+', X_val, v_val, loc)
    add_assign = ir.Assign(add_call, add_var, loc)
    # acc_var = sum_var + add_var
    acc_var = ir.Var(scope, mk_unique_var("$acc_var"), loc)
    typemap[acc_var.name] = el_typ
    acc_call = ir.Expr.inplace_binop('+=', '+', sum_var, add_var, loc)
    acc_assign = ir.Assign(acc_call, acc_var, loc)
    # sum_var = acc_var
    final_assign = ir.Assign(acc_var, sum_var, loc)
    # jump to header
    b_jump_header = ir.Jump(-1, loc)
    body_block.body = [inner_index_assign, tuple_assign,
        getitem_assign, v_getitem_assign, add_assign, acc_assign,
        final_assign, b_jump_header]
    return body_block

def lower_parfor2(func_ir, typemap, calltypes):
    """lower parfor to sequential or parallel Numba IR.
    """
    # TODO: lower to parallel
    new_blocks = {}
    for (block_label, block) in func_ir.blocks.items():
        scope = block.scope
        for (i, inst) in enumerate(block.body):
            if isinstance(inst, Parfor2):
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
                body_label = min(inst.loop_body.keys())
                header_block.body[-1].truebr = body_label
                header_block.body[-1].falsebr = block_label
                header_block.body[-2].target = loopnest.index_variable
                new_blocks[header_label] = header_block
                # add parfor body to blocks
                for (l, b) in inst.loop_body.items():
                    new_blocks[l] = b

        # old block stays either way
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    print("function after parfor lowering:")
    func_ir.dump()
    return None
