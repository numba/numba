#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

"""
This module transforms data-parallel operations such as Numpy calls into
'Parfor' nodes, which are nested loops that can be parallelized.
It also implements optimizations such as loop fusion, and extends the rest of
compiler analysis and optimizations to support Parfors.
This is similar to ParallelAccelerator package in Julia:
https://github.com/IntelLabs/ParallelAccelerator.jl
'Parallelizing Julia with a Non-invasive DSL', T. Anderson et al., ECOOP'17.
"""
from __future__ import print_function, division, absolute_import
import types as pytypes  # avoid confusion with numba.types
import sys

from numba import ir, ir_utils, types, typing, rewrites, config, analysis
from numba import array_analysis, postproc, typeinfer

from numba.ir_utils import (
    mk_unique_var,
    next_label,
    mk_alloc,
    get_np_ufunc_typ,
    mk_range_block,
    mk_loop_header,
    find_op_typ,
    get_name_var_table,
    replace_vars,
    visit_vars,
    visit_vars_inner,
    remove_dels,
    remove_dead,
    copy_propagate,
    get_block_copies,
    apply_copy_propagate,
    dprint_func_ir,
    find_topo_order,
    get_stmt_writes,
    rename_labels,
    get_call_table,
    simplify,
    simplify_CFG,
    has_no_side_effect,
    canonicalize_array_math,
    find_callname,
    guard,
    require)

from numba.analysis import (compute_use_defs, compute_live_map,
                            compute_dead_maps, compute_cfg_from_blocks)
from numba.controlflow import CFGraph
from numba.typing import npydecl, signature
from numba.types.functions import Function
from numba.array_analysis import (random_int_args, random_1arg_size,
                                  random_2arg_sizelast, random_3arg_sizelast,
                                  random_calls)
import copy
import numpy
# circular dependency: import numba.npyufunc.dufunc.DUFunc

sequential_parfor_lowering = False


class prange(object):

    def __new__(cls, *args):
        return range(*args)


_reduction_ops = {
    'sum': ('+=', '+', 0),
    'dot': ('+=', '+', 0),
    'prod': ('*=', '*', 1),
}


class LoopNest(object):

    '''The LoopNest class holds information of a single loop including
    the index variable (of a non-negative integer value), and the
    range variable, e.g. range(r) is 0 to r-1 with step size 1.
    '''

    def __init__(self, index_variable, start, stop, step):
        self.index_variable = index_variable
        self.start = start
        self.stop = stop
        self.step = step

    def __repr__(self):
        return ("LoopNest(index_variable={}, range={},{},{})".
                format(self.index_variable, self.start, self.stop, self.step))


class Parfor(ir.Expr, ir.Stmt):

    id_counter = 0

    def __init__(
            self,
            loop_nests,
            init_block,
            loop_body,
            loc,
            index_var,
            equiv_set):
        super(Parfor, self).__init__(
            op='parfor',
            loc=loc
        )

        self.id = type(self).id_counter
        type(self).id_counter += 1
        #self.input_info  = input_info
        #self.output_info = output_info
        self.loop_nests = loop_nests
        self.init_block = init_block
        self.loop_body = loop_body
        self.index_var = index_var
        self.params = None  # filled right before parallel lowering
        self.equiv_set = equiv_set

    def __repr__(self):
        return repr(self.loop_nests) + \
            repr(self.loop_body) + repr(self.index_var)

    def list_vars(self):
        """list variables used (read/written) in this parfor by
        traversing the body and combining block uses.
        """
        all_uses = []
        for l, b in self.loop_body.items():
            for stmt in b.body:
                all_uses += stmt.list_vars()

        for loop in self.loop_nests:
            all_uses.append(loop.index_variable)
            if isinstance(loop.start, ir.Var):
                all_uses.append(loop.start)
            if isinstance(loop.stop, ir.Var):
                all_uses.append(loop.stop)
            if isinstance(loop.step, ir.Var):
                all_uses.append(loop.step)

        for stmt in self.init_block.body:
            all_uses += stmt.list_vars()

        return all_uses

    def get_shape_classes(self, var):
        return self.equiv_set.get_shape_classes(var)

    def dump(self, file=None):
        file = file or sys.stdout
        print(("begin parfor {}".format(self.id)).center(20, '-'), file=file)
        print("index_var = ", self.index_var)
        for loopnest in self.loop_nests:
            print(loopnest, file=file)
        print("init block:", file=file)
        self.init_block.dump()
        for offset, block in sorted(self.loop_body.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file)
        print(("end parfor {}".format(self.id)).center(20, '-'), file=file)


class ParforPass(object):

    """ParforPass class is responsible for converting Numpy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """

    def __init__(self, func_ir, typemap, calltypes, return_type, typingctx):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.typingctx = typingctx
        self.return_type = return_type
        self.array_analysis = array_analysis.ArrayAnalysis(typingctx, func_ir, typemap,
                                                           calltypes)
        ir_utils._max_label = max(func_ir.blocks.keys())

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR."""
        self.func_ir.blocks = simplify_CFG(self.func_ir.blocks)
        # remove Del statements for easier optimization
        remove_dels(self.func_ir.blocks)
        # e.g. convert A.sum() to np.sum(A) for easier match and optimization
        canonicalize_array_math(self.func_ir, self.typemap,
                                self.calltypes, self.typingctx)
        self.array_analysis.run()
        self._convert_prange(self.func_ir.blocks)
        self._convert_numpy(self.func_ir.blocks)

        dprint_func_ir(self.func_ir, "after parfor pass")
        simplify(self.func_ir, self.typemap, self.calltypes)

        #dprint_func_ir(self.func_ir, "after remove_dead")
        # reorder statements to maximize fusion
        maximize_fusion(self.func_ir)
        dprint_func_ir(self.func_ir, "after maximize fusion")
        fuse_parfors(self.array_analysis, self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after fusion")
        # remove dead code after fusion to remove extra arrays and variables
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names, self.typemap)
        #dprint_func_ir(self.func_ir, "after second remove_dead")
        # push function call variables inside parfors so gufunc function
        # wouldn't need function variables as argument
        push_call_vars(self.func_ir.blocks, {}, {})
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names, self.typemap)
        dprint_func_ir(self.func_ir, "after optimization")
        if config.DEBUG_ARRAY_OPT == 1:
            print("variable types: ", sorted(self.typemap.items()))
            print("call types: ", self.calltypes)
        # run post processor again to generate Del nodes
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()
        if self.func_ir.is_generator:
            fix_generator_types(self.func_ir.generator_info, self.return_type,
                                self.typemap)
        if sequential_parfor_lowering:
            lower_parfor_sequential(
                self.typingctx, self.func_ir, self.typemap, self.calltypes)
        else:
            # prepare for parallel lowering
            # add parfor params to parfors here since lowering is destructive
            # changing the IR after this is not allowed
            get_parfor_params(self.func_ir.blocks)
        return

    def _convert_numpy(self, blocks):
        topo_order = find_topo_order(blocks)
        # variables available in the program so far (used for finding map
        # functions in array_expr lowering)
        avail_vars = []
        for label in topo_order:
            block = blocks[label]
            new_body = []
            equiv_set = self.array_analysis.get_equiv_set(label)
            for instr in block.body:
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    lhs = instr.target
                    if self._is_C_order(lhs.name):
                        # only translate C order since we can't allocate F
                        if guard(self._is_supported_npycall, expr):
                            instr = self._numpy_to_parfor(equiv_set, lhs, expr)
                        elif isinstance(expr, ir.Expr) and expr.op == 'arrayexpr':
                            instr = self._arrayexpr_to_parfor(
                                equiv_set, lhs, expr, avail_vars)
                    elif guard(self._is_supported_npyreduction, expr):
                        instr = self._reduction_to_parfor(equiv_set, lhs, expr)
                    avail_vars.append(lhs.name)
                new_body.append(instr)
            block.body = new_body

    def _convert_prange(self, blocks):
        call_table, _ = get_call_table(blocks)
        cfg = compute_cfg_from_blocks(blocks)
        for loop in cfg.loops().values():
            if len(loop.entries) != 1 or len(loop.exits) != 1:
                continue
            entry = list(loop.entries)[0]
            for inst in blocks[entry].body:
                # if prange call
                if (isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr)
                        and inst.value.op == 'call'
                        and self._is_prange(inst.value.func.name, call_table)):
                    body_labels = list(loop.body - {loop.header})
                    args = inst.value.args
                    # find loop index variable (pair_first in header block)
                    for stmt in blocks[loop.header].body:
                        if (isinstance(stmt, ir.Assign)
                                and isinstance(stmt.value, ir.Expr)
                                and stmt.value.op == 'pair_first'):
                            loop_index = stmt.target.name
                            break
                    # loop_index may be assigned to other vars
                    # get header copies to find all of them
                    cps, _ = get_block_copies({0: blocks[loop.header]},
                                              self.typemap)
                    cps = cps[0]
                    loop_index_vars = set(t for t, v in cps if v == loop_index)
                    loop_index_vars.add(loop_index)
                    start = 0
                    step = 1
                    size_var = args[0]
                    if len(args) == 2:
                        start = args[0]
                        size_var = args[1]
                    if len(args) == 3:
                        start = args[0]
                        size_var = args[1]
                        try:
                            step = self.func_ir.get_definition(args[2])
                        except KeyError:
                            raise NotImplementedError(
                                "Only known step size is supported for prange")
                        if not isinstance(step, ir.Const):
                            raise NotImplementedError(
                                "Only constant step size is supported for prange")
                        step = step.value
                        if step != 1:
                            raise NotImplementedError(
                                "Only constant step size of 1 is supported for prange")

                    # set l=l for dead remove
                    inst.value = inst.target
                    scope = blocks[entry].scope
                    loc = inst.loc
                    init_block = ir.Block(scope, loc)
                    body = {l: blocks[l] for l in body_labels}
                    index_var = ir.Var(
                        scope, mk_unique_var("parfor_index"), loc)
                    self.typemap[index_var.name] = types.intp
                    index_var_map = {v: index_var for v in loop_index_vars}
                    replace_vars(body, index_var_map)
                    # TODO: find correlation
                    parfor_loop = LoopNest(index_var, start, size_var, step)
                    parfor = Parfor([parfor_loop], init_block, body, loc, index_var,
                                    self.array_analysis.get_equiv_set(entry))
                    # add parfor to entry block, change jump target to exit
                    jump = blocks[entry].body.pop()
                    blocks[entry].body.append(parfor)
                    jump.target = list(loop.exits)[0]
                    blocks[entry].body.append(jump)
                    # remove jumps back to header block
                    for l in body_labels:
                        last_inst = body[l].body[-1]
                        if isinstance(
                                last_inst,
                                ir.Jump) and last_inst.target == loop.header:
                            body[l].body.pop()
                    # remove loop blocks from top level dict
                    blocks.pop(loop.header)
                    for l in body_labels:
                        blocks.pop(l)
                    # run on parfor body
                    parfor_blocks = wrap_parfor_blocks(parfor)
                    self._convert_prange(parfor_blocks)
                    self._convert_numpy(parfor_blocks)
                    unwrap_parfor_blocks(parfor, parfor_blocks)
                    # run convert again to handle other prange loops
                    return self._convert_prange(blocks)

    def _is_prange(self, func_var, call_table):
        # prange can be either getattr (numba.prange) or global (prange)
        if func_var not in call_table:
            return False
        call = call_table[func_var]
        return len(call) > 0 and (call[0] == 'prange' or call[0] == prange)

    def _is_C_order(self, arr_name):
        typ = self.typemap[arr_name]
        return isinstance(typ, types.npytypes.Array) and typ.layout == 'C' and typ.ndim > 0

    def _make_index_var(self, scope, index_vars, body_block):
        ndims = len(index_vars)
        loc = body_block.loc
        if ndims > 1:
            tuple_var = ir.Var(scope, mk_unique_var(
                "$parfor_index_tuple_var"), loc)
            self.typemap[tuple_var.name] = types.containers.UniTuple(
                types.intp, ndims)
            tuple_call = ir.Expr.build_tuple(list(index_vars), loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            body_block.body.append(tuple_assign)
            return tuple_var, types.containers.UniTuple(types.intp, ndims)
        elif ndims == 1:
            return index_vars[0], types.intp
        else:
            raise NotImplementedError(
                "Parfor does not handle arrays of dimension 0")

    def _arrayexpr_to_parfor(self, equiv_set, lhs, arrayexpr, avail_vars):
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
        index_vars = []
        size_vars = equiv_set.get_shape(lhs)
        for size_var in size_vars:
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
            index_vars.append(index_var)
            self.typemap[index_var.name] = types.intp
            loopnests.append(LoopNest(index_var, 0, size_var, 1))

        # generate init block and body
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(self.typemap, self.calltypes, lhs,
                                   tuple(size_vars), el_typ, scope, loc)
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var("$expr_out_var"), loc)
        self.typemap[expr_out_var.name] = el_typ

        index_var, index_var_typ = self._make_index_var(
            scope, index_vars, body_block)

        body_block.body.extend(
            _arrayexpr_tree_to_ir(
                self.func_ir,
                self.typemap,
                self.calltypes,
                equiv_set,
                expr_out_var,
                expr,
                index_var,
                index_vars,
                avail_vars))

        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set)

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        self.calltypes[setitem_node] = signature(
            types.none, self.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT == 1:
            parfor.dump()
        return parfor

    def _is_supported_npycall(self, expr):
        """check if we support parfor translation for
        this Numpy call.
        """
        call_name, mod_name = find_callname(self.func_ir, expr)
        supported_calls = ['zeros', 'ones'] + random_calls
        if call_name in supported_calls:
            return True
        # TODO: add more calls
        if call_name == 'dot':
            # only translate matrix/vector and vector/vector multiply to parfor
            # (don't translate matrix/matrix multiply)
            if (self._get_ndims(expr.args[0].name) <= 2 and
                    self._get_ndims(expr.args[1].name) == 1):
                return True
        return False

    def _is_supported_npyreduction(self, expr):
        """check if we support parfor translation for
        this Numpy reduce call.
        """
        func_name, mod_name = find_callname(self.func_ir, expr)
        return (func_name in _reduction_ops)

    def _get_ndims(self, arr):
        # return len(self.array_analysis.array_shape_classes[arr])
        return self.typemap[arr].ndim

    def _numpy_to_parfor(self, equiv_set, lhs, expr):
        call_name, mod_name = find_callname(self.func_ir, expr)
        args = expr.args
        kws = dict(expr.kws)
        if call_name in ['zeros', 'ones'] or call_name.startswith('random.'):
            return self._numpy_map_to_parfor(equiv_set, call_name, lhs, args, kws, expr)
        if call_name == 'dot':
            assert len(args) == 2 or len(args) == 3
            # if 3 args, output is allocated already
            out = None
            if len(args) == 3:
                out = args[2]
            if 'out' in kws:
                out = kws['out']

            in1 = args[0]
            in2 = args[1]
            el_typ = self.typemap[lhs.name].dtype
            assert self._get_ndims(
                in1.name) <= 2 and self._get_ndims(
                in2.name) == 1
            # loop range correlation is same as first dimention of 1st input
            size_vars = equiv_set.get_shape(in1)
            size_var = size_vars[0]
            scope = lhs.scope
            loc = expr.loc
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), lhs.loc)
            self.typemap[index_var.name] = types.intp
            loopnests = [LoopNest(index_var, 0, size_var, 1)]
            init_block = ir.Block(scope, loc)
            parfor = Parfor(loopnests, init_block, {}, loc, index_var,
                            equiv_set)

            if self._get_ndims(in1.name) == 2:
                # for 2D input, there is an inner loop
                # correlation of inner dimension
                inner_size_var = size_vars[1]
                # loop structure: range block, header block, body

                range_label = next_label()
                header_label = next_label()
                body_label = next_label()
                out_label = next_label()

                if out is None:
                    alloc_nodes = mk_alloc(self.typemap, self.calltypes, lhs,
                                           size_var, el_typ, scope, loc)
                    init_block.body = alloc_nodes
                else:
                    out_assign = ir.Assign(out, lhs, loc)
                    init_block.body = [out_assign]
                init_block.body.extend(
                    _gen_dotmv_check(
                        self.typemap,
                        self.calltypes,
                        in1,
                        in2,
                        lhs,
                        scope,
                        loc))
                # sum_var = 0
                const_node = ir.Const(0, loc)
                const_var = ir.Var(scope, mk_unique_var("$const"), loc)
                self.typemap[const_var.name] = el_typ
                const_assign = ir.Assign(const_node, const_var, loc)
                sum_var = ir.Var(scope, mk_unique_var("$sum_var"), loc)
                self.typemap[sum_var.name] = el_typ
                sum_assign = ir.Assign(const_var, sum_var, loc)

                range_block = mk_range_block(
                    self.typemap, 0, inner_size_var, 1, self.calltypes, scope, loc)
                range_block.body = [
                    const_assign, sum_assign] + range_block.body
                range_block.body[-1].target = header_label  # fix jump target
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
                self.calltypes[setitem_node] = signature(
                    types.none, self.typemap[lhs.name], types.intp, el_typ)
                out_block.body = [setitem_node]
                parfor.loop_body = {
                    range_label: range_block,
                    header_label: header_block,
                    body_label: body_block,
                    out_label: out_block}
            else:  # self._get_ndims(in1.name)==1 (reduction)
                NotImplementedError("no reduction for dot() " + expr)
            if config.DEBUG_ARRAY_OPT == 1:
                print("generated parfor for numpy call:")
                parfor.dump()
            return parfor
        # return error if we couldn't handle it (avoid rewrite infinite loop)
        raise NotImplementedError("parfor translation failed for ", expr)

    def _numpy_map_to_parfor(self, equiv_set, call_name, lhs, args, kws, expr):
        """generate parfor from Numpy calls that are maps.
        """
        scope = lhs.scope
        loc = lhs.loc
        arr_typ = self.typemap[lhs.name]
        el_typ = arr_typ.dtype

        # generate loopnests and size variables from lhs correlations
        loopnests = []
        index_vars = []
        size_vars = equiv_set.get_shape(lhs)
        for size_var in size_vars:
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
            index_vars.append(index_var)
            self.typemap[index_var.name] = types.intp
            loopnests.append(LoopNest(index_var, 0, size_var, 1))

        # generate init block and body
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(self.typemap, self.calltypes, lhs,
                                   tuple(size_vars), el_typ, scope, loc)
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var("$expr_out_var"), loc)
        self.typemap[expr_out_var.name] = el_typ

        index_var, index_var_typ = self._make_index_var(
            scope, index_vars, body_block)

        if call_name == 'zeros':
            value = ir.Const(0, loc)
        elif call_name == 'ones':
            value = ir.Const(1, loc)
        elif call_name.startswith('random.'):
            # remove size arg to reuse the call expr for single value
            _remove_size_arg(call_name, expr)
            # update expr type
            new_arg_typs, new_kw_types = _get_call_arg_types(
                expr, self.typemap)
            self.calltypes.pop(expr)
            self.calltypes[expr] = self.typemap[expr.func.name].get_call_type(
                typing.Context(), new_arg_typs, new_kw_types)
            value = expr
        else:
            NotImplementedError(
                "Map of numpy.{} to parfor is not implemented".format(call_name))

        value_assign = ir.Assign(value, expr_out_var, loc)
        body_block.body.append(value_assign)

        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set)

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        self.calltypes[setitem_node] = signature(
            types.none, self.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT == 1:
            print("generated parfor for numpy map:")
            parfor.dump()
        return parfor

    def _reduction_to_parfor(self, equiv_set, lhs, expr):
        call_name, mod_name = find_callname(self.func_ir, expr)
        args = expr.args
        kws = dict(expr.kws)
        if call_name in _reduction_ops:
            acc_op, im_op, init_val = _reduction_ops[call_name]
            assert len(args) in [1, 2]  # vector dot has 2 args
            in1 = args[0]
            arr_typ = self.typemap[in1.name]
            in_typ = arr_typ.dtype
            im_op_func_typ = find_op_typ(im_op, [in_typ, in_typ])
            el_typ = im_op_func_typ.return_type
            ndims = arr_typ.ndim

            # For full reduction, loop range correlation is same as 1st input
            sizes = equiv_set.get_shape(in1)
            assert ndims == len(sizes)
            scope = lhs.scope
            loc = expr.loc
            loopnests = []
            parfor_index = []
            for i in range(ndims):
                index_var = ir.Var(
                    scope, mk_unique_var(
                        "$parfor_index" + str(i)), loc)
                self.typemap[index_var.name] = types.intp
                parfor_index.append(index_var)
                loopnests.append(LoopNest(index_var, 0, sizes[i], 1))

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
            index_var, index_var_type = self._make_index_var(
                scope, parfor_index, acc_block)
            getitem_call = ir.Expr.getitem(in1, index_var, loc)
            self.calltypes[getitem_call] = signature(
                in_typ, arr_typ, index_var_type)
            acc_block.body.append(ir.Assign(getitem_call, tmp_var, loc))

            if call_name is 'dot':
                # dot has two inputs
                tmp_var1 = tmp_var
                in2 = args[1]
                tmp_var2 = ir.Var(scope, mk_unique_var("$val"), loc)
                self.typemap[tmp_var2.name] = in_typ
                getitem_call2 = ir.Expr.getitem(in2, index_var, loc)
                self.calltypes[getitem_call2] = signature(
                    in_typ, arr_typ, index_var_type)
                acc_block.body.append(ir.Assign(getitem_call2, tmp_var2, loc))
                mult_call = ir.Expr.binop('*', tmp_var1, tmp_var2, loc)
                mult_func_typ = find_op_typ('*', [in_typ, in_typ])
                self.calltypes[mult_call] = mult_func_typ
                tmp_var = ir.Var(scope, mk_unique_var("$val"), loc)
                acc_block.body.append(ir.Assign(mult_call, tmp_var, loc))

            acc_call = ir.Expr.inplace_binop(
                acc_op, im_op, acc_var, tmp_var, loc)
            # for some reason, type template of += returns None,
            # so type template of + should be used
            self.calltypes[acc_call] = im_op_func_typ
            # FIXME: we had to break assignment: acc += ... acc ...
            # into two assignment: acc_tmp = ... acc ...; x = acc_tmp
            # in order to avoid an issue in copy propagation.
            acc_tmp_var = ir.Var(scope, mk_unique_var("$acc"), loc)
            self.typemap[acc_tmp_var.name] = el_typ
            acc_block.body.append(ir.Assign(acc_call, acc_tmp_var, loc))
            acc_block.body.append(ir.Assign(acc_tmp_var, acc_var, loc))
            loop_body = {next_label(): acc_block}

            # parfor
            parfor = Parfor(loopnests, init_block, loop_body, loc, index_var,
                            equiv_set)
            return parfor
        # return error if we couldn't handle it (avoid rewrite infinite loop)
        raise NotImplementedError("parfor translation failed for ", expr)


def _remove_size_arg(call_name, expr):
    "remove size argument from args or kws"
    # remove size kwarg
    kws = dict(expr.kws)
    kws.pop('size', '')
    expr.kws = tuple(kws.items())

    # remove size arg if available
    if call_name in random_1arg_size + random_int_args:
        # these calls have only a "size" argument or list of ints
        # so remove all args
        expr.args = []

    if call_name in random_3arg_sizelast:
        # normal, uniform, ... have 3 args, last one is size
        if len(expr.args) == 3:
            expr.args.pop()

    if call_name in random_2arg_sizelast:
        # have 2 args, last one is size
        if len(expr.args) == 2:
            expr.args.pop()

    if call_name == 'random.randint':
        # has 4 args, 3rd one is size
        if len(expr.args) == 3:
            expr.args.pop()
        if len(expr.args) == 4:
            dt_arg = expr.args.pop()
            expr.args.pop()  # remove size
            expr.args.append(dt_arg)

    if call_name == 'random.triangular':
        # has 4 args, last one is size
        if len(expr.args) == 4:
            expr.args.pop()

    return


def _get_call_arg_types(expr, typemap):
    new_arg_typs = []
    for arg in expr.args:
        new_arg_typs.append(typemap[arg.name])

    new_kw_types = {}
    for name, arg in expr.kws:
        new_kw_types[name] = typemap[arg.name]

    return tuple(new_arg_typs), new_kw_types


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
    call_node = ir.Expr.call(g_var, [in1, in2, out], (), loc)
    calltypes[call_node] = func_typ.get_call_type(
        typing.Context(), [typemap[in1.name], typemap[in2.name], typemap[out.name]], {})
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
    typemap[inner_index.name] = types.intp
    inner_index_assign = ir.Assign(phi_b_var, inner_index, loc)
    # tuple_var = build_tuple(index_var, inner_index)
    tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
    typemap[tuple_var.name] = types.containers.UniTuple(types.intp, 2)
    tuple_call = ir.Expr.build_tuple([index_var, inner_index], loc)
    tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
    # X_val = getitem(X, tuple_var)
    X_val = ir.Var(scope, mk_unique_var("$" + in1.name + "_val"), loc)
    typemap[X_val.name] = el_typ
    getitem_call = ir.Expr.getitem(in1, tuple_var, loc)
    calltypes[getitem_call] = signature(el_typ, typemap[in1.name],
                                        typemap[tuple_var.name])
    getitem_assign = ir.Assign(getitem_call, X_val, loc)
    # v_val = getitem(V, inner_index)
    v_val = ir.Var(scope, mk_unique_var("$" + in2.name + "_val"), loc)
    typemap[v_val.name] = el_typ
    v_getitem_call = ir.Expr.getitem(in2, inner_index, loc)
    calltypes[v_getitem_call] = signature(
        el_typ, typemap[in2.name], types.intp)
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
    body_block.body = [
        inner_index_assign,
        tuple_assign,
        getitem_assign,
        v_getitem_assign,
        add_assign,
        acc_assign,
        final_assign,
        b_jump_header]
    return body_block


def _arrayexpr_tree_to_ir(
        func_ir,
        typemap,
        calltypes,
        equiv_set,
        expr_out_var,
        expr,
        parfor_index_tuple_var,
        all_parfor_indices,
        avail_vars):
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
            out_ir += _arrayexpr_tree_to_ir(func_ir,
                                            typemap,
                                            calltypes,
                                            equiv_set,
                                            arg_out_var,
                                            arg,
                                            parfor_index_tuple_var,
                                            all_parfor_indices,
                                            avail_vars)
            arg_vars.append(arg_out_var)
        if op in npydecl.supported_array_operators:
            el_typ1 = typemap[arg_vars[0].name]
            if len(arg_vars) == 2:
                el_typ2 = typemap[arg_vars[1].name]
                func_typ = find_op_typ(op, [el_typ1, el_typ2])
                ir_expr = ir.Expr.binop(op, arg_vars[0], arg_vars[1], loc)
                if op == '/':
                    func_typ, ir_expr = _gen_np_divide(
                        arg_vars[0], arg_vars[1], out_ir, typemap)
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
                func_var = ir.Var(
                    scope, _find_func_var(
                        typemap, op, avail_vars), loc)
                func_var_def = func_ir.get_definition(func_var.name)
                ir_expr = ir.Expr.call(func_var, arg_vars, (), loc)
                call_typ = typemap[func_var.name].get_call_type(
                    typing.Context(), [el_typ] * len(arg_vars), {})
                calltypes[ir_expr] = call_typ
                el_typ = call_typ.return_type
                #signature(el_typ, el_typ)
                out_ir.append(ir.Assign(func_var_def, func_var, loc))
                out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Var):
        var_typ = typemap[expr.name]
        if isinstance(var_typ, types.Array):
            el_typ = var_typ.dtype
            ir_expr = _gen_arrayexpr_getitem(
                equiv_set,
                expr,
                parfor_index_tuple_var,
                all_parfor_indices,
                el_typ,
                calltypes,
                typemap,
                out_ir)
        else:
            # assert typemap[expr.name]==el_typ
            el_typ = var_typ
            ir_expr = expr
        out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Const):
        el_typ = typing.Context().resolve_value_type(expr.value)
        out_ir.append(ir.Assign(expr, expr_out_var, loc))

    if len(out_ir) == 0:
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
    func_var_typ = get_np_ufunc_typ(numpy.divide)
    typemap[attr_var.name] = func_var_typ
    attr_assign = ir.Assign(div_attr_call, attr_var, loc)
    # divide call:  div_attr(arg1, arg2)
    div_call = ir.Expr.call(attr_var, [arg1, arg2], (), loc)
    func_typ = func_var_typ.get_call_type(
        typing.Context(), [typemap[arg1.name], typemap[arg2.name]], {})
    out_ir.extend([g_np_assign, attr_assign])
    return func_typ, div_call


def _gen_arrayexpr_getitem(
        equiv_set,
        var,
        parfor_index_tuple_var,
        all_parfor_indices,
        el_typ,
        calltypes,
        typemap,
        out_ir):
    """if there is implicit dimension broadcast, generate proper access variable
    for getitem. For example, if indices are (i1,i2,i3) but shape is (c1,0,c3),
    generate a tuple with (i1,0,i3) for access.  Another example: for (i1,i2,i3)
    and (c1,c2) generate (i2,i3).
    """
    loc = var.loc
    index_var = parfor_index_tuple_var
    ndims = typemap[var.name].ndim
    num_indices = len(all_parfor_indices)
    size_vars = equiv_set.get_shape(var)
    size_consts = [equiv_set.get_equiv_const(x) for x in size_vars]
    if ndims == 1:
        # Use last index for 1D arrays
        index_var = all_parfor_indices[-1]
    elif any([x != None for x in size_consts]):
        # Need a tuple as index
        ind_offset = num_indices - ndims
        tuple_var = ir.Var(var.scope, mk_unique_var(
            "$parfor_index_tuple_var_bcast"), loc)
        typemap[tuple_var.name] = types.containers.UniTuple(types.intp, ndims)
        # Just in case, const var for size 1 dim access index: $const0 =
        # Const(0)
        const_node = ir.Const(0, var.loc)
        const_var = ir.Var(var.scope, mk_unique_var("$const_ind_0"), loc)
        typemap[const_var.name] = types.intp
        const_assign = ir.Assign(const_node, const_var, loc)
        out_ir.append(const_assign)
        index_vars = []
        for i in reversed(range(ndims)):
            size_var = size_vars[i]
            size_const = size_consts[i]
            if size_const == 1:
                index_vars.append(const_var)
            else:
                index_vars.append(all_parfor_indices[ind_offset + i])
        index_vars = list(reversed(index_vars))
        tuple_call = ir.Expr.build_tuple(index_vars, loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out_ir.append(tuple_assign)
        index_var = tuple_var

    ir_expr = ir.Expr.getitem(var, index_var, loc)
    calltypes[ir_expr] = signature(el_typ, typemap[var.name],
                                   typemap[index_var.name])
    return ir_expr


def _find_func_var(typemap, func, avail_vars):
    """find variable in typemap which represents the function func.
    """
    for v in avail_vars:
        t = typemap[v]
        # Function types store actual functions in typing_key.
        if isinstance(t, Function) and t.typing_key == func:
            return v
    raise RuntimeError("ufunc call variable not found")


def lower_parfor_sequential(typingctx, func_ir, typemap, calltypes):
    ir_utils._max_label = ir_utils.find_max_label(func_ir.blocks) + 1

    parfor_found = False
    new_blocks = {}
    for (block_label, block) in func_ir.blocks.items():
        block_label, parfor_found = _lower_parfor_sequential_block(
            block_label, block, new_blocks, typemap, calltypes, parfor_found)
        # old block stays either way
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    # rename only if parfor found and replaced (avoid test_flow_control error)
    if parfor_found:
        func_ir.blocks = rename_labels(func_ir.blocks)
    dprint_func_ir(func_ir, "after parfor sequential lowering")
    simplify(func_ir, typemap, calltypes)
    dprint_func_ir(func_ir, "after parfor sequential simplify")

    return


def _lower_parfor_sequential_block(
        block_label,
        block,
        new_blocks,
        typemap,
        calltypes,
        parfor_found):
    scope = block.scope
    i = _find_first_parfor(block.body)
    while i != -1:
        parfor_found = True
        inst = block.body[i]
        loc = inst.init_block.loc
        # split block across parfor
        prev_block = ir.Block(scope, loc)
        prev_block.body = block.body[:i]
        block.body = block.body[i + 1:]
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
            range_block = mk_range_block(
                typemap,
                loopnest.start,
                loopnest.stop,
                loopnest.step,
                calltypes,
                scope,
                loc)
            range_block.body[-1].target = header_label  # fix jump target
            phi_var = range_block.body[-2].target
            new_blocks[range_label] = range_block
            header_block = mk_loop_header(typemap, phi_var, calltypes,
                                          scope, loc)
            header_block.body[-2].target = loopnest.index_variable
            new_blocks[header_label] = header_block
            # jump to this new inner loop
            if i == 0:
                inst.init_block.body.append(ir.Jump(range_label, loc))
                header_block.body[-1].falsebr = block_label
            else:
                new_blocks[prev_header_label].body[-1].truebr = range_label
                header_block.body[-1].falsebr = prev_header_label
            prev_header_label = header_label  # to set truebr next loop

        # last body block jump to inner most header
        body_last_label = max(inst.loop_body.keys())
        inst.loop_body[body_last_label].body.append(
            ir.Jump(header_label, loc))
        # inner most header jumps to first body block
        body_first_label = min(inst.loop_body.keys())
        header_block.body[-1].truebr = body_first_label
        # add parfor body to blocks
        for (l, b) in inst.loop_body.items():
            l, parfor_found = _lower_parfor_sequential_block(
                l, b, new_blocks, typemap, calltypes, parfor_found)
            new_blocks[l] = b
        i = _find_first_parfor(block.body)
    return block_label, parfor_found


def _find_first_parfor(body):
    for (i, inst) in enumerate(body):
        if isinstance(inst, Parfor):
            return i
    return -1


def get_parfor_params(blocks):
    """find variables used in body of parfors from outside and save them.
    computed as live variables at entry of first block.
    """

    # since parfor wrap creates a back-edge to first non-init basic block,
    # live_map[first_non_init_block] contains variables defined in parfor body
    # that could be undefined before. So we only consider variables that are
    # actually defined before the parfor body in the program.
    pre_defs = set()
    _, all_defs = compute_use_defs(blocks)
    topo_order = find_topo_order(blocks)
    for label in topo_order:
        block = blocks[label]
        for i, parfor in _find_parfors(block.body):
            # find variable defs before the parfor in the same block
            dummy_block = ir.Block(block.scope, block.loc)
            dummy_block.body = block.body[:i]
            before_defs = compute_use_defs({0: dummy_block}).defmap[0]
            pre_defs |= before_defs
            parfor.params = get_parfor_params_inner(parfor, pre_defs)

        pre_defs |= all_defs[label]
    return


def get_parfor_params_inner(parfor, pre_defs):

    blocks = wrap_parfor_blocks(parfor)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    unwrap_parfor_blocks(parfor)
    keylist = sorted(live_map.keys())
    init_block = keylist[0]
    first_non_init_block = keylist[1]

    before_defs = usedefs.defmap[init_block] | pre_defs
    params = live_map[first_non_init_block] & before_defs
    return params


def _find_parfors(body):
    for i, inst in enumerate(body):
        if isinstance(inst, Parfor):
            yield i, inst


def get_parfor_outputs(parfor, parfor_params):
    """get arrays that are written to inside the parfor and need to be passed
    as parameters to gufunc.
    """
    # FIXME: The following assumes the target of all SetItem are outputs,
    # which is wrong!
    last_label = max(parfor.loop_body.keys())
    outputs = []
    for blk in parfor.loop_body.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.SetItem):
                if stmt.index.name == parfor.index_var.name:
                    outputs.append(stmt.target.name)
    # make sure these written arrays are in parfor parameters (live coming in)
    outputs = list(set(outputs) & set(parfor_params))
    return sorted(outputs)


def get_parfor_reductions(parfor, parfor_params, reductions=None, names=None):
    """get variables that are accumulated using inplace_binop inside the parfor
    and need to be passed as reduction parameters to gufunc.
    """
    if reductions is None:
        reductions = {}
    if names is None:
        names = []
    last_label = max(parfor.loop_body.keys())

    for blk in parfor.loop_body.values():
        for stmt in blk.body:
            if (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "inplace_binop"):
                name = stmt.value.lhs.name
                if name in parfor_params:
                    names.append(name)
                    red_info = None
                    for (acc_op, imm_op, init_val) in _reduction_ops.values():
                        if imm_op == stmt.value.immutable_fn:
                            red_info = (
                                stmt.value.fn, stmt.value.immutable_fn, init_val)
                            break
                    if red_info is None:
                        raise NotImplementedError(
                            "Reduction is not support for inplace operator %s" %
                            stmt.value.fn)
                    reductions[name] = red_info
            if isinstance(stmt, Parfor):
                # recursive parfors can have reductions like test_prange8
                get_parfor_reductions(stmt, parfor_params, reductions, names)
    return names, reductions


def visit_vars_parfor(parfor, callback, cbdata):
    if config.DEBUG_ARRAY_OPT == 1:
        print("visiting parfor vars for:", parfor)
        print("cbdata: ", sorted(cbdata.items()))
    for l in parfor.loop_nests:
        l.index_variable = visit_vars_inner(l.index_variable, callback, cbdata)
        if isinstance(l.start, ir.Var):
            l.start = visit_vars_inner(l.start, callback, cbdata)
        if isinstance(l.stop, ir.Var):
            l.stop = visit_vars_inner(l.stop, callback, cbdata)
        if isinstance(l.step, ir.Var):
            l.step = visit_vars_inner(l.step, callback, cbdata)
    visit_vars({-1: parfor.init_block}, callback, cbdata)
    visit_vars(parfor.loop_body, callback, cbdata)
    return


# add call to visit parfor variable
ir_utils.visit_vars_extensions[Parfor] = visit_vars_parfor


def parfor_defs(parfor, use_set=None, def_set=None):
    """list variables written in this parfor by recursively
    calling compute_use_defs() on body and combining block defs.
    """
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
    blocks = wrap_parfor_blocks(parfor)
    uses, defs = compute_use_defs(blocks)
    cfg = compute_cfg_from_blocks(blocks)
    last_label = max(blocks.keys())
    unwrap_parfor_blocks(parfor)

    # Conservatively, only add defs for blocks that are definitely executed
    # Go through blocks in order, as if they are statements of the block that
    # includes the parfor, and update uses/defs.

    # no need for topo order of ir_utils
    topo_order = cfg.topo_order()
    # blocks that dominate last block are definitely executed
    definitely_executed = cfg.dominators()[last_label]
    # except loop bodies that might not execute
    for loop in cfg.loops().values():
        definitely_executed -= loop.body
    for label in topo_order:
        if label in definitely_executed:
            # see compute_use_defs() in analysis.py
            # variables defined in the block that includes the parfor are not
            # uses of that block (are not potentially live in the beginning of
            # the block)
            use_set.update(uses[label] - def_set)
            def_set.update(defs[label])
        else:
            use_set.update(uses[label] - def_set)

    # treat loop variables and size variables as use
    loop_vars = {
        l.start.name for l in parfor.loop_nests if isinstance(
            l.start, ir.Var)}
    loop_vars |= {
        l.stop.name for l in parfor.loop_nests if isinstance(
            l.stop, ir.Var)}
    loop_vars |= {
        l.step.name for l in parfor.loop_nests if isinstance(
            l.step, ir.Var)}
    use_set.update(loop_vars)

    return analysis._use_defs_result(usemap=use_set, defmap=def_set)


analysis.ir_extension_usedefs[Parfor] = parfor_defs


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
    loop_vars = {
        l.start.name for l in parfor.loop_nests if isinstance(
            l.start, ir.Var)}
    loop_vars |= {
        l.stop.name for l in parfor.loop_nests if isinstance(
            l.stop, ir.Var)}
    loop_vars |= {
        l.step.name for l in parfor.loop_nests if isinstance(
            l.step, ir.Var)}
    loop_vars |= {l.index_variable.name for l in parfor.loop_nests}
    # for var_list in parfor.array_analysis.array_size_vars.values():
    #    loop_vars |= {v.name for v in var_list if isinstance(v, ir.Var)}

    dead_set = set()
    for label in blocks.keys():
        # only kill vars that are actually dead at the parfor's block
        dead_map.internal[label] &= curr_dead_set
        dead_map.internal[label] -= loop_vars
        dead_set |= dead_map.internal[label]
        dead_map.escaping[label] &= curr_dead_set
        dead_map.escaping[label] -= loop_vars
        dead_set |= dead_map.escaping[label]

    # dummy class to replace func_ir. _patch_var_dels only accesses blocks
    class DummyFuncIR(object):

        def __init__(self, blocks):
            self.blocks = blocks
    post_proc = postproc.PostProcessor(DummyFuncIR(blocks))
    post_proc._patch_var_dels(dead_map.internal, dead_map.escaping)
    unwrap_parfor_blocks(parfor)

    return dead_set | loop_vars


postproc.ir_extension_insert_dels[Parfor] = parfor_insert_dels

# reorder statements to maximize fusion


def maximize_fusion(func_ir):
    blocks = func_ir.blocks
    call_table, _ = get_call_table(blocks)
    for block in blocks.values():
        order_changed = True
        while order_changed:
            order_changed = False
            i = 0
            while i < len(block.body) - 2:
                stmt = block.body[i]
                next_stmt = block.body[i + 1]
                # swap only parfors with non-parfors
                # don't reorder calls with side effects (e.g. file close)
                # only read-read dependencies are OK
                # make sure there is no write-write, write-read dependencies
                if (isinstance(
                        stmt, Parfor) and not isinstance(
                        next_stmt, Parfor)
                        and (not isinstance(next_stmt, ir.Assign)
                             or has_no_side_effect(
                            next_stmt.value, set(), call_table)
                        or guard(is_assert_equiv, func_ir, next_stmt.value))):
                    stmt_accesses = {v.name for v in stmt.list_vars()}
                    stmt_writes = get_parfor_writes(stmt)
                    next_accesses = {v.name for v in next_stmt.list_vars()}
                    next_writes = get_stmt_writes(next_stmt)
                    if len((stmt_writes & next_accesses)
                            | (next_writes & stmt_accesses)) == 0:
                        block.body[i] = next_stmt
                        block.body[i + 1] = stmt
                        order_changed = True
                i += 1
    return


def is_assert_equiv(func_ir, expr):
    func_name, mod_name = find_callname(func_ir, expr)
    return func_name == 'assert_equiv'


def get_parfor_writes(parfor):
    assert isinstance(parfor, Parfor)
    writes = set()
    blocks = parfor.loop_body.copy()
    blocks[-1] = parfor.init_block
    for block in blocks.values():
        for stmt in block.body:
            writes.update(get_stmt_writes(stmt))
            if isinstance(stmt, Parfor):
                writes.update(get_parfor_writes(stmt))
    return writes


def fuse_parfors(array_analysis, blocks):
    for label, block in blocks.items():
        equiv_set = array_analysis.get_equiv_set(label)
        fusion_happened = True
        while fusion_happened:
            fusion_happened = False
            new_body = []
            i = 0
            while i < len(block.body) - 1:
                stmt = block.body[i]
                next_stmt = block.body[i + 1]
                if isinstance(stmt, Parfor) and isinstance(next_stmt, Parfor):
                    fused_node = try_fuse(equiv_set, stmt, next_stmt)
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


def try_fuse(equiv_set, parfor1, parfor2):
    """try to fuse parfors and return a fused parfor, otherwise return None
    """
    dprint("try_fuse trying to fuse \n", parfor1, "\n", parfor2)

    # fusion of parfors with different dimensions not supported yet
    if len(parfor1.loop_nests) != len(parfor2.loop_nests):
        dprint("try_fuse parfors number of dimensions mismatch")
        return None

    ndims = len(parfor1.loop_nests)
    # all loops should be equal length

    def is_equiv(x, y):
        return x == y or equiv_set.is_equiv(x, y)

    for i in range(ndims):
        nest1 = parfor1.loop_nests[i]
        nest2 = parfor2.loop_nests[i]
        if not (is_equiv(nest1.start, nest2.start) and
                is_equiv(nest1.stop, nest2.stop) and
                is_equiv(nest1.step, nest2.step)):
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
    init2_uses = compute_use_defs({0: parfor2.init_block}).usemap[0]
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
    index_dict = {parfor2.index_var.name: parfor1.index_var}
    for i in range(ndims):
        index_dict[parfor2.loop_nests[i].index_variable.name] = parfor1.loop_nests[
            i].index_variable
    replace_vars(parfor1.loop_body, index_dict)
    nameset = set(x.name for x in index_dict.values())
    remove_duplicate_definitions(parfor1.loop_body, nameset)
    remove_empty_block(parfor1.loop_body)

    return parfor1


def remove_duplicate_definitions(blocks, nameset):
    """Remove duplicated definition for variables in the given nameset, which
    is often a result of parfor fusion.
    """
    for label, block in blocks.items():
        body = block.body
        new_body = []
        defined = set()
        for inst in body:
            if isinstance(inst, ir.Assign):
                name = inst.target.name
                if name in nameset:
                    if name in defined:
                        continue
                    defined.add(name)
            new_body.append(inst)
        block.body = new_body
    return


def remove_empty_block(blocks):
    """Remove empty blocks and any jumps to them, which can be a result
    from prange conversion and/or fusion.
    """
    emptyset = set()
    for label, block in blocks.items():
        if len(block.body) == 0:
            emptyset.add(label)
    for label in emptyset:
        blocks.pop(label)
    for label, block in blocks.items():
        inst = block.body[-1]
        if isinstance(inst, ir.Jump) and inst.target in emptyset:
            block.body.pop()


def has_cross_iter_dep(parfor):
    # we consevatively assume there is cross iteration dependency when
    # the parfor index is used in any expression since the expression could
    # be used for indexing arrays
    # TODO: make it more accurate using ud-chains
    indices = {l.index_variable for l in parfor.loop_nests}
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
    if config.DEBUG_ARRAY_OPT == 1:
        print(*s)


def remove_dead_parfor(parfor, lives, arg_aliases, alias_map, typemap):
    # remove dead get/sets in last block
    # FIXME: I think that "in the last block" is not sufficient in general.  We might need to
    # remove from any block.
    last_label = max(parfor.loop_body.keys())
    last_block = parfor.loop_body[last_label]

    # save array values set to replace getitems
    saved_values = {}
    new_body = []
    for stmt in last_block.body:
        if (isinstance(stmt, ir.SetItem) and stmt.index.name ==
                parfor.index_var.name and stmt.target.name not in lives):
            saved_values[stmt.target.name] = stmt.value
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
            rhs = stmt.value
            if rhs.op == 'getitem' and rhs.index.name == parfor.index_var.name:
                # replace getitem if value saved
                stmt.value = saved_values.get(rhs.value.name, rhs)
        new_body.append(stmt)
    last_block.body = new_body

    alias_set = set(alias_map.keys())
    # after getitem replacement, remove extra setitems
    new_body = []
    in_lives = copy.copy(lives)
    for stmt in reversed(last_block.body):
        # aliases of lives are also live for setitems
        alias_lives = in_lives & alias_set
        for v in alias_lives:
            in_lives |= alias_map[v]
        if (isinstance(stmt, ir.SetItem) and stmt.index.name ==
                parfor.index_var.name and stmt.target.name not in in_lives):
            continue
        in_lives |= {v.name for v in stmt.list_vars()}
        new_body.append(stmt)
    new_body.reverse()
    last_block.body = new_body

    # process parfor body recursively
    remove_dead_parfor_recursive(
        parfor, lives, arg_aliases, alias_map, typemap)
    return


ir_utils.remove_dead_extensions[Parfor] = remove_dead_parfor


def remove_dead_parfor_recursive(parfor, lives, arg_aliases, alias_map, typemap):
    """create a dummy function from parfor and call remove dead recursively
    """
    blocks = parfor.loop_body.copy()  # shallow copy is enough
    first_body_block = min(blocks.keys())
    assert first_body_block > 0  # we are using 0 for init block here
    last_label = max(blocks.keys())
    return_label = last_label + 1

    loc = blocks[last_label].loc
    scope = blocks[last_label].scope
    blocks[return_label] = ir.Block(scope, loc)

    # add dummy jump in init_block for CFG to work
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(first_body_block, loc))

    # add lives in a dummpy return to last block to avoid their removal
    tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
    # dummy type for tuple_var
    typemap[tuple_var.name] = types.containers.UniTuple(
        types.intp, 2)
    live_vars = [ir.Var(scope, v, loc) for v in lives]
    tuple_call = ir.Expr.build_tuple(live_vars, loc)
    blocks[return_label].body.append(ir.Assign(tuple_call, tuple_var, loc))
    blocks[return_label].body.append(ir.Return(tuple_var, loc))

    branch = ir.Branch(0, first_body_block, return_label, loc)
    blocks[last_label].body.append(branch)

    # args var including aliases is ok
    remove_dead(blocks, arg_aliases, typemap, alias_map, arg_aliases)
    typemap.pop(tuple_var.name)  # remove dummy tuple type
    blocks[0].body.pop()  # remove dummy jump
    blocks[last_label].body.pop()  # remove branch
    return


def find_potential_aliases_parfor(parfor, args, typemap, alias_map, arg_aliases):
    blocks = wrap_parfor_blocks(parfor)
    ir_utils.find_potential_aliases(
        blocks, args, typemap, alias_map, arg_aliases)
    unwrap_parfor_blocks(parfor)
    return

ir_utils.alias_analysis_extensions[Parfor] = find_potential_aliases_parfor


def wrap_parfor_blocks(parfor):
    """wrap parfor blocks for analysis/optimization like CFG"""
    blocks = parfor.loop_body.copy()  # shallow copy is enough
    first_body_block = min(blocks.keys())
    assert first_body_block > 0  # we are using 0 for init block here
    last_label = max(blocks.keys())
    loc = blocks[last_label].loc

    # add dummy jump in init_block for CFG to work
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(first_body_block, loc))
    blocks[last_label].body.append(ir.Jump(first_body_block, loc))
    return blocks


def unwrap_parfor_blocks(parfor, blocks=None):
    """
    unwrap parfor blocks after analysis/optimization.
    Allows changes to the parfor loop.
    """
    if blocks is not None:
        # make sure init block isn't removed
        assert 0 in blocks
        # update loop body blocks
        blocks.pop(0)
        parfor.loop_body = blocks

    # make sure dummy jump to loop body isn't altered
    first_body_label = min(parfor.loop_body.keys())
    assert isinstance(parfor.init_block.body[-1], ir.Jump)
    assert parfor.init_block.body[-1].target == first_body_label

    # remove dummy jump to loop body
    parfor.init_block.body.pop()

    # make sure dummy jump back to loop body isn't altered
    last_label = max(parfor.loop_body.keys())
    assert isinstance(parfor.loop_body[last_label].body[-1], ir.Jump)
    assert parfor.loop_body[last_label].body[-1].target == first_body_label
    # remove dummy jump back to loop
    parfor.loop_body[last_label].body.pop()
    return


def get_copies_parfor(parfor, typemap):
    """find copies generated/killed by parfor"""
    blocks = wrap_parfor_blocks(parfor)
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks, typemap)
    in_gen_copies, in_extra_kill = get_block_copies(blocks, typemap)
    unwrap_parfor_blocks(parfor)

    # parfor's extra kill is kills of its init block,
    # and all possible gens and kills of it's body loop.
    # body doesn't gen and only kills since it may or may not run
    # TODO: save copies that are repeated in parfor
    kill_set = in_extra_kill[0]
    for label in parfor.loop_body.keys():
        kill_set |= {l for l, r in in_gen_copies[label]}
        kill_set |= in_extra_kill[label]

    # gen copies is copies generated by init that are not killed by body
    last_label = max(parfor.loop_body.keys())
    gens = out_copies_parfor[last_label] & in_gen_copies[0]

    if config.DEBUG_ARRAY_OPT == 1:
        print("copy propagate parfor gens:", gens, "kill_set", kill_set)
    return gens, kill_set


ir_utils.copy_propagate_extensions[Parfor] = get_copies_parfor


def apply_copies_parfor(parfor, var_dict, name_var_table, ext_func, ext_data,
                        typemap, calltypes):
    """apply copy propagate recursively in parfor"""
    blocks = wrap_parfor_blocks(parfor)
    # add dummy assigns for each copy
    assign_list = []
    for lhs_name, rhs in var_dict.items():
        assign_list.append(ir.Assign(rhs, name_var_table[lhs_name],
                                     ir.Loc("dummy", -1)))
    blocks[0].body = assign_list + blocks[0].body
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks, typemap)
    apply_copy_propagate(blocks, in_copies_parfor, name_var_table, ext_func,
                         ext_data, typemap, calltypes)
    unwrap_parfor_blocks(parfor)
    # remove dummy assignments
    blocks[0].body = blocks[0].body[len(assign_list):]
    return


ir_utils.apply_copy_propagate_extensions[Parfor] = apply_copies_parfor


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
                        # and isinstance(rhs.value, pytypes.ModuleType)):
                    saved_globals[lhs.name] = stmt
                    block_defs.add(lhs.name)
                    # continue
                elif isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
                    if (rhs.value.name in saved_globals
                            or rhs.value.name in saved_getattrs):
                        saved_getattrs[lhs.name] = stmt
                        block_defs.add(lhs.name)
                        # continue
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


def get_parfor_call_table(parfor, call_table=None, reverse_call_table=None):
    if call_table is None:
        call_table = {}
    if reverse_call_table is None:
        reverse_call_table = {}
    blocks = wrap_parfor_blocks(parfor)
    call_table, reverse_call_table = get_call_table(blocks, call_table,
                                                    reverse_call_table)
    unwrap_parfor_blocks(parfor)
    return call_table, reverse_call_table


ir_utils.call_table_extensions[Parfor] = get_parfor_call_table


def get_parfor_tuple_table(parfor, tuple_table=None):
    if tuple_table is None:
        tuple_table = {}
    blocks = wrap_parfor_blocks(parfor)
    tuple_table = ir_utils.get_tuple_table(blocks, tuple_table)
    unwrap_parfor_blocks(parfor)
    return tuple_table


ir_utils.tuple_table_extensions[Parfor] = get_parfor_tuple_table


def get_parfor_array_accesses(parfor, accesses=None):
    if accesses is None:
        accesses = {}
    blocks = wrap_parfor_blocks(parfor)
    accesses = ir_utils.get_array_accesses(blocks, accesses)
    unwrap_parfor_blocks(parfor)
    return accesses


# parfor handler is same as
ir_utils.array_accesses_extensions[Parfor] = get_parfor_array_accesses


def parfor_add_offset_to_labels(parfor, offset):
    blocks = wrap_parfor_blocks(parfor)
    blocks = ir_utils.add_offset_to_labels(blocks, offset)
    blocks[0] = blocks[offset]
    blocks.pop(offset)
    unwrap_parfor_blocks(parfor, blocks)
    return


ir_utils.add_offset_to_labels_extensions[Parfor] = parfor_add_offset_to_labels


def parfor_find_max_label(parfor):
    blocks = wrap_parfor_blocks(parfor)
    max_label = ir_utils.find_max_label(blocks)
    unwrap_parfor_blocks(parfor)
    return max_label

ir_utils.find_max_label_extensions[Parfor] = parfor_find_max_label


def parfor_typeinfer(parfor, typeinferer):
    save_blocks = typeinferer.blocks
    blocks = wrap_parfor_blocks(parfor)
    index_vars = [l.index_variable for l in parfor.loop_nests]
    if len(parfor.loop_nests) > 1:
        index_vars.append(parfor.index_var)
    first_block = min(blocks.keys())
    loc = blocks[first_block].loc
    index_assigns = [ir.Assign(ir.Const(1, loc), v, loc) for v in index_vars]
    save_first_block_body = blocks[first_block].body
    blocks[first_block].body = index_assigns + blocks[first_block].body
    typeinferer.blocks = blocks
    typeinferer.build_constraint()
    typeinferer.blocks = save_blocks
    blocks[first_block].body = save_first_block_body
    unwrap_parfor_blocks(parfor)


typeinfer.typeinfer_extensions[Parfor] = parfor_typeinfer
