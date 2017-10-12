#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import types as pytypes
import numba
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature
from numba import ir_utils, ir, utils, array_analysis, config
from numba.ir_utils import (get_call_table, find_topo_order, mk_unique_var,
                            compile_to_numba_ir, replace_arg_nodes, guard,
                            find_callname)
from operator import add
import numpy as np
import numbers
import copy

def stencil():
    pass

@infer_global(stencil)
class Stencil(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *args)

class StencilPass(object):
    def __init__(self, func_ir, typemap, calltypes, array_analysis, typingctx):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.array_analysis = array_analysis
        self.typingctx = typingctx

    def run(self):
        from numba.stencil import StencilFunc
        from numba import compiler

        call_table, _ = get_call_table(self.func_ir.blocks)
        stencil_calls = []
        stencil_dict = {}
        for call_varname, call_list in call_table.items():
            if isinstance(call_list[0], StencilFunc):
                stencil_calls.append(call_varname)
                stencil_dict[call_varname] = call_list[0]
        if not stencil_calls:
            return  # return early if no stencil calls found

        # find and transform stencil calls
        for label, block in self.func_ir.blocks.items():
            for i, stmt in enumerate(block.body):
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'
                        and stmt.value.func.name in stencil_calls):
                    kws = dict(stmt.value.kws)
                    input_dict = {i: stmt.value.args[i] for i in
                                                    range(len(stmt.value.args))}
                    in_args = stmt.value.args
                    arg_typemap = tuple(self.typemap[i.name] for i in
                                                                stmt.value.args)
                    if 'out' in kws:
                        out_arr = kws['out']
                    else:
                        out_arr = None

                    sf = stencil_dict[stmt.value.func.name]
                    stencil_blocks, rt, arg_to_arr_dict = get_stencil_blocks(sf,
                            self.typingctx, arg_typemap,
                            block.scope, block.loc, input_dict,
                            self.typemap, self.calltypes)
                    index_offsets = sf.options.get('index_offsets', None)
                    gen_nodes = self._mk_stencil_parfor(label, in_args, out_arr,
                            stencil_blocks, index_offsets, stmt.target, rt, sf,
                            arg_to_arr_dict)
                    block.body = block.body[:i] + gen_nodes + block.body[i+1:]
                    return self.run()
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'
                        and guard(find_callname, self.func_ir, stmt.value)
                                    == ('stencil', 'numba')):
                    # remove dummy stencil() call
                    stmt.value = ir.Const(0, stmt.loc)
        return

    def _mk_stencil_parfor(self, label, in_args, out_arr, stencil_blocks,
                           index_offsets, target, return_type, stencil_func,
                           arg_to_arr_dict):
        gen_nodes = []

        if config.DEBUG_ARRAY_OPT == 1:
            print("_mk_stencil_parfor", label, in_args, out_arr, index_offsets,
                   return_type, stencil_func, stencil_blocks)
            ir_utils.dump_blocks(stencil_blocks)

        in_arr = in_args[0]
        # run copy propagate to replace in_args copies (e.g. a = A)
        in_arr_typ = self.typemap[in_arr.name]
        in_cps, out_cps = ir_utils.copy_propagate(stencil_blocks, self.typemap)
        name_var_table = ir_utils.get_name_var_table(stencil_blocks)
        ir_utils.apply_copy_propagate(
            stencil_blocks,
            in_cps,
            name_var_table,
            lambda a, b, c, d:None, # a null func
            None,                   # no extra data
            self.typemap,
            self.calltypes)
        if config.DEBUG_ARRAY_OPT == 1:
            ir_utils.dump_blocks(stencil_blocks)
        ir_utils.remove_dead(stencil_blocks, self.func_ir.arg_names,
                                                                self.typemap)
        if config.DEBUG_ARRAY_OPT == 1:
            ir_utils.dump_blocks(stencil_blocks)

        # create parfor vars
        ndims = self.typemap[in_arr.name].ndim
        scope = in_arr.scope
        loc = in_arr.loc
        parfor_vars = []
        for i in range(ndims):
            parfor_var = ir.Var(scope, mk_unique_var(
                "$parfor_index_var"), loc)
            self.typemap[parfor_var.name] = types.intp
            parfor_vars.append(parfor_var)

        start_lengths, end_lengths = self._replace_stencil_accesses(
             stencil_blocks, parfor_vars, in_args, index_offsets, stencil_func,
             arg_to_arr_dict)

        # create parfor loop nests
        loopnests = []
        equiv_set = self.array_analysis.get_equiv_set(label)
        in_arr_dim_sizes = equiv_set.get_shape(in_arr.name)

        assert ndims == len(in_arr_dim_sizes)
        for i in range(ndims):
            last_ind = self._get_stencil_last_ind(in_arr_dim_sizes[i],
                                        end_lengths[i], gen_nodes, scope, loc)
            start_ind = self._get_stencil_start_ind(
                                        start_lengths[i], gen_nodes, scope, loc)
            # start from stencil size to avoid invalid array access
            loopnests.append(numba.parfor.LoopNest(parfor_vars[i],
                                start_ind, last_ind, 1))

        # replace return value to setitem to output array
        return_node = stencil_blocks[max(stencil_blocks.keys())].body.pop()
        assert isinstance(return_node, ir.Return)
        last_node = stencil_blocks[max(stencil_blocks.keys())].body.pop()
        assert isinstance(last_node, ir.Assign)
        assert isinstance(last_node.value, ir.Expr)
        assert last_node.value.op == 'cast'
        return_val = last_node.value.value

        # create parfor index var
        if ndims == 1:
            parfor_ind_var = parfor_vars[0]
        else:
            parfor_ind_var = ir.Var(scope, mk_unique_var(
                "$parfor_index_tuple_var"), loc)
            self.typemap[parfor_ind_var.name] = types.containers.UniTuple(
                types.intp, ndims)
            tuple_call = ir.Expr.build_tuple(parfor_vars, loc)
            tuple_assign = ir.Assign(tuple_call, parfor_ind_var, loc)
            stencil_blocks[max(stencil_blocks.keys())].body.append(tuple_assign)

        # empty init block
        init_block = ir.Block(scope, loc)
        if out_arr == None:
            in_arr_typ = self.typemap[in_arr.name]

            shape_name = ir_utils.mk_unique_var("in_arr_shape")
            shape_var = ir.Var(scope, shape_name, loc)
            shape_getattr = ir.Expr.getattr(in_arr, "shape", loc)
            self.typemap[shape_name] = types.containers.UniTuple(types.intp,
                                                               in_arr_typ.ndim)
            init_block.body.extend([ir.Assign(shape_getattr, shape_var, loc)])

            zero_name = ir_utils.mk_unique_var("zero_val")
            zero_var = ir.Var(scope, zero_name, loc)
            if "cval" in stencil_func.options:
                temp2 = return_type.dtype(stencil_func.options["cval"])
            else:
                temp2 = return_type.dtype(0)
            full_const = ir.Const(temp2, loc)
            self.typemap[zero_name] = return_type.dtype
            init_block.body.extend([ir.Assign(full_const, zero_var, loc)])

            so_name = ir_utils.mk_unique_var("stencil_output")
            out_arr = ir.Var(scope, so_name, loc)
            self.typemap[out_arr.name] = numba.types.npytypes.Array(
                                                           return_type.dtype,
                                                           in_arr_typ.ndim,
                                                           in_arr_typ.layout)
            dtype_g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
            self.typemap[dtype_g_np_var.name] = types.misc.Module(np)
            dtype_g_np = ir.Global('np', np, loc)
            dtype_g_np_assign = ir.Assign(dtype_g_np, dtype_g_np_var, loc)
            init_block.body.append(dtype_g_np_assign)

            dtype_np_attr_call = ir.Expr.getattr(dtype_g_np_var, return_type.dtype.name, loc)
            dtype_attr_var = ir.Var(scope, mk_unique_var("$np_attr_attr"), loc)
            self.typemap[dtype_attr_var.name] = types.functions.NumberClass(return_type.dtype)
            dtype_attr_assign = ir.Assign(dtype_np_attr_call, dtype_attr_var, loc)
            init_block.body.append(dtype_attr_assign)

            stmts = ir_utils.gen_np_call("full",
                                       np.full,
                                       out_arr,
                                       [shape_var, zero_var, dtype_attr_var],
                                       self.typingctx,
                                       self.typemap,
                                       self.calltypes)
            init_block.body.extend(stmts)

        setitem_call = ir.SetItem(out_arr, parfor_ind_var, return_val, loc)
        self.calltypes[setitem_call] = signature(
                                        types.none, self.typemap[out_arr.name],
                                        self.typemap[parfor_ind_var.name],
                                        self.typemap[out_arr.name].dtype
                                        )
        stencil_blocks[max(stencil_blocks.keys())].body.append(setitem_call)

        parfor = numba.parfor.Parfor(loopnests, init_block, stencil_blocks,
                                     loc, parfor_ind_var, equiv_set)
        parfor.patterns = [('stencil', [start_lengths, end_lengths])]
        gen_nodes.append(parfor)
        gen_nodes.append(ir.Assign(out_arr, target, loc))
        return gen_nodes

    def _get_stencil_last_ind(self, dim_size, end_length, gen_nodes, scope,
                                                                        loc):
        last_ind = dim_size
        # TODO: support negative end length
        if end_length != 0:
            # set last index to size minus stencil size to avoid invalid
            # memory access
            index_const = ir.Var(scope, mk_unique_var("stencil_const_var"),
                                                                        loc)
            self.typemap[index_const.name] = types.intp
            if isinstance(end_length, numbers.Number):
                const_assign = ir.Assign(ir.Const(end_length, loc),
                                                        index_const, loc)
            else:
                const_assign = ir.Assign(end_length, index_const, loc)

            gen_nodes.append(const_assign)
            last_ind = ir.Var(scope, mk_unique_var("last_ind"), loc)
            self.typemap[last_ind.name] = types.intp
            index_call = ir.Expr.binop('-', dim_size, index_const, loc)
            self.calltypes[index_call] = ir_utils.find_op_typ('+',
                                                [types.intp, types.intp])
            index_assign = ir.Assign(index_call, last_ind, loc)
            gen_nodes.append(index_assign)

        return last_ind

    def _get_stencil_start_ind(self, start_length, gen_nodes, scope, loc):
        if isinstance(start_length, int):
            return abs(min(start_length, 0))
        def get_start_ind(s_length):
            return abs(min(s_length, 0))
        f_ir = compile_to_numba_ir(get_start_ind, {}, self.typingctx,
                                 (types.intp,), self.typemap, self.calltypes)
        assert len(f_ir.blocks) == 1
        block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(block, [start_length])
        gen_nodes += block.body[:-2]
        ret_var = block.body[-2].value.value
        return ret_var

    def _replace_stencil_accesses(self, stencil_blocks, parfor_vars, in_args,
                                  index_offsets, stencil_func, arg_to_arr_dict):
        in_arr = in_args[0]
        in_arg_names = [x.name for x in in_args]

        if "standard_indexing" in stencil_func.options:
            standard_indexed = [arg_to_arr_dict[x] for x in
                                     stencil_func.options["standard_indexing"]]
        else:
            standard_indexed = []

        ndims = self.typemap[in_arr.name].ndim
        scope = in_arr.scope
        loc = in_arr.loc
        # replace access indices, find access lengths in each dimension
        need_to_calc_kernel = False
        if stencil_func.neighborhood is None:
            need_to_calc_kernel = True

        if need_to_calc_kernel:
            start_lengths = ndims*[0]
            end_lengths = ndims*[0]
        else:
            start_lengths = [x[0] for x in stencil_func.neighborhood]
            end_lengths   = [x[1] for x in stencil_func.neighborhood]

        for label, block in stencil_blocks.items():
            new_body = []
            for stmt in block.body:
                if ((isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op in ['setitem', 'static_setitem']
                        and stmt.value.value.name in in_arg_names) or 
                   ((isinstance(stmt, ir.SetItem) or
                     isinstance(stmt, ir.StaticSetItem))
                        and stmt.target.name in in_arg_names)):
                    raise ValueError("Assignments to arrays passed to stencil kernels is not allowed.")
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op in ['static_getitem', 'getitem']
                        and stmt.value.value.name in in_arg_names
                        and stmt.value.value.name not in standard_indexed):
                    index_list = stmt.value.index
                    # handle 1D case
                    if ndims == 1:
                        #assert isinstance(index_list, int)
                        index_list = [index_list]
                    if index_offsets:
                        index_list = self._add_index_offsets(index_list,
                                    list(index_offsets), new_body, scope, loc)

                    # update min and max indices
                    if need_to_calc_kernel:
                        # all indices should be integer to be able to calculate
                        # neighborhood automatically
                        if (isinstance(index_list, ir.Var) or 
                            any([not isinstance(v, int) for v in index_list])):
                            raise ValueError("Variable stencil index only "
                                "possible with known neighborhood")
                        start_lengths = list(map(min, start_lengths,
                                                                    index_list))
                        end_lengths = list(map(max, end_lengths, index_list))

                    # update access indices
                    index_vars = self._add_index_offsets(parfor_vars,
                                list(index_list), new_body, scope, loc)

                    # new access index tuple
                    if ndims == 1:
                        ind_var = index_vars[0]
                    else:
                        ind_var = ir.Var(scope, mk_unique_var(
                            "$parfor_index_ind_var"), loc)
                        self.typemap[ind_var.name] = types.containers.UniTuple(
                            types.intp, ndims)
                        tuple_call = ir.Expr.build_tuple(index_vars, loc)
                        tuple_assign = ir.Assign(tuple_call, ind_var, loc)
                        new_body.append(tuple_assign)

                    # getitem return type is scalar if all indices are integer
                    if all([self.typemap[v.name] == types.intp
                                                        for v in index_vars]):
                        getitem_return_typ = self.typemap[
                                                    stmt.value.value.name].dtype
                    else:
                        # getitem returns an array
                        getitem_return_typ = self.typemap[stmt.value.value.name]
                    # new getitem with the new index var
                    getitem_call = ir.Expr.getitem(stmt.value.value, ind_var,
                                                                            loc)
                    self.calltypes[getitem_call] = signature(
                        getitem_return_typ,
                        self.typemap[stmt.value.value.name],
                        self.typemap[ind_var.name])
                    stmt.value = getitem_call

                new_body.append(stmt)
            block.body = new_body

        return start_lengths, end_lengths

    def _add_index_offsets(self, index_list, index_offsets, new_body, scope,
                                                                        loc):
        # shortcut if all values are integer
        if all([isinstance(v, int) for v in index_list+index_offsets]):
            # add offsets in all dimensions
            return list(map(add, index_list, index_offsets))
        assert len(index_list) == len(index_offsets)

        out_nodes = []
        index_vars = []
        for i in range(len(index_list)):
            # new_index = old_index + offset
            old_index_var = index_list[i]
            if isinstance(old_index_var, int):
                old_index_var = ir.Var(scope,
                                mk_unique_var("old_index_var"), loc)
                self.typemap[old_index_var.name] = types.intp
                const_assign = ir.Assign(ir.Const(index_list[i], loc),
                                                    old_index_var, loc)
                out_nodes.append(const_assign)

            offset_var = index_offsets[i]
            if isinstance(offset_var, int):
                offset_var = ir.Var(scope,
                                mk_unique_var("offset_var"), loc)
                self.typemap[offset_var.name] = types.intp
                const_assign = ir.Assign(ir.Const(index_offsets[i], loc),
                                                offset_var, loc)
                out_nodes.append(const_assign)

            if (isinstance(old_index_var, slice)
                    or isinstance(self.typemap[old_index_var.name],
                                    types.misc.SliceType)):
                # only one arg can be slice
                assert self.typemap[offset_var.name] == types.intp
                index_var = self._add_offset_to_slice(old_index_var, offset_var,
                                                        out_nodes, scope, loc)
                index_vars.append(index_var)
                continue

            if (isinstance(offset_var, slice)
                    or isinstance(self.typemap[offset_var.name],
                                    types.misc.SliceType)):
                # only one arg can be slice
                assert self.typemap[old_index_var.name] == types.intp
                index_var = self._add_offset_to_slice(offset_var, old_index_var,
                                                        out_nodes, scope, loc)
                index_vars.append(index_var)
                continue

            index_var = ir.Var(scope,
                            mk_unique_var("offset_stencil_index"), loc)
            self.typemap[index_var.name] = types.intp
            index_call = ir.Expr.binop('+', old_index_var,
                                                offset_var, loc)
            self.calltypes[index_call] = ir_utils.find_op_typ('+',
                                        [types.intp, types.intp])
            index_assign = ir.Assign(index_call, index_var, loc)
            out_nodes.append(index_assign)
            index_vars.append(index_var)

        new_body.extend(out_nodes)
        return index_vars

    def _add_offset_to_slice(self, slice_var, offset_var, out_nodes, scope,
                                loc):
        if isinstance(slice_var, slice):
            f_text = """def f(offset):
                return slice({} + offset, {} + offset)
            """.format(slice_var.start, slice_var.stop)
            loc = {}
            exec(f_text, {}, loc)
            f = loc['f']
            args = [offset_var]
            arg_typs = (types.intp,)
        else:
            def f(old_slice, offset):
                return slice(old_slice.start + offset, old_slice.stop + offset)
            args = [slice_var, offset_var]
            slice_type = self.typemap[slice_var.name]
            arg_typs = (slice_type, types.intp,)
        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, _globals, self.typingctx, arg_typs,
                                    self.typemap, self.calltypes)
        _, block = f_ir.blocks.popitem()
        replace_arg_nodes(block, args)
        new_index = block.body[-2].value.value
        out_nodes.extend(block.body[:-2])  # ignore return nodes
        return new_index

def get_stencil_blocks(sf, typingctx, args, scope, loc, input_dict, typemap,
                                                                    calltypes):
    """get typed IR from stencil bytecode
    """
    from numba.targets.cpu import CPUContext
    from numba.targets.registry import cpu_target
    from numba.annotations import type_annotations
    from numba.compiler import type_inference_stage

    # get untyped IR
    stencil_func_ir = sf.kernel_ir.copy()
    # copy the IR nodes to avoid changing IR in the StencilFunc object
    stencil_blocks = copy.deepcopy(stencil_func_ir.blocks)
    stencil_func_ir.blocks = stencil_blocks

    # get typed IR with a dummy pipeline (similar to test_parfors.py)
    targetctx = CPUContext(typingctx)
    with cpu_target.nested_context(typingctx, targetctx):
        tp = DummyPipeline(typingctx, targetctx, args, stencil_func_ir)

        numba.rewrites.rewrite_registry.apply(
            'before-inference', tp, tp.func_ir)

        tp.typemap, tp.return_type, tp.calltypes = type_inference_stage(
            tp.typingctx, tp.func_ir, tp.args, None)

        type_annotation = type_annotations.TypeAnnotation(
            func_ir=tp.func_ir,
            typemap=tp.typemap,
            calltypes=tp.calltypes,
            lifted=(),
            lifted_from=None,
            args=tp.args,
            return_type=tp.return_type,
            html_output=numba.config.HTML)

        numba.rewrites.rewrite_registry.apply(
            'after-inference', tp, tp.func_ir)

    # make block labels unique
    stencil_blocks = ir_utils.add_offset_to_labels(stencil_blocks,
                                                        ir_utils.next_label())
    min_label = min(stencil_blocks.keys())
    max_label = max(stencil_blocks.keys())
    ir_utils._max_label = max_label

    if config.DEBUG_ARRAY_OPT == 1:
        print("Initial stencil_blocks")
        ir_utils.dump_blocks(stencil_blocks)

    # rename variables,
    var_dict = {}
    for v, typ in tp.typemap.items():
        new_var = ir.Var(scope, mk_unique_var(v), loc)
        var_dict[v] = new_var
        typemap[new_var.name] = typ  # add new var type for overall function
    ir_utils.replace_vars(stencil_blocks, var_dict)

    if config.DEBUG_ARRAY_OPT == 1:
        print("After replace_vars")
        ir_utils.dump_blocks(stencil_blocks)

    # add call types to overall function
    for call, call_typ in tp.calltypes.items():
        calltypes[call] = call_typ

    arg_to_arr_dict = {}
    # TODO: handle closure vars
    # replace arg with arr
    for block in stencil_blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                if config.DEBUG_ARRAY_OPT == 1:
                    print("input_dict", input_dict, stmt.value.index,
                               stmt.value.name, stmt.value.index in input_dict)
                arg_to_arr_dict[stmt.value.name] = input_dict[stmt.value.index].name
                stmt.value = input_dict[stmt.value.index]

    if config.DEBUG_ARRAY_OPT == 1:
        print("arg_to_arr_dict", arg_to_arr_dict)
        print("After replace arg with arr")
        ir_utils.dump_blocks(stencil_blocks)

    ir_utils.remove_dels(stencil_blocks)
    return stencil_blocks, sf.get_return_type(args)[0], arg_to_arr_dict

class DummyPipeline(object):
    def __init__(self, typingctx, targetctx, args, f_ir):
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.args = args
        self.func_ir = f_ir
        self.typemap = None
        self.return_type = None
        self.calltypes = None

def fix_func_code(fcode, glbls):
    # similar to inline_closurecall.py

    nfree = len(fcode.co_freevars)
    func_env = "\n".join(["  c_%d = None" % i for i in range(nfree)])
    func_clo = ",".join(["c_%d" % i for i in range(nfree)])
    func_arg = ",".join(["x_%d" % i for i in range(fcode.co_argcount)])
    func_text = "def g():\n%s\n  def f(%s):\n    return (%s)\n  return f" % (
        func_env, func_arg, func_clo)
    loc = {}
    exec(func_text, glbls, loc)

    # hack parameter name .0 for Python 3 versions < 3.6
    if utils.PYVERSION >= (3,) and utils.PYVERSION < (3, 6):
        co_varnames = list(fcode.co_varnames)
        if co_varnames[0] == ".0":
            co_varnames[0] = "implicit0"
        fcode = pytypes.CodeType(
            fcode.co_argcount,
            fcode.co_kwonlyargcount,
            fcode.co_nlocals,
            fcode.co_stacksize,
            fcode.co_flags,
            fcode.co_code,
            fcode.co_consts,
            fcode.co_names,
            tuple(co_varnames),
            fcode.co_filename,
            fcode.co_name,
            fcode.co_firstlineno,
            fcode.co_lnotab,
            fcode.co_freevars,
            fcode.co_cellvars)

    f = loc['g']()
    f.__code__ = fcode
    f.__name__ = fcode.co_name
    return f
