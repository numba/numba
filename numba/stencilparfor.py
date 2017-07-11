import numba
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature
from numba import ir_utils, ir, utils, array_analysis
from numba.ir_utils import get_call_table, find_topo_order, mk_unique_var

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
        call_table, _ = get_call_table(self.func_ir.blocks)
        stencil_calls = []
        for call_varname, call_list in call_table.items():
            if call_list == ['stencil', numba] or call_list == [stencil]:
                stencil_calls.append(call_varname)
        if not stencil_calls:
            return  # return early if no stencil calls found

        # find and transform stencil calls
        for label, block in self.func_ir.blocks.items():
            for i, stmt in enumerate(block.body):
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'
                        and stmt.value.func.name in stencil_calls):
                    assert len(stmt.value.args) == 2
                    in_arr = stmt.value.args[0]
                    out_arr = stmt.value.args[1]
                    # XXX is this correct?
                    fcode = fix_func_code(stmt.value.stencil_def.code,
                                        self.func_ir.func_id.func.__globals__)
                    stencil_blocks = get_stencil_blocks(fcode, self.typingctx,
                            (self.typemap[in_arr.name],), block.scope,
                            block.loc, in_arr, self.typemap, self.calltypes)
                    gen_nodes = self._mk_stencil_parfor(in_arr, out_arr,
                                                                stencil_blocks)
                    block.body = block.body[:i] + gen_nodes + block.body[i+1:]
                    return self.run()
        return

    def _mk_stencil_parfor(self, in_arr, out_arr, stencil_blocks):
        gen_nodes = []

        # run copy propagate to replace in_arr copies (e.g. a = A)
        in_cps, out_cps = ir_utils.copy_propagate(stencil_blocks, self.typemap)
        name_var_table = ir_utils.get_name_var_table(stencil_blocks)
        ir_utils.apply_copy_propagate(
            stencil_blocks,
            in_cps,
            name_var_table,
            array_analysis.copy_propagate_update_analysis,
            self.array_analysis,
            self.typemap,
            self.calltypes)
        ir_utils.remove_dead(stencil_blocks, self.func_ir.arg_names,
                                                                self.typemap)

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
                                            stencil_blocks, parfor_vars, in_arr)

        # create parfor loop nests
        loopnests = []
        corrs = self.array_analysis.array_shape_classes[in_arr.name]
        sizes = self.array_analysis.array_size_vars[in_arr.name]
        assert ndims == len(sizes) and ndims == len(corrs)
        for i in range(ndims):
            # set last index to size minus stencil size to avoid invalid access
            index_const = ir.Var(scope, mk_unique_var("stencil_const_var"), loc)
            self.typemap[index_const.name] = types.intp
            const_assign = ir.Assign(ir.Const(end_lengths[i], loc), index_const,
                                                                            loc)
            gen_nodes.append(const_assign)
            last_ind = ir.Var(scope, mk_unique_var("last_ind"), loc)
            self.typemap[last_ind.name] = types.intp
            index_call = ir.Expr.binop('-', sizes[i], index_const, loc)
            self.calltypes[index_call] = ir_utils.find_op_typ('+',
                                                    [types.intp, types.intp])
            index_assign = ir.Assign(index_call, last_ind, loc)
            gen_nodes.append(index_assign)
            # start from stencil size to avoid invalid array access
            loopnests.append(numba.parfor.LoopNest(parfor_vars[i],
                                abs(start_lengths[i]), last_ind, 1, corrs[i]))

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

        # replace return value to setitem to output array
        last_node = stencil_blocks[max(stencil_blocks.keys())].body.pop()
        assert isinstance(last_node, ir.Return)
        return_val = last_node.value
        setitem_call = ir.SetItem(out_arr, parfor_ind_var, return_val, loc)
        self.calltypes[setitem_call] = signature(
                                        types.none, self.typemap[out_arr.name],
                                        self.typemap[parfor_ind_var.name],
                                        self.typemap[out_arr.name].dtype
                                        )
        stencil_blocks[max(stencil_blocks.keys())].body.append(setitem_call)

        # empty init block
        init_block = ir.Block(scope, loc)
        parfor = numba.parfor.Parfor(loopnests, init_block, stencil_blocks, loc,
                        self.array_analysis, parfor_ind_var)

        gen_nodes.append(parfor)
        return gen_nodes

    def _replace_stencil_accesses(self, stencil_blocks, parfor_vars, in_arr):
        ndims = self.typemap[in_arr.name].ndim
        scope = in_arr.scope
        loc = in_arr.loc
        # replace access indices, find access lengths in each dimension
        start_lengths = ndims*[0]
        end_lengths = ndims*[0]
        for lable, block in stencil_blocks.items():
            new_body = []
            for stmt in block.body:
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'static_getitem'
                        and stmt.value.value.name == in_arr.name):
                    index_list = stmt.value.index
                    # handle 1D case
                    if ndims == 1:
                        assert isinstance(index_list, int)
                        index_list = [index_list]

                    # update min and max indices
                    start_lengths = list(map(min, start_lengths, index_list))
                    end_lengths = list(map(max, end_lengths, index_list))

                    # update access indices
                    index_vars = []
                    for i in range(ndims):
                        # stencil_index = parfor_index + stencil_const
                        index_const = ir.Var(scope,
                                        mk_unique_var("stencil_const_var"), loc)
                        self.typemap[index_const.name] = types.intp
                        const_assign = ir.Assign(ir.Const(index_list[i], loc),
                                                            index_const, loc)
                        index_var = ir.Var(scope,
                                        mk_unique_var("stencil_index_var"), loc)
                        self.typemap[index_var.name] = types.intp
                        index_call = ir.Expr.binop('+', parfor_vars[i],
                                                            index_const, loc)
                        self.calltypes[index_call] = ir_utils.find_op_typ('+',
                                                    [types.intp, types.intp])
                        index_assign = ir.Assign(index_call, index_var, loc)
                        new_body.extend([const_assign, index_assign])
                        index_vars.append(index_var)

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

                    # new getitem with the new index var
                    getitem_call = ir.Expr.getitem(in_arr, ind_var, loc)
                    self.calltypes[getitem_call] = signature(
                        self.typemap[in_arr.name].dtype,
                        self.typemap[in_arr.name],
                        self.typemap[ind_var.name])
                    stmt.value = getitem_call

                new_body.append(stmt)
            block.body = new_body

        return start_lengths, end_lengths

def get_stencil_blocks(fcode, typingctx, args, scope, loc, in_arr, typemap,
                                                                    calltypes):
    """get typed IR from stencil bytecode
    """
    from numba.targets.cpu import CPUContext
    from numba.targets.registry import cpu_target
    from numba.annotations import type_annotations

    # get untyped IR
    stencil_func_ir = numba.compiler.run_frontend(fcode)

    # get typed IR with a dummy pipeline (similar to test_parfors.py)
    targetctx = CPUContext(typingctx)
    with cpu_target.nested_context(typingctx, targetctx):
        tp = DummyPipeline(typingctx, targetctx, args, stencil_func_ir)

        numba.rewrites.rewrite_registry.apply(
            'before-inference', tp, tp.func_ir)

        tp.typemap, tp.return_type, tp.calltypes = numba.compiler.type_inference_stage(
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
    stencil_blocks = ir_utils.add_offset_to_labels(stencil_func_ir.blocks,
                                                        ir_utils.next_label())
    min_label = min(stencil_blocks.keys())
    max_label = max(stencil_blocks.keys())
    ir_utils._max_label = max_label

    # rename variables,
    var_dict = {}
    for v, typ in tp.typemap.items():
        new_var = ir.Var(scope, mk_unique_var(v), loc)
        var_dict[v] = new_var
        typemap[new_var.name] = typ  # add new var type for overall function
    ir_utils.replace_vars(stencil_blocks, var_dict)

    # add call types to overall function
    for call, call_typ in tp.calltypes.items():
        calltypes[call] = call_typ

    # TODO: handle closure vars
    # replace arg with arr
    for block in stencil_blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                stmt.value = in_arr
                break
    ir_utils.remove_dels(stencil_blocks)
    return stencil_blocks

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
