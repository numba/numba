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
            if call_list == ['stencil', numba]:
                stencil_calls.append(call_varname)
        if not stencil_calls:
            return  # return early if no stencil calls found
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            block = self.func_ir.blocks[label]
            for i, stmt in enumerate(block.body):
                # first find a stencil call
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
                                (self.typemap[in_arr.name],), block.scope, block.loc, in_arr,
                                self.typemap, self.calltypes)
                    out = self._mk_stencil_parfor(in_arr, out_arr, stencil_blocks)
                    block.body = block.body[:i] + out + block.body[i+1:]
                    return self.run()
        return

    def _mk_stencil_parfor(self, in_arr, out_arr, stencil_blocks):
        # run copy propagate to replace in_arr copies
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
        ir_utils.remove_dead(stencil_blocks, self.func_ir.arg_names, self.typemap)

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
                    # update min and max indices
                    start_lengths = list(map(min, start_lengths, index_list))
                    end_lengths = list(map(max, end_lengths, index_list))
                    # update access indices
                    tuple_var = ir.Var(scope, mk_unique_var(
                        "$parfor_index_tuple_var"), loc)
                    self.typemap[tuple_var.name] = types.containers.UniTuple(
                        types.intp, ndims)
                    index_vars = []
                    for i in range(ndims):
                        # stencil_index = parfor_index + stencil_const
                        index_const = ir.Var(scope, mk_unique_var("stencil_const_var"), loc)
                        self.typemap[index_const.name] = types.intp
                        const_assign = ir.Assign(ir.Const(index_list[i], loc), index_const, loc)
                        index_var = ir.Var(scope, mk_unique_var("stencil_index_var"), loc)
                        self.typemap[index_var.name] = types.intp
                        index_call = ir.Expr.binop('+', parfor_vars[i], index_const, loc)
                        self.calltypes[index_call] = ir_utils.find_op_typ('+', [types.intp, types.intp])
                        index_assign = ir.Assign(index_call, index_var, loc)
                        new_body.extend([const_assign, index_assign])
                        index_vars.append(index_var)

                    tuple_call = ir.Expr.build_tuple(index_vars, loc)
                    tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
                    new_body.append(tuple_assign)
                    getitem_call = ir.Expr.getitem(in_arr, tuple_var, loc)
                    self.calltypes[getitem_call] = signature(
                        self.typemap[in_arr.name].dtype, self.typemap[in_arr.name], self.typemap[tuple_var.name])
                    stmt.value = getitem_call

                new_body.append(stmt)
            block.body = new_body

        loopnests = []
        corrs = self.array_analysis.array_shape_classes[in_arr.name]
        sizes = self.array_analysis.array_size_vars[in_arr.name]
        assert ndims == len(sizes) and ndims == len(corrs)
        out = []
        for i in range(ndims):
            index_const = ir.Var(scope, mk_unique_var("stencil_const_var"), loc)
            self.typemap[index_const.name] = types.intp
            const_assign = ir.Assign(ir.Const(end_lengths[i], loc), index_const, loc)
            out.append(const_assign)
            last_ind = ir.Var(scope, mk_unique_var("last_ind"), loc)
            self.typemap[last_ind.name] = types.intp
            index_call = ir.Expr.binop('-', sizes[i], index_const, loc)
            self.calltypes[index_call] = ir_utils.find_op_typ('+', [types.intp, types.intp])
            index_assign = ir.Assign(index_call, last_ind, loc)
            out.append(index_assign)
            loopnests.append(numba.parfor.LoopNest(parfor_vars[i], start_lengths[i], last_ind, 1, corrs[i]))

        parfor_tuple_ind = ir.Var(scope, mk_unique_var(
            "$parfor_index_tuple_var"), loc)
        self.typemap[parfor_tuple_ind.name] = types.containers.UniTuple(
            types.intp, ndims)
        tuple_call = ir.Expr.build_tuple(parfor_vars, loc)
        tuple_assign = ir.Assign(tuple_call, parfor_tuple_ind, loc)
        assert isinstance(stencil_blocks[max(stencil_blocks.keys())].body[-1], ir.Return)
        return_val = stencil_blocks[max(stencil_blocks.keys())].body.pop().value
        stencil_blocks[max(stencil_blocks.keys())].body.append(tuple_assign)
        setitem_call = ir.SetItem(out_arr, parfor_tuple_ind, return_val, loc)
        self.calltypes[setitem_call] = signature(
            types.none, self.typemap[out_arr.name], self.typemap[parfor_tuple_ind.name], self.typemap[out_arr.name].dtype)
        stencil_blocks[max(stencil_blocks.keys())].body.append(setitem_call)

        init_block = ir.Block(scope, loc)
        parfor = numba.parfor.Parfor(loopnests, init_block, stencil_blocks, loc,
                        self.array_analysis, parfor_tuple_ind)
        stencil_blocks[min(stencil_blocks.keys())].dump()
        out.append(parfor)
        return out

def get_stencil_blocks(fcode, typingctx, args, scope, loc, in_arr, typemap, calltypes):
    from numba.targets.cpu import CPUContext
    from numba.targets.registry import cpu_target
    from numba.annotations import type_annotations

    targetctx = CPUContext(typingctx)
    stencil_func_ir = numba.compiler.run_frontend(fcode)
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
    # rename variables
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
