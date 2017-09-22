#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import numpy as np
import copy
from numba import compiler, types, ir_utils, ir, typing, numpy_support, utils
from numba import config
from numba.typing.templates import (AbstractTemplate, signature, infer_global,
                                    AbstractTemplate)
from numba.targets import cpu, registry
from numba.targets.imputils import lower_builtin
from llvmlite import ir as lir

def replace_return_with_setitem(blocks, index_vars):
    """
    Find return statements in the IR and replace them with a SetItem
    call of the value "returned" by the kernel into the result array.
    """
    for block in blocks.values():
        scope = block.scope
        loc = block.loc
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Return):
                # If 1D array then avoid the tuple construction.
                if len(index_vars) == 1:
                    rvar = ir.Var(scope, "out", loc)
                    ivar = ir.Var(scope, index_vars[0], loc)
                    new_body.append(ir.SetItem(rvar, ivar, stmt.value, loc))
                else:
                    # Convert the string names of the index variables into
                    # ir.Var's.
                    var_index_vars = []
                    for one_var in index_vars:
                        index_var = ir.Var(scope, one_var, loc)
                        var_index_vars += [index_var]

                    s_index_name = ir_utils.mk_unique_var("stencil_index")
                    s_index_var  = ir.Var(scope, s_index_name, loc)
                    # Build a tuple from the index ir.Var's.
                    tuple_call = ir.Expr.build_tuple(var_index_vars, loc)
                    new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                    rvar = ir.Var(scope, "out", loc)
                    # Write the return statements original value into
                    # the array using the tuple index.
                    si = ir.SetItem(rvar, s_index_var, stmt.value, loc)
                    new_body.append(si)
            else:
                new_body.append(stmt)
        block.body = new_body

def add_indices_to_kernel(kernel, ndim, neighborhood):
    """
    Transforms the stencil kernel as specified by the user into one
    that includes each dimension's index variable as part of the getitem
    calls.  So, in effect array[-1] becomes array[index0-1].
    """
    const_dict = {}
    kernel_consts = []

    if config.DEBUG_ARRAY_OPT == 1:
        print("add_indices_to_kernel", ndim, neighborhood)
        ir_utils.dump_blocks(kernel.blocks)

    need_to_calc_kernel = False
    if neighborhood is None:
        need_to_calc_kernel = True
    else:
        if len(neighborhood) != ndim:
            raise RuntimeError("%d dimensional neighborhood specified for %d dimensional input array" % (len(self.neighborhood), ndim))

    for block in kernel.blocks.values():
        scope = block.scope
        loc = block.loc
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Const):
                if config.DEBUG_ARRAY_OPT == 1:
                    print("remembering in const_dict", stmt.target.name, stmt.value.value)
                # Remember consts for use later.
                const_dict[stmt.target.name] = stmt.value.value
            if (isinstance(stmt, ir.Assign) and
                isinstance(stmt.value, ir.Expr) and
                stmt.value.op in ['getitem', 'static_getitem'] and
                stmt.value.value.name in kernel.arg_names):
                if config.DEBUG_ARRAY_OPT == 1:
                    print("found getitem to modify")
                # We found a getitem from the input array.
                if stmt.value.op == 'getitem':
                    stmt_index_var = stmt.value.index
                else:
                    stmt_index_var = stmt.value.index_var

                # Store the index used after looking up the variable in
                # the const dictionary.
                if need_to_calc_kernel:
                    if stmt_index_var.name in const_dict:
                        kernel_consts += [const_dict[stmt_index_var.name]]
                    else:
                        raise ValueError("Non-constant specified for stencil kernel index.")

                if ndim == 1:
                    # Single dimension always has index variable 'index0'.
                    # tmpvar will hold the real index and is computed by
                    # adding the relative offset in stmt.value.index to
                    # the current absolute location in index0.
                    index_var = ir.Var(scope, "index0", loc)
                    tmpname = ir_utils.mk_unique_var("stencil_index")
                    tmpvar  = ir.Var(scope, tmpname, loc)
                    acc_call = ir.Expr.binop('+', stmt_index_var,
                                             index_var, loc)
                    new_body.append(ir.Assign(acc_call, tmpvar, loc))
                    new_body.append(ir.Assign(
                                   ir.Expr.getitem(stmt.value.value,tmpvar,loc),
                                   stmt.target,loc))
                else:
                    index_vars = []
                    sum_results = []
                    s_index_name = ir_utils.mk_unique_var("stencil_index")
                    s_index_var  = ir.Var(scope, s_index_name, loc)
                    const_index_vars = []

                    # Same idea as above but you have to extract
                    # individual elements out of the tuple indexing
                    # expression and add the corresponding index variable
                    # to them and then reconstitute as a tuple that can
                    # index the array.
                    for dim in range(ndim):
                        tmpname = ir_utils.mk_unique_var("const_index")
                        tmpvar  = ir.Var(scope, tmpname, loc)
                        new_body.append(ir.Assign(ir.Const(dim, loc),
                                                  tmpvar, loc))
                        const_index_vars += [tmpvar]
                        index_var = ir.Var(scope, "index" + str(dim), loc)
                        index_vars += [index_var]

                    ind_stencils = []

                    for dim in range(ndim):
                        tmpname = ir_utils.mk_unique_var("ind_stencil_index")
                        tmpvar  = ir.Var(scope, tmpname, loc)
                        ind_stencils += [tmpvar]
                        getitemname = ir_utils.mk_unique_var("getitem")
                        getitemvar  = ir.Var(scope, getitemname, loc)
                        getitemcall = ir.Expr.getitem(stmt_index_var,
                                                   const_index_vars[dim], loc)
                        new_body.append(ir.Assign(getitemcall, getitemvar, loc))
                        acc_call = ir.Expr.binop('+', getitemvar,
                                                 index_vars[dim], loc)
                        new_body.append(ir.Assign(acc_call, tmpvar, loc))

                    tuple_call = ir.Expr.build_tuple(ind_stencils, loc)
                    new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                    new_body.append(ir.Assign(
                              ir.Expr.getitem(stmt.value.value,s_index_var,loc),
                              stmt.target,loc))
            else:
                new_body.append(stmt)
        block.body = new_body

    if need_to_calc_kernel:
        # Find the size of the kernel by finding the maximum absolute value
        # index used in the kernel specification.
        neighborhood = [[0,0] for _ in range(ndim)]
        #max_const = 0
        for index in kernel_consts:
            if isinstance(index, tuple):
                for i in range(len(index)):
                    te = index[i]
                    #max_const = max(max_const, abs(te))
                    neighborhood[i][0] = min(neighborhood[i][0], te)
                    neighborhood[i][1] = max(neighborhood[i][1], te)
                index_len = len(index)
            elif isinstance(index, int):
                #max_const = max(max_const, abs(index))
                index_len = 1
                neighborhood[0][0] = min(neighborhood[0][0], index)
                neighborhood[0][1] = max(neighborhood[0][1], index)
            else:
                raise ValueError("Non-tuple or non-integer used as stencil index.")
            if index_len != ndim:
                raise ValueError("Stencil index does not match array dimensionality.")
        self.neighborhood = neighborhood

    return neighborhood


class StencilFuncLowerer(object):
    '''Callable class responsible for lowering calls to a specific DUFunc.
    '''
    def __init__(self, sf):
        self.stencilFunc = sf

    def __call__(self, context, builder, sig, args):
        cres = self.stencilFunc.compile_for_argtys(sig.args, {}, sig.return_type, None)
        return context.call_internal(builder, cres.fndesc, sig, args)


class StencilFunc(object):
    """
    A special type to hold stencil information for the IR.
    """

    id_counter = 0

    def __init__(self, kernel_ir, mode, options):
        from numba import jit
        self.id = type(self).id_counter
        type(self).id_counter += 1
        self.kernel_ir = kernel_ir
        self.mode = mode
        self.options = options

        # stencils only supported for CPU context currently
        self._typingctx = registry.cpu_target.typing_context
        self._targetctx = registry.cpu_target.target_context
        self._typingctx.refresh()
        self._targetctx.refresh()
        self._install_type(self._typingctx)
        if "neighborhood" in self.options:
            self.neighborhood = self.options["neighborhood"]
        else:
            self.neighborhood = None
        self._cache = []
        self._type_cache = []
        self._lower_me = StencilFuncLowerer(self)

    def get_return_type(self, argtys):
        if config.DEBUG_ARRAY_OPT == 1:
            print("get_return_type", argtys)
            ir_utils.dump_blocks(self.kernel_ir.blocks)

        _, return_type, _ = compiler.type_inference_stage(
                self._typingctx,
                self.kernel_ir,
                argtys,
                None,
                {})
        real_ret = types.npytypes.Array(return_type, argtys[0].ndim,
                                                     argtys[0].layout)
        return real_ret

    def _install_type(self, typingctx):
        """Constructs and installs a typing class for a StencilFunc object in
        the input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
        _ty_cls = type('StencilFuncTyping_' +
                       str(hex(self.id).replace("-", "_")),
                       (AbstractTemplate,),
                       dict(key=self, generic=self._type_me))
        typingctx.insert_user_function(self, _ty_cls)

    def find_in_cache(self, argtys, kwtys):
        if config.DEBUG_ARRAY_OPT == 1:
            print("find_in_cache", argtys, kwtys, self._cache)
        largs = list(argtys)
        if 'out' in kwtys:
            largs.append(kwtys['out'])

        for centry in self._cache:
            (centry_argtys, cres, sigret) = centry
            if config.DEBUG_ARRAY_OPT == 1:
                print("find_in_cache search", centry_argtys)
            if centry_argtys == largs:
                if config.DEBUG_ARRAY_OPT == 1:
                    print("find_in_cache match")
                return centry
        if config.DEBUG_ARRAY_OPT == 1:
            print("find_in_cache NO match")
        return None

    def compile_for_argtys(self, argtys, kwtys, return_type, sigret):
        # look in cached functions first
        centry = self.find_in_cache(argtys, kwtys)
        if centry is not None:
            (argtys, cres, sigret) = centry
            return cres

        # look in the type cache to find if result array is passed
        result = None
        for (_argtyps, _sig, _res) in self._type_cache:
            if _argtyps == argtys:
                result = _res

        new_func = self._stencil_wrapper(result, sigret, return_type, *argtys)
        return new_func

    def _type_me(self, argtys, kwtys):
        """
        Implement AbstractTemplate.generic() for the typing class
        built by StencilFunc._install_type().
        Return the call-site signature.
        """
        if self.neighborhood is not None and len(self.neighborhood) != argtys[0].ndim:
            raise RuntimeError("%d dimensional neighborhood specified for %d dimensional input array" % (len(self.neighborhood), argtys[0].ndim))
        argtys_with_out = argtys
        if 'out' in kwtys:
            argtys_with_out += (kwtys['out'],)
        # look in the type cache first
        for (_argtyps, _sig, _) in self._type_cache:
            if _argtyps == argtys_with_out:
                return _sig

        real_ret = self.get_return_type(argtys)
        sig = signature(real_ret, *argtys_with_out)
        # use a dummy wrapper function to get pysignature
        if 'out' in kwtys:
            out_sig = ", out=None"
            result = kwtys['out']
        else:
            out_sig = ""
            result = None
        dummy_text = ("def __numba_dummy_stencil("
            + ",".join(self.kernel_ir.arg_names) + out_sig
            + "):\n    pass\n")
        exec(dummy_text)
        dummy_func = eval("__numba_dummy_stencil")
        sig.pysig = utils.pysignature(dummy_func)
        self._targetctx.insert_func_defn([(self._lower_me, self, argtys_with_out)])
        self._type_cache.append((argtys_with_out, sig, result))
        return sig

    def _stencil_wrapper(self, result, sigret, return_type, *args):
        # Copy the kernel so that our changes for this callsite
        # won't effect other callsites.
        kernel_copy = self.kernel_ir.copy()
        kernel_copy.blocks = copy.deepcopy(self.kernel_ir.blocks)
        ir_utils.remove_args(kernel_copy.blocks)

        the_array = args[0]

        if config.DEBUG_ARRAY_OPT == 1:
            print("_stencil_wrapper", return_type, return_type.dtype,
                                      type(return_type.dtype), args)
            ir_utils.dump_blocks(kernel_copy.blocks)

        stencil_func_name = "__numba_stencil_%s_%s" % (
                                        hex(id(the_array)).replace("-", "_"),
                                        hex(self.id).replace("-", "_"))

        index_vars = []
        for i in range(the_array.ndim):
            index_var_name = "index" + str(i)
            index_vars += [index_var_name]

        kernel_size = add_indices_to_kernel(kernel_copy, the_array.ndim,
                                            self.neighborhood)
        if config.DEBUG_ARRAY_OPT == 1:
            print("after add indices")
            ir_utils.dump_blocks(kernel_copy.blocks)

        replace_return_with_setitem(kernel_copy.blocks, index_vars)

        func_text = "def " + stencil_func_name + "("
        if result is None:
            func_text += ",".join(kernel_copy.arg_names) + "):\n"
        else:
            func_text += ",".join(kernel_copy.arg_names) + ", out=None):\n"
        func_text += "    full_shape = "
        func_text += kernel_copy.arg_names[0] + ".shape\n"
        if result is None:
            if "cval" in self.options:
                func_text += "    out = np.full(full_shape, " + str(self.options["cval"]) + ", dtype=np." + str(return_type.dtype) + ")\n"
            else:
                func_text += "    out = np.zeros(full_shape, dtype=np." + str(return_type.dtype) + ")\n"

        offset = 1
        for i in range(the_array.ndim):
            stri = str(i)
            for j in range(offset):
                func_text += "    "
            func_text += "for " + index_vars[i] + " in range("
            func_text += str(abs(kernel_size[i][0])) + ", full_shape["
            func_text += stri + "] - " + str(kernel_size[i][1]) + "):\n"
            offset += 1

        for j in range(offset):
            func_text += "    "
        func_text += "__sentinel__ = 0\n"
        func_text += "    return out\n"

        if config.DEBUG_ARRAY_OPT == 1:
            print("new stencil func text")
            print(func_text)

        exec(func_text)
        stencil_func = eval(stencil_func_name)
        if sigret is not None:
            pysig = utils.pysignature(stencil_func)
            sigret.pysig = pysig
        stencil_ir = compiler.run_frontend(stencil_func)
        ir_utils.remove_dels(stencil_ir.blocks)

        # rename all variables in stencil_ir afresh
        var_table = ir_utils.get_name_var_table(stencil_ir.blocks)
        new_var_dict = {}
        reserved_names = (["__sentinel__", "out"] +
                          kernel_copy.arg_names + index_vars)
        #  + list(param_dict.values()) + legal_loop_indices
        for name, var in var_table.items():
            if not (name in reserved_names):
                new_var_dict[name] = ir_utils.mk_unique_var(name)
        ir_utils.replace_var_names(stencil_ir.blocks, new_var_dict)

        stencil_stub_last_label = max(stencil_ir.blocks.keys()) + 1

        kernel_copy.blocks = ir_utils.add_offset_to_labels(
                                kernel_copy.blocks, stencil_stub_last_label)
        new_label = max(kernel_copy.blocks.keys()) + 1

        if config.DEBUG_ARRAY_OPT == 1:
            print("before replace sentinel stencil_ir")
            ir_utils.dump_blocks(stencil_ir.blocks)
            print("before replace sentinel kernel_copy")
            ir_utils.dump_blocks(kernel_copy.blocks)

        # Search all the block in the stencil outline for the sentinel.
        for label, block in stencil_ir.blocks.items():
            for i, inst in enumerate(block.body):
                if isinstance(
                        inst,
                        ir.Assign) and inst.target.name == "__sentinel__":
                    # We found the sentinel assignment.
                    loc = inst.loc
                    scope = block.scope
                    # split block across __sentinel__
                    # A new block is allocated for the statements prior to the
                    # sentinel but the new block maintains the current block
                    # label.
                    prev_block = ir.Block(scope, loc)
                    prev_block.body = block.body[:i]
                    # The current block is used for statements after sentinel.
                    block.body = block.body[i + 1:]
                    # But the current block gets a new label.
                    body_first_label = min(kernel_copy.blocks.keys())

                    # The previous block jumps to the minimum labelled block of
                    # the parfor body.
                    prev_block.append(ir.Jump(body_first_label, loc))
                    # Add all the parfor loop body blocks to the gufunc
                    # function's # IR.
                    for (l, b) in kernel_copy.blocks.items():
                        stencil_ir.blocks[l] = b
                    body_last_label = max(kernel_copy.blocks.keys())
                    stencil_ir.blocks[new_label] = block
                    stencil_ir.blocks[label] = prev_block
                    # Add a jump from the last parfor body block to the block
                    # containing statements after the sentinel.
                    stencil_ir.blocks[body_last_label].append(
                        ir.Jump(new_label, loc))
                    break
            else:
                continue
            break

        stencil_ir.blocks = ir_utils.rename_labels(stencil_ir.blocks)
        ir_utils.remove_dels(stencil_ir.blocks)

        assert(isinstance(the_array, types.Type))
        array_types = args

        new_stencil_param_types = list(array_types)

        if config.DEBUG_ARRAY_OPT == 1:
            print("new_stencil_param_types", new_stencil_param_types)
            ir_utils.dump_blocks(stencil_ir.blocks)

        new_func = compiler.compile_ir(
            self._typingctx,
            self._targetctx,
            stencil_ir,
            new_stencil_param_types,
            None,
            compiler.DEFAULT_FLAGS,
            {})
        if sigret is not None:
            self._cache.append((list(new_stencil_param_types),
                                new_func, sigret))
        return new_func

    def __call__(self, *args, **kwargs):
        if self.neighborhood is not None and len(self.neighborhood) != args[0].ndim:
            raise RuntimeError("%d dimensional neighborhood specified for %d dimensional input array" % (len(self.neighborhood), args[0].ndim))

        if 'out' in kwargs:
            result = kwargs['out']
            rdtype = result.dtype
            # Is 'C' correct here?
            rttype = numpy_support.from_dtype(rdtype)
            result_type = types.npytypes.Array(rttype, result.ndim, 'C')
            array_types = tuple([typing.typeof.typeof(x) for x in args])
            array_types_full = tuple([typing.typeof.typeof(x) for x in args] + [result_type])
        else:
            result = None
            array_types = tuple([typing.typeof.typeof(x) for x in args])
            array_types_full = array_types

        if config.DEBUG_ARRAY_OPT == 1:
            print("__call__", array_types, args, kwargs)

        real_ret = self.get_return_type(array_types)
        new_func = self._stencil_wrapper(result, None, real_ret, *array_types_full)

        if result is None:
            return new_func.entry_point(*args)
        else:
            return new_func.entry_point(*(args+(result,)))

def stencil(func_or_mode='constant', **options):
    # called on function without specifying mode style
    if not isinstance(func_or_mode, str):
        mode = 'constant'  # default style
        func = func_or_mode
    else:
        assert isinstance(func_or_mode, str), """stencil mode should be
                                                        a string"""
        mode = func_or_mode
        func = None
    wrapper = _stencil(mode, options)
    if func is not None:
        return wrapper(func)
    return wrapper

def _stencil(mode, options):
    if mode != 'constant':
        raise ValueError("Unsupported mode style " + mode)

    def decorated(func):
        kernel_ir = compiler.run_frontend(func)
        return StencilFunc(kernel_ir, mode, options)

    return decorated


@infer_global(stencil)
class StencilInfer(AbstractTemplate):
    "type inference for dummy stencil calls"
    def generic(self, args, kws):
        return signature(types.intp, *(args + tuple(kws.values())))

@lower_builtin(stencil)
def stencil_dummy_lower(context, builder, sig, args):
    "lowering for dummy stencil calls"
    return lir.Constant(lir.IntType(types.intp.bitwidth), 0)
