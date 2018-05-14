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
import sys, math
from functools import reduce
from collections import defaultdict
from contextlib import contextmanager

import numba
from numba import ir, ir_utils, types, typing, rewrites, config, analysis, prange, pndindex
from numba import array_analysis, postproc, typeinfer
from numba.numpy_support import as_dtype
from numba.typing.templates import infer_global, AbstractTemplate
from numba import stencilparfor
from numba.stencilparfor import StencilPass


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
    replace_vars_inner,
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
    add_offset_to_labels,
    find_callname,
    find_build_sequence,
    guard,
    require,
    GuardException,
    compile_to_numba_ir,
    get_definition,
    build_definitions,
    replace_arg_nodes,
    replace_returns,
    is_getitem,
    is_setitem,
    is_get_setitem,
    index_var_of_get_setitem,
    set_index_var_of_get_setitem)

from numba.analysis import (compute_use_defs, compute_live_map,
                            compute_dead_maps, compute_cfg_from_blocks)
from numba.controlflow import CFGraph
from numba.typing import npydecl, signature
from numba.types.functions import Function
from numba.array_analysis import (random_int_args, random_1arg_size,
                                  random_2arg_sizelast, random_3arg_sizelast,
                                  random_calls, assert_equiv)
from numba.extending import overload
import copy
import numpy
import numpy as np
# circular dependency: import numba.npyufunc.dufunc.DUFunc

sequential_parfor_lowering = False

# init_prange is a sentinel call that specifies the start of the initialization
# code for the computation in the upcoming prange call
# This lets the prange pass to put the code in the generated parfor's init_block
def init_prange():
    return

@overload(init_prange)
def init_prange_overload():
    def no_op():
        return
    return no_op

class internal_prange(object):

    def __new__(cls, *args):
        return range(*args)

def min_parallel_impl(return_type, arg):
    # XXX: use prange for 1D arrays since pndindex returns a 1-tuple instead of
    # integer. This causes type and fusion issues.
    if arg.ndim == 1:
        def min_1(in_arr):
            numba.parfor.init_prange()
            val = numba.targets.builtins.get_type_max_value(in_arr.dtype)
            for i in numba.parfor.internal_prange(len(in_arr)):
                val = min(val, in_arr[i])
            return val
    else:
        def min_1(in_arr):
            numba.parfor.init_prange()
            val = numba.targets.builtins.get_type_max_value(in_arr.dtype)
            for i in numba.pndindex(in_arr.shape):
                val = min(val, in_arr[i])
            return val
    return min_1

def max_parallel_impl(return_type, arg):
    if arg.ndim == 1:
        def max_1(in_arr):
            numba.parfor.init_prange()
            val = numba.targets.builtins.get_type_min_value(in_arr.dtype)
            for i in numba.parfor.internal_prange(len(in_arr)):
                val = max(val, in_arr[i])
            return val
    else:
        def max_1(in_arr):
            numba.parfor.init_prange()
            val = numba.targets.builtins.get_type_min_value(in_arr.dtype)
            for i in numba.pndindex(in_arr.shape):
                val = max(val, in_arr[i])
            return val
    return max_1

def argmin_parallel_impl(in_arr):
    numba.parfor.init_prange()
    A = in_arr.ravel()
    init_val = numba.targets.builtins.get_type_max_value(A.dtype)
    ival = numba.typing.builtins.IndexValue(0, init_val)
    for i in numba.parfor.internal_prange(len(A)):
        curr_ival = numba.typing.builtins.IndexValue(i, A[i])
        ival = min(ival, curr_ival)
    return ival.index

def argmax_parallel_impl(in_arr):
    numba.parfor.init_prange()
    A = in_arr.ravel()
    init_val = numba.targets.builtins.get_type_min_value(A.dtype)
    ival = numba.typing.builtins.IndexValue(0, init_val)
    for i in numba.parfor.internal_prange(len(A)):
        curr_ival = numba.typing.builtins.IndexValue(i, A[i])
        ival = max(ival, curr_ival)
    return ival.index

def dotvv_parallel_impl(a, b):
    numba.parfor.init_prange()
    l = a.shape[0]
    m = b.shape[0]
    # TODO: investigate assert_equiv
    #assert_equiv("sizes of l, m do not match", l, m)
    s = 0
    for i in numba.parfor.internal_prange(l):
        s += a[i] * b[i]
    return s

def dotvm_parallel_impl(a, b):
    numba.parfor.init_prange()
    l = a.shape
    m, n = b.shape
    # TODO: investigate assert_equiv
    #assert_equiv("Sizes of l, m do not match", l, m)
    c = np.zeros(n, a.dtype)
    # TODO: evaluate dotvm implementation options
    #for i in prange(n):
    #    s = 0
    #    for j in range(m):
    #        s += a[j] * b[j, i]
    #    c[i] = s
    for i in numba.parfor.internal_prange(m):
        c += a[i] * b[i, :]
    return c

def dotmv_parallel_impl(a, b):
    numba.parfor.init_prange()
    m, n = a.shape
    l = b.shape
    # TODO: investigate assert_equiv
    #assert_equiv("sizes of n, l do not match", n, l)
    c = np.empty(m, a.dtype)
    for i in numba.parfor.internal_prange(m):
        s = 0
        for j in range(n):
            s += a[i, j] * b[j]
        c[i] = s
    return c

def dot_parallel_impl(return_type, atyp, btyp):
    # Note that matrix matrix multiply is not translated.
    if (isinstance(atyp, types.npytypes.Array) and
        isinstance(btyp, types.npytypes.Array)):
        if atyp.ndim == btyp.ndim == 1:
            return dotvv_parallel_impl
        # TODO: evaluate support for dotvm and enable
        #elif atyp.ndim == 1 and btyp.ndim == 2:
        #    return dotvm_parallel_impl
        elif atyp.ndim == 2 and btyp.ndim == 1:
            return dotmv_parallel_impl

def sum_parallel_impl(return_type, arg):
    zero = return_type(0)

    if arg.ndim == 1:
        def sum_1(in_arr):
            numba.parfor.init_prange()
            val = zero
            for i in numba.parfor.internal_prange(len(in_arr)):
                val += in_arr[i]
            return val
    else:
        def sum_1(in_arr):
            numba.parfor.init_prange()
            val = zero
            for i in numba.pndindex(in_arr.shape):
                val += in_arr[i]
            return val
    return sum_1

def prod_parallel_impl(return_type, arg):
    one = return_type(1)

    if arg.ndim == 1:
        def prod_1(in_arr):
            numba.parfor.init_prange()
            val = one
            for i in numba.parfor.internal_prange(len(in_arr)):
                val *= in_arr[i]
            return val
    else:
        def prod_1(in_arr):
            numba.parfor.init_prange()
            val = one
            for i in numba.pndindex(in_arr.shape):
                val *= in_arr[i]
            return val
    return prod_1


def mean_parallel_impl(return_type, arg):
    # can't reuse sum since output type is different
    zero = return_type(0)

    if arg.ndim == 1:
        def mean_1(in_arr):
            numba.parfor.init_prange()
            val = zero
            for i in numba.parfor.internal_prange(len(in_arr)):
                val += in_arr[i]
            return val/len(in_arr)
    else:
        def mean_1(in_arr):
            numba.parfor.init_prange()
            val = zero
            for i in numba.pndindex(in_arr.shape):
                val += in_arr[i]
            return val/in_arr.size
    return mean_1

def var_parallel_impl(return_type, arg):

    if arg.ndim == 1:
        def var_1(in_arr):
            # Compute the mean
            m = in_arr.mean()
            # Compute the sum of square diffs
            numba.parfor.init_prange()
            ssd = 0
            for i in numba.parfor.internal_prange(len(in_arr)):
                ssd += (in_arr[i] - m) ** 2
            return ssd / len(in_arr)
    else:
        def var_1(in_arr):
            # Compute the mean
            m = in_arr.mean()
            # Compute the sum of square diffs
            numba.parfor.init_prange()
            ssd = 0
            for i in numba.pndindex(in_arr.shape):
                ssd += (in_arr[i] - m) ** 2
            return ssd / in_arr.size
    return var_1

def std_parallel_impl(return_type, arg):
    def std_1(in_arr):
        return in_arr.var() ** 0.5
    return std_1

def arange_parallel_impl(return_type, *args):
    dtype = as_dtype(return_type.dtype)

    def arange_1(stop):
        return np.arange(0, stop, 1, dtype)

    def arange_2(start, stop):
        return np.arange(start, stop, 1, dtype)

    def arange_3(start, stop, step):
        return np.arange(start, stop, step, dtype)

    if any(isinstance(a, types.Complex) for a in args):
        def arange_4(start, stop, step, dtype):
            numba.parfor.init_prange()
            nitems_c = (stop - start) / step
            nitems_r = math.ceil(nitems_c.real)
            nitems_i = math.ceil(nitems_c.imag)
            nitems = int(max(min(nitems_i, nitems_r), 0))
            arr = np.empty(nitems, dtype)
            for i in numba.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr
    else:
        def arange_4(start, stop, step, dtype):
            numba.parfor.init_prange()
            nitems_r = math.ceil((stop - start) / step)
            nitems = int(max(nitems_r, 0))
            arr = np.empty(nitems, dtype)
            val = start
            for i in numba.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr

    if len(args) == 1:
        return arange_1
    elif len(args) == 2:
        return arange_2
    elif len(args) == 3:
        return arange_3
    elif len(args) == 4:
        return arange_4
    else:
        raise ValueError("parallel arange with types {}".format(args))

def linspace_parallel_impl(return_type, *args):
    dtype = as_dtype(return_type.dtype)

    def linspace_2(start, stop):
        return np.linspace(start, stop, 50)

    def linspace_3(start, stop, num):
        numba.parfor.init_prange()
        arr = np.empty(num, dtype)
        div = num - 1
        delta = stop - start
        arr[0] = start
        for i in numba.parfor.internal_prange(num):
            arr[i] = start + delta * (i / div)
        return arr

    if len(args) == 2:
        return linspace_2
    elif len(args) == 3:
        return linspace_3
    else:
        raise ValueError("parallel linspace with types {}".format(args))

replace_functions_map = {
    ('argmin', 'numpy'): lambda r,a: argmin_parallel_impl,
    ('argmax', 'numpy'): lambda r,a: argmax_parallel_impl,
    ('min', 'numpy'): min_parallel_impl,
    ('max', 'numpy'): max_parallel_impl,
    ('sum', 'numpy'): sum_parallel_impl,
    ('prod', 'numpy'): prod_parallel_impl,
    ('mean', 'numpy'): mean_parallel_impl,
    ('var', 'numpy'): var_parallel_impl,
    ('std', 'numpy'): std_parallel_impl,
    ('dot', 'numpy'): dot_parallel_impl,
    ('arange', 'numpy'): arange_parallel_impl,
    ('linspace', 'numpy'): linspace_parallel_impl,
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
        return ("LoopNest(index_variable = {}, range = ({}, {}, {}))".
                format(self.index_variable, self.start, self.stop, self.step))

    def list_vars(self):
       all_uses = []
       all_uses.append(self.index_variable)
       if isinstance(self.start, ir.Var):
           all_uses.append(self.start)
       if isinstance(self.stop, ir.Var):
           all_uses.append(self.stop)
       if isinstance(self.step, ir.Var):
           all_uses.append(self.step)
       return all_uses

class Parfor(ir.Expr, ir.Stmt):

    id_counter = 0

    def __init__(
            self,
            loop_nests,
            init_block,
            loop_body,
            loc,
            index_var,
            equiv_set,
            pattern,
            flags,
            no_sequential_lowering=False):
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
        # The parallel patterns this parfor was generated from and their options
        # for example, a parfor could be from the stencil pattern with
        # the neighborhood option
        self.patterns = [pattern]
        self.flags = flags
        # if True, this parfor shouldn't be lowered sequentially even with the
        # sequential lowering option
        self.no_sequential_lowering = no_sequential_lowering
        if config.DEBUG_ARRAY_OPT_STATS:
            fmt = 'Parallel for-loop #{} is produced from pattern \'{}\' at {}'
            print(fmt.format(
                  self.id, pattern, loc))

    def __repr__(self):
        return "id=" + str(self.id) + repr(self.loop_nests) + \
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
            all_uses += loop.list_vars()

        for stmt in self.init_block.body:
            all_uses += stmt.list_vars()

        return all_uses

    def get_shape_classes(self, var):
        return self.equiv_set.get_shape_classes(var)

    def dump(self, file=None):
        file = file or sys.stdout
        print(("begin parfor {}".format(self.id)).center(20, '-'), file=file)
        print("index_var = ", self.index_var, file=file)
        for loopnest in self.loop_nests:
            print(loopnest, file=file)
        print("init block:", file=file)
        self.init_block.dump(file)
        for offset, block in sorted(self.loop_body.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file)
        print(("end parfor {}".format(self.id)).center(20, '-'), file=file)

def _analyze_parfor(parfor, equiv_set, typemap, array_analysis):
    """Recursive array analysis for parfor nodes.
    """
    func_ir = array_analysis.func_ir
    parfor_blocks = wrap_parfor_blocks(parfor)
    # Since init_block get label 0 after wrap, we need to save
    # the equivset for the real block label 0.
    backup_equivset = array_analysis.equiv_sets.get(0, None)
    array_analysis.run(parfor_blocks, equiv_set)
    unwrap_parfor_blocks(parfor, parfor_blocks)
    parfor.equiv_set = array_analysis.equiv_sets[0]
    # Restore equivset for block 0 after parfor is unwrapped
    if backup_equivset:
        array_analysis.equiv_sets[0] = backup_equivset
    return [], []

array_analysis.array_analysis_extensions[Parfor] = _analyze_parfor


class PreParforPass(object):
    """Preprocessing for the Parfor pass. It mostly inlines parallel
    implementations of numpy functions if available.
    """
    def __init__(self, func_ir, typemap, calltypes, typingctx, options):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.typingctx = typingctx
        self.options = options

    def run(self):
        """Run pre-parfor processing pass.
        """
        # e.g. convert A.sum() to np.sum(A) for easier match and optimization
        canonicalize_array_math(self.func_ir, self.typemap,
                                self.calltypes, self.typingctx)
        if self.options.numpy:
            self._replace_parallel_functions(self.func_ir.blocks)
        self.func_ir.blocks = simplify_CFG(self.func_ir.blocks)

    def _replace_parallel_functions(self, blocks):
        """
        Replace functions with their parallel implemntation in
        replace_functions_map if available.
        The implementation code is inlined to enable more optimization.
        """
        from numba.inline_closurecall import inline_closure_call
        work_list = list(blocks.items())
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    lhs = instr.target
                    lhs_typ = self.typemap[lhs.name]
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        # Try inline known calls with their parallel implementations
                        def replace_func():
                            func_def = get_definition(self.func_ir, expr.func)
                            callname = find_callname(self.func_ir, expr)
                            repl_func = replace_functions_map.get(callname, None)
                            require(repl_func != None)
                            typs = tuple(self.typemap[x.name] for x in expr.args)
                            try:
                                new_func =  repl_func(lhs_typ, *typs)
                            except:
                                new_func = None
                            require(new_func != None)
                            g = copy.copy(self.func_ir.func_id.func.__globals__)
                            g['numba'] = numba
                            g['np'] = numpy
                            g['math'] = math
                            # inline the parallel implementation
                            inline_closure_call(self.func_ir, g,
                                            block, i, new_func, self.typingctx, typs,
                                            self.typemap, self.calltypes, work_list)
                            return True
                        if guard(replace_func):
                            break
                    elif (isinstance(expr, ir.Expr) and expr.op == 'getattr' and
                          expr.attr == 'dtype'):
                        # Replace getattr call "A.dtype" with the actual type itself.
                        # This helps remove superfulous dependencies from parfor.
                        typ = self.typemap[expr.value.name]
                        if isinstance(typ, types.npytypes.Array):
                            dtype = typ.dtype
                            scope = block.scope
                            loc = instr.loc
                            g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
                            self.typemap[g_np_var.name] = types.misc.Module(numpy)
                            g_np = ir.Global('np', numpy, loc)
                            g_np_assign = ir.Assign(g_np, g_np_var, loc)
                            typ_var = ir.Var(scope, mk_unique_var("$np_typ_var"), loc)
                            self.typemap[typ_var.name] = types.DType(dtype)
                            dtype_str = str(dtype)
                            if dtype_str == 'bool':
                                dtype_str = 'bool_'
                            np_typ_getattr = ir.Expr.getattr(g_np_var, dtype_str, loc)
                            typ_var_assign = ir.Assign(np_typ_getattr, typ_var, loc)
                            instr.value = typ_var
                            block.body.insert(0, typ_var_assign)
                            block.body.insert(0, g_np_assign)
                            break


class ParforPass(object):

    """ParforPass class is responsible for converting Numpy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """

    def __init__(self, func_ir, typemap, calltypes, return_type, typingctx, options, flags):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.typingctx = typingctx
        self.return_type = return_type
        self.options = options
        self.array_analysis = array_analysis.ArrayAnalysis(typingctx, func_ir, typemap,
                                                           calltypes)
        ir_utils._max_label = max(func_ir.blocks.keys())
        self.flags = flags

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR."""
        # run array analysis, a pre-requisite for parfor translation
        remove_dels(self.func_ir.blocks)
        self.array_analysis.run(self.func_ir.blocks)
        # run stencil translation to parfor
        if self.options.stencil:
            stencil_pass = StencilPass(self.func_ir, self.typemap, self.calltypes,
                                            self.array_analysis, self.typingctx, self.flags)
            stencil_pass.run()
        if self.options.setitem:
            self._convert_setitem(self.func_ir.blocks)
        if self.options.numpy:
            self._convert_numpy(self.func_ir.blocks)
        if self.options.reduction:
            self._convert_reduce(self.func_ir.blocks)
        if self.options.prange:
           self._convert_loop(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after parfor pass")

        # simplify CFG of parfor body loops since nested parfors with extra
        # jumps can be created with prange conversion
        simplify_parfor_body_CFG(self.func_ir.blocks)
        # simplify before fusion
        simplify(self.func_ir, self.typemap, self.calltypes)
        # need two rounds of copy propagation to enable fusion of long sequences
        # of parfors like test_fuse_argmin (some PYTHONHASHSEED values since
        # apply_copies_parfor depends on set order for creating dummy assigns)
        simplify(self.func_ir, self.typemap, self.calltypes)

        if self.options.fusion:
            self.func_ir._definitions = build_definitions(self.func_ir.blocks)
            self.array_analysis.equiv_sets = dict()
            self.array_analysis.run(self.func_ir.blocks)
            # reorder statements to maximize fusion
            # push non-parfors down
            maximize_fusion(self.func_ir, self.func_ir.blocks,
                                                            up_direction=False)
            dprint_func_ir(self.func_ir, "after maximize fusion down")
            self.fuse_parfors(self.array_analysis, self.func_ir.blocks)
            # push non-parfors up
            maximize_fusion(self.func_ir, self.func_ir.blocks)
            dprint_func_ir(self.func_ir, "after maximize fusion up")
            # try fuse again after maximize
            self.fuse_parfors(self.array_analysis, self.func_ir.blocks)
            dprint_func_ir(self.func_ir, "after fusion")
        # simplify again
        simplify(self.func_ir, self.typemap, self.calltypes)
        # push function call variables inside parfors so gufunc function
        # wouldn't need function variables as argument
        push_call_vars(self.func_ir.blocks, {}, {})
        # simplify again
        simplify(self.func_ir, self.typemap, self.calltypes)
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
            parfor_ids = get_parfor_params(self.func_ir.blocks, self.options.fusion)
            if config.DEBUG_ARRAY_OPT_STATS:
                name = self.func_ir.func_id.func_qualname
                n_parfors = len(parfor_ids)
                if n_parfors > 0:
                    after_fusion = ("After fusion" if self.options.fusion
                                    else "With fusion disabled")
                    print(('{}, function {} has '
                           '{} parallel for-loop(s) #{}.').format(
                           after_fusion, name, n_parfors, parfor_ids))
                else:
                    print('Function {} has no Parfor.'.format(name))
        return

    def _convert_numpy(self, blocks):
        """
        Convert supported Numpy functions, as well as arrayexpr nodes, to
        parfor nodes.
        """
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
                            if isinstance(instr, tuple):
                                pre_stmts, instr = instr
                                new_body.extend(pre_stmts)
                        elif isinstance(expr, ir.Expr) and expr.op == 'arrayexpr':
                            instr = self._arrayexpr_to_parfor(
                                equiv_set, lhs, expr, avail_vars)
                    avail_vars.append(lhs.name)
                new_body.append(instr)
            block.body = new_body

    def _convert_reduce(self, blocks):
        """
        Find reduce() calls and convert them to parfors.
        """
        topo_order = find_topo_order(blocks)
        for label in topo_order:
            block = blocks[label]
            new_body = []
            equiv_set = self.array_analysis.get_equiv_set(label)
            for instr in block.body:
                parfor = None
                if isinstance(instr, ir.Assign):
                    loc = instr.loc
                    lhs = instr.target
                    expr = instr.value
                    callname = guard(find_callname, self.func_ir, expr)
                    if (callname == ('reduce', 'builtins')
                        or callname == ('reduce', '_functools')):
                        # reduce function with generic function
                        parfor = guard(self._reduce_to_parfor, equiv_set, lhs,
                                       expr.args, loc)
                    if parfor:
                        instr = parfor
                new_body.append(instr)
            block.body = new_body
        return

    def _convert_setitem(self, blocks):
        # convert setitem expressions like A[C] = c or A[C] = B[C] to parfor,
        # where C is a boolean array.
        topo_order = find_topo_order(blocks)
        # variables available in the program so far (used for finding map
        # functions in array_expr lowering)
        avail_vars = []
        for label in topo_order:
            block = blocks[label]
            new_body = []
            equiv_set = self.array_analysis.get_equiv_set(label)
            for instr in block.body:
                if isinstance(instr, ir.StaticSetItem) or isinstance(instr, ir.SetItem):
                    loc = instr.loc
                    target = instr.target
                    index = instr.index if isinstance(instr, ir.SetItem) else instr.index_var
                    value = instr.value
                    target_typ = self.typemap[target.name]
                    index_typ = self.typemap[index.name]
                    value_typ = self.typemap[value.name]
                    if isinstance(target_typ, types.npytypes.Array):
                        if (isinstance(index_typ, types.npytypes.Array) and
                            isinstance(index_typ.dtype, types.Boolean) and
                            target_typ.ndim == index_typ.ndim):
                            if isinstance(value_typ, types.Number):
                                instr = self._setitem_to_parfor(equiv_set,
                                        loc, target, index, value)
                            elif isinstance(value_typ, types.npytypes.Array):
                                val_def = guard(get_definition, self.func_ir,
                                                value.name)
                                if (isinstance(val_def, ir.Expr) and
                                    val_def.op == 'getitem' and
                                    val_def.index.name == index.name):
                                    instr = self._setitem_to_parfor(equiv_set,
                                            loc, target, index, val_def.value)
                        else:
                            shape = equiv_set.get_shape(instr)
                            if shape != None:
                                instr = self._setitem_to_parfor(equiv_set,
                                        loc, target, index, value, shape=shape)
                new_body.append(instr)
            block.body = new_body

    def _convert_loop(self, blocks):
        call_table, _ = get_call_table(blocks)
        cfg = compute_cfg_from_blocks(blocks)
        usedefs = compute_use_defs(blocks)
        live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
        loops = cfg.loops()
        sized_loops = [(loops[k], len(loops[k].body)) for k in loops.keys()]
        moved_blocks = []
        # We go over all loops, smaller loops first (inner first)
        for loop, s in sorted(sized_loops, key=lambda tup: tup[1]):
            if len(loop.entries) != 1 or len(loop.exits) != 1:
                continue
            entry = list(loop.entries)[0]
            for inst in blocks[entry].body:
                # if prange or pndindex call
                if (isinstance(inst, ir.Assign)
                        and isinstance(inst.value, ir.Expr)
                        and inst.value.op == 'call'
                        and self._is_parallel_loop(inst.value.func.name, call_table)):
                    body_labels = [ l for l in loop.body if
                                    l in blocks and l != loop.header ]
                    args = inst.value.args
                    loop_kind = self._get_loop_kind(inst.value.func.name,
                                                                    call_table)
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

                    scope = blocks[entry].scope
                    loc = inst.loc
                    equiv_set = self.array_analysis.get_equiv_set(loop.header)
                    init_block = ir.Block(scope, loc)
                    init_block.body = self._get_prange_init_block(blocks[entry],
                                                            call_table, args)
                    # set l=l for remove dead prange call
                    inst.value = inst.target
                    loop_body = {l: blocks[l] for l in body_labels}
                    # Add an empty block to the end of loop body
                    end_label = next_label()
                    loop_body[end_label] = ir.Block(scope, loc)
                    # replace jumps to header block with the end block
                    for l in body_labels:
                        last_inst = loop_body[l].body[-1]
                        if (isinstance(last_inst, ir.Jump) and
                            last_inst.target == loop.header):
                            last_inst.target = end_label

                    def find_indexed_arrays():
                        """find expressions that involve getitem using the
                        index variable. Return both the arrays and expressions.
                        """
                        indices = copy.copy(loop_index_vars)
                        for block in loop_body.values():
                            for inst in block.find_insts(ir.Assign):
                                if (isinstance(inst.value, ir.Var) and
                                    inst.value.name in indices):
                                    indices.add(inst.target.name)
                        arrs = []
                        exprs = []
                        for block in loop_body.values():
                            for inst in block.body:
                                lv = set(x.name for x in inst.list_vars())
                                if lv & indices:
                                    if lv.issubset(indices):
                                        continue
                                    require(isinstance(inst, ir.Assign))
                                    expr = inst.value
                                    require(isinstance(expr, ir.Expr) and
                                       expr.op in ['getitem', 'static_getitem'])
                                    arrs.append(expr.value.name)
                                    exprs.append(expr)
                        return arrs, exprs

                    mask_var = None
                    mask_indices = None
                    def find_mask_from_size(size_var):
                        """Find the case where size_var is defined by A[M].shape,
                        where M is a boolean array.
                        """
                        size_def = get_definition(self.func_ir, size_var)
                        require(size_def and isinstance(size_def, ir.Expr) and
                                size_def.op == 'getattr' and size_def.attr == 'shape')
                        arr_var = size_def.value
                        live_vars = set.union(*[live_map[l] for l in loop.exits])
                        index_arrs, index_exprs = find_indexed_arrays()
                        require([arr_var.name] == list(index_arrs))
                        # input array has to be dead after loop
                        require(arr_var.name not in live_vars)
                        # loop for arr's definition, where size = arr.shape
                        arr_def = get_definition(self.func_ir, size_def.value)
                        result = self._find_mask(arr_def)
                        # Found the mask.
                        # Replace B[i] with A[i], where B = A[M]
                        for expr in index_exprs:
                            expr.value = result[0]
                        return result

                    # pndindex and prange are provably positive except when
                    # user provides negative start to prange()
                    unsigned_index = True
                    # TODO: support array mask optimization for prange
                    # TODO: refactor and simplify array mask optimization
                    if loop_kind == 'pndindex':
                        assert(equiv_set.has_shape(args[0]))
                        # see if input array to pndindex is output of array
                        # mask like B = A[M]
                        result = guard(find_mask_from_size, args[0])
                        if result:
                            in_arr, mask_var, mask_typ, mask_indices = result
                        else:
                            in_arr = args[0]
                        size_vars = equiv_set.get_shape(in_arr
                                        if mask_indices == None else mask_var)
                        index_vars, loops = self._mk_parfor_loops(
                                                size_vars, scope, loc)
                        orig_index = index_vars
                        if mask_indices:
                            # replace mask indices if required;
                            # integer indices of original array should be used
                            # instead of parfor indices
                            index_vars = tuple(x if x else index_vars[0]
                                               for x in mask_indices)
                        first_body_block = loop_body[min(loop_body.keys())]
                        body_block = ir.Block(scope, loc)
                        index_var, index_var_typ = self._make_index_var(
                                                scope, index_vars, body_block)
                        body = body_block.body + first_body_block.body
                        first_body_block.body = body
                        if mask_indices:
                            orig_index_var = orig_index[0]
                        else:
                            orig_index_var = index_var

                        # if masked array optimization is being applied, create
                        # the branch for array selection
                        if mask_var != None:
                            body_label = next_label()
                            # loop_body needs new labels greater than body_label
                            loop_body = add_offset_to_labels(loop_body,
                                            body_label - min(loop_body.keys()) + 1)
                            labels = loop_body.keys()
                            true_label = min(labels)
                            false_label = max(labels)
                            body_block = ir.Block(scope, loc)
                            loop_body[body_label] = body_block
                            mask = ir.Var(scope, mk_unique_var("$mask_val"), loc)
                            self.typemap[mask.name] = mask_typ
                            mask_val = ir.Expr.getitem(mask_var, orig_index_var, loc)
                            body_block.body.extend([
                               ir.Assign(mask_val, mask, loc),
                               ir.Branch(mask, true_label, false_label, loc)
                            ])
                    else: # prange
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
                        index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
                        # assume user-provided start to prange can be negative
                        # this is the only case parfor can have negative index
                        if isinstance(start, int) and start >= 0:
                            index_var_typ = types.uintp
                        else:
                            index_var_typ = types.intp
                            unsigned_index = False
                        loops = [LoopNest(index_var, start, size_var, step)]
                        self.typemap[index_var.name] = index_var_typ

                    index_var_map = {v: index_var for v in loop_index_vars}
                    replace_vars(loop_body, index_var_map)
                    if unsigned_index:
                        # need to replace signed array access indices to enable
                        # optimizations (see #2846)
                        self._replace_loop_access_indices(
                            loop_body, loop_index_vars, index_var)
                    parfor = Parfor(loops, init_block, loop_body, loc,
                                    orig_index_var if mask_indices else index_var,
                                    equiv_set,
                                    ("prange", loop_kind),
                                    self.flags)
                    # add parfor to entry block's jump target
                    jump = blocks[entry].body[-1]
                    jump.target = list(loop.exits)[0]
                    blocks[jump.target].body.insert(0, parfor)
                    # remove loop blocks from top level dict
                    blocks.pop(loop.header)
                    for l in body_labels:
                        blocks.pop(l)

    def _replace_loop_access_indices(self, loop_body, index_set, new_index):
        """
        Replace array access indices in a loop body with a new index.
        index_set has all the variables that are equivalent to loop index.
        """
        # treat new index like others since replacing it with itself is ok
        index_set.add(new_index.name)

        with dummy_return_in_loop_body(loop_body):
            labels = find_topo_order(loop_body)

        first_label = labels[0]
        added_indices = set()

        # traverse loop body and replace indices in getitem/setitem with
        # new_index if possible.
        # also, find equivalent indices defined in first block.
        for l in labels:
            block = loop_body[l]
            for stmt in block.body:
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Var)):
                    # the first block dominates others so we can use copies
                    # of indices safely
                    if (l == first_label and stmt.value.name in index_set
                            and stmt.target.name not in index_set):
                        index_set.add(stmt.target.name)
                        added_indices.add(stmt.target.name)
                    # make sure parallel index is not overwritten
                    elif stmt.target.name in index_set:
                        raise ValueError(
                            "Overwrite of parallel loop index at {}".format(
                            stmt.target.loc))

                if is_get_setitem(stmt):
                    index = index_var_of_get_setitem(stmt)
                    # statics can have none indices
                    if index is None:
                        continue
                    ind_def = guard(get_definition, self.func_ir,
                                    index, lhs_only=True)
                    if (index.name in index_set
                            or (ind_def is not None
                                and ind_def.name in index_set)):
                        set_index_var_of_get_setitem(stmt, new_index)
                    # corner case where one dimension of a multi-dim access
                    # should be replaced
                    guard(self._replace_multi_dim_ind, ind_def, index_set,
                                                                     new_index)

                if isinstance(stmt, Parfor):
                    self._replace_loop_access_indices(stmt.loop_body, index_set, new_index)

        # remove added indices for currect recursive parfor handling
        index_set -= added_indices
        return

    def _replace_multi_dim_ind(self, ind_var, index_set, new_index):
        """
        replace individual indices in multi-dimensional access variable, which
        is a build_tuple
        """
        require(ind_var is not None)
        # check for Tuple instead of UniTuple since some dims could be slices
        require(isinstance(self.typemap[ind_var.name],
                (types.Tuple, types.UniTuple)))
        ind_def_node = get_definition(self.func_ir, ind_var)
        require(isinstance(ind_def_node, ir.Expr)
                and ind_def_node.op == 'build_tuple')
        ind_def_node.items = [new_index if v.name in index_set else v
                              for v in ind_def_node.items]

    def _find_mask(self, arr_def):
        """check if an array is of B[...M...], where M is a
        boolean array, and other indices (if available) are ints.
        If found, return B, M, M's type, and a tuple representing mask indices.
        Otherwise, raise GuardException.
        """
        require(isinstance(arr_def, ir.Expr) and arr_def.op == 'getitem')
        value = arr_def.value
        index = arr_def.index
        value_typ = self.typemap[value.name]
        index_typ = self.typemap[index.name]
        ndim = value_typ.ndim
        require(isinstance(value_typ, types.npytypes.Array))
        if (isinstance(index_typ, types.npytypes.Array) and
            isinstance(index_typ.dtype, types.Boolean) and
            ndim == index_typ.ndim):
            return value, index, index_typ.dtype, None
        elif isinstance(index_typ, types.BaseTuple):
            # Handle multi-dimension differently by requiring
            # all indices to be constant except the one for mask.
            seq, op = find_build_sequence(self.func_ir, index)
            require(op == 'build_tuple' and len(seq) == ndim)
            count_consts = 0
            mask_indices = []
            mask_var = None
            for ind in seq:
                index_typ = self.typemap[ind.name]
                if (isinstance(index_typ, types.npytypes.Array) and
                    isinstance(index_typ.dtype, types.Boolean)):
                    mask_var = ind
                    mask_typ = index_typ.dtype
                    mask_indices.append(None)
                elif (isinstance(index_typ, types.npytypes.Array) and
                    isinstance(index_typ.dtype, types.Integer)):
                    mask_var = ind
                    mask_typ = index_typ.dtype
                    mask_indices.append(None)
                elif isinstance(index_typ, types.Integer):
                    count_consts += 1
                    mask_indices.append(ind)
            require(mask_var and count_consts == ndim - 1)
            return value, mask_var, mask_typ, mask_indices
        raise GuardException

    def _get_prange_init_block(self, entry_block, call_table, prange_args):
        """
        If there is init_prange, find the code between init_prange and prange
        calls. Remove the code from entry_block and return it.
        """
        init_call_ind = -1
        prange_call_ind = -1
        init_body = []
        for i, inst in enumerate(entry_block.body):
            # if init_prange call
            if (isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr)
                    and inst.value.op == 'call'
                    and self._is_prange_init(inst.value.func.name, call_table)):
                init_call_ind = i
            if (isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr)
                    and inst.value.op == 'call'
                    and self._is_parallel_loop(inst.value.func.name, call_table)):
                prange_call_ind = i
        if init_call_ind != -1 and prange_call_ind != -1:
            # we save instructions that are used to calculate prange call args
            # in the entry block. The rest go to parfor init_block
            arg_related_vars = {v.name for v in prange_args}
            saved_nodes = []
            for i in reversed(range(init_call_ind+1, prange_call_ind)):
                inst = entry_block.body[i]
                inst_vars = {v.name for v in inst.list_vars()}
                if arg_related_vars & inst_vars:
                    arg_related_vars |= inst_vars
                    saved_nodes.append(inst)
                else:
                    init_body.append(inst)

            init_body.reverse()
            saved_nodes.reverse()
            entry_block.body = (entry_block.body[:init_call_ind]
                        + saved_nodes + entry_block.body[prange_call_ind+1:])

        return init_body

    def _is_prange_init(self, func_var, call_table):
        if func_var not in call_table:
            return False
        call = call_table[func_var]
        return len(call) > 0 and (call[0] == 'init_prange' or call[0] == init_prange)

    def _is_parallel_loop(self, func_var, call_table):
        # prange can be either getattr (numba.prange) or global (prange)
        if func_var not in call_table:
            return False
        call = call_table[func_var]
        return len(call) > 0 and (call[0] == 'prange' or call[0] == prange
                or call[0] == 'internal_prange' or call[0] == internal_prange
                or call[0] == 'pndindex' or call[0] == pndindex)

    def _get_loop_kind(self, func_var, call_table):
        """see if prange is user prange or internal"""
        # prange can be either getattr (numba.prange) or global (prange)
        assert func_var in call_table
        call = call_table[func_var]
        assert len(call) > 0
        kind = 'user'
        if call[0] == 'internal_prange' or call[0] == internal_prange:
            kind = 'internal'
        elif call[0] == 'pndindex' or call[0] == pndindex:
            kind = 'pndindex'
        return kind

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
                types.uintp, ndims)
            tuple_call = ir.Expr.build_tuple(list(index_vars), loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            body_block.body.append(tuple_assign)
            return tuple_var, types.containers.UniTuple(types.uintp, ndims)
        elif ndims == 1:
            return index_vars[0], types.uintp
        else:
            raise NotImplementedError(
                "Parfor does not handle arrays of dimension 0")

    def _mk_parfor_loops(self, size_vars, scope, loc):
        """
        Create loop index variables and build LoopNest objects for a parfor.
        """
        loopnests = []
        index_vars = []
        for size_var in size_vars:
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
            index_vars.append(index_var)
            self.typemap[index_var.name] = types.uintp
            loopnests.append(LoopNest(index_var, 0, size_var, 1))
        return index_vars, loopnests

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
        size_vars = equiv_set.get_shape(lhs)
        index_vars, loopnests = self._mk_parfor_loops(size_vars, scope, loc)

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
                self.typingctx,
                self.typemap,
                self.calltypes,
                equiv_set,
                init_block,
                expr_out_var,
                expr,
                index_var,
                index_vars,
                avail_vars))

        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set,
                        ('arrayexpr {}'.format(repr_arrayexpr(arrayexpr.expr)),),
                        self.flags)

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        self.calltypes[setitem_node] = signature(
            types.none, self.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT == 1:
            parfor.dump()
        return parfor

    def _setitem_to_parfor(self, equiv_set, loc, target, index, value, shape=None):
        """generate parfor from setitem node with a boolean or slice array indices.
        The value can be either a scalar or an array variable, and if a boolean index
        is used for the latter case, the same index must be used for the value too.
        """
        scope = target.scope
        arr_typ = self.typemap[target.name]
        el_typ = arr_typ.dtype
        index_typ = self.typemap[index.name]
        init_block = ir.Block(scope, loc)

        if shape:
            # Slice index is being used on the target array, we'll have to create
            # a sub-array so that the target dimension matches the given shape.
            assert(isinstance(index_typ, types.BaseTuple) or
                   isinstance(index_typ, types.SliceType))
            # setitem has a custom target shape
            size_vars = shape
            # create a new target array via getitem
            subarr_var = ir.Var(scope, mk_unique_var("$subarr"), loc)
            getitem_call = ir.Expr.getitem(target, index, loc)
            subarr_typ = typing.arraydecl.get_array_index_type( arr_typ, index_typ).result
            self.typemap[subarr_var.name] = subarr_typ
            self.calltypes[getitem_call] = signature(subarr_typ, arr_typ,
                                                     index_typ)
            init_block.append(ir.Assign(getitem_call, subarr_var, loc))
            target = subarr_var
        else:
            # Otherwise it is a boolean array that is used as index.
            assert(isinstance(index_typ, types.ArrayCompatible))
            size_vars = equiv_set.get_shape(target)
            bool_typ = index_typ.dtype


        # generate loopnests and size variables from lhs correlations
        loopnests = []
        index_vars = []
        for size_var in size_vars:
            index_var = ir.Var(scope, mk_unique_var("parfor_index"), loc)
            index_vars.append(index_var)
            self.typemap[index_var.name] = types.uintp
            loopnests.append(LoopNest(index_var, 0, size_var, 1))

        # generate body
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        index_var, index_var_typ = self._make_index_var(
                 scope, index_vars, body_block)
        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set,
                        ('setitem',), self.flags)
        if shape:
            # slice subarray
            parfor.loop_body = {body_label: body_block}
            true_block = body_block
            end_label = None
        else:
            # boolean mask
            true_label = next_label()
            true_block = ir.Block(scope, loc)
            end_label = next_label()
            end_block = ir.Block(scope, loc)
            parfor.loop_body = {body_label: body_block,
                                true_label: true_block,
                                end_label:  end_block,
                                }
            mask_var = ir.Var(scope, mk_unique_var("$mask_var"), loc)
            self.typemap[mask_var.name] = bool_typ
            mask_val = ir.Expr.getitem(index, index_var, loc)
            body_block.body.extend([
               ir.Assign(mask_val, mask_var, loc),
               ir.Branch(mask_var, true_label, end_label, loc)
            ])

        value_typ = self.typemap[value.name]
        if isinstance(value_typ, types.npytypes.Array):
            value_var = ir.Var(scope, mk_unique_var("$value_var"), loc)
            self.typemap[value_var.name] = value_typ.dtype
            getitem_call = ir.Expr.getitem(value, index_var, loc)
            self.calltypes[getitem_call] = signature(
                value_typ.dtype, value_typ, index_var_typ)
            true_block.body.append(ir.Assign(getitem_call, value_var, loc))
        else:
            value_var = value
        setitem_node = ir.SetItem(target, index_var, value_var, loc)
        self.calltypes[setitem_node] = signature(
            types.none, self.typemap[target.name], index_var_typ, el_typ)
        true_block.body.append(setitem_node)
        if end_label:
            true_block.body.append(ir.Jump(end_label, loc))

        if config.DEBUG_ARRAY_OPT == 1:
            parfor.dump()
        return parfor

    def _is_supported_npycall(self, expr):
        """check if we support parfor translation for
        this Numpy call.
        """
        call_name, mod_name = find_callname(self.func_ir, expr)
        if not (isinstance(mod_name, str) and mod_name.startswith('numpy')):
            return False
        if call_name in ['zeros', 'ones']:
            return True
        if call_name in ['arange', 'linspace']:
            return True
        if mod_name == 'numpy.random' and call_name in random_calls:
            return True
        # TODO: add more calls
        return False

    def _get_ndims(self, arr):
        # return len(self.array_analysis.array_shape_classes[arr])
        return self.typemap[arr].ndim

    def _numpy_to_parfor(self, equiv_set, lhs, expr):
        call_name, mod_name = find_callname(self.func_ir, expr)
        args = expr.args
        kws = dict(expr.kws)
        if call_name in ['zeros', 'ones'] or mod_name == 'numpy.random':
            return self._numpy_map_to_parfor(equiv_set, call_name, lhs, args, kws, expr)
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
        size_vars = equiv_set.get_shape(lhs)
        index_vars, loopnests = self._mk_parfor_loops(size_vars, scope, loc)

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
            value = ir.Const(el_typ(0), loc)
        elif call_name == 'ones':
            value = ir.Const(el_typ(1), loc)
        elif call_name in random_calls:
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

        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set,
                        ('{} function'.format(call_name,)), self.flags)

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        self.calltypes[setitem_node] = signature(
            types.none, self.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT == 1:
            print("generated parfor for numpy map:")
            parfor.dump()
        return parfor

    def _mk_reduction_body(self, call_name, scope, loc,
                           index_vars, in_arr, acc_var):
        """
        Produce the body blocks for a reduction function indicated by call_name.
        """
        from numba.inline_closurecall import check_reduce_func
        reduce_func = get_definition(self.func_ir, call_name)
        check_reduce_func(self.func_ir, reduce_func)

        arr_typ = self.typemap[in_arr.name]
        in_typ = arr_typ.dtype
        body_block = ir.Block(scope, loc)
        index_var, index_var_type = self._make_index_var(
            scope, index_vars, body_block)

        tmp_var = ir.Var(scope, mk_unique_var("$val"), loc)
        self.typemap[tmp_var.name] = in_typ
        getitem_call = ir.Expr.getitem(in_arr, index_var, loc)
        self.calltypes[getitem_call] = signature(
            in_typ, arr_typ, index_var_type)
        body_block.append(ir.Assign(getitem_call, tmp_var, loc))

        reduce_f_ir = compile_to_numba_ir(reduce_func,
                                        self.func_ir.func_id.func.__globals__,
                                        self.typingctx,
                                        (in_typ, in_typ),
                                        self.typemap,
                                        self.calltypes)
        loop_body = reduce_f_ir.blocks
        end_label = next_label()
        end_block = ir.Block(scope, loc)
        loop_body[end_label] = end_block
        first_reduce_label = min(reduce_f_ir.blocks.keys())
        first_reduce_block = reduce_f_ir.blocks[first_reduce_label]
        body_block.body.extend(first_reduce_block.body)
        first_reduce_block.body = body_block.body
        replace_arg_nodes(first_reduce_block, [acc_var, tmp_var])
        replace_returns(loop_body, acc_var, end_label)
        return index_var, loop_body

    def _reduce_to_parfor(self, equiv_set, lhs, args, loc):
        """
        Convert a reduce call to a parfor.
        The call arguments should be (call_name, array, init_value).
        """
        scope = lhs.scope
        call_name = args[0]
        in_arr = args[1]
        arr_def = get_definition(self.func_ir, in_arr.name)

        mask_var = None
        mask_indices = None
        result = guard(self._find_mask, arr_def)
        if result:
            in_arr, mask_var, mask_typ, mask_indices = result

        init_val = args[2]
        size_vars = equiv_set.get_shape(in_arr if mask_indices == None else mask_var)
        index_vars, loopnests = self._mk_parfor_loops(size_vars, scope, loc)
        mask_index = index_vars
        if mask_indices:
            index_vars = tuple(x if x else index_vars[0] for x in mask_indices)
        acc_var = lhs

        # init block has to init the reduction variable
        init_block = ir.Block(scope, loc)
        init_block.body.append(ir.Assign(init_val, acc_var, loc))

        # produce loop body
        body_label = next_label()
        index_var, loop_body = self._mk_reduction_body(call_name,
                                scope, loc, index_vars, in_arr, acc_var)
        if mask_indices:
            index_var = mask_index[0]

        if mask_var != None:
            true_label = min(loop_body.keys())
            false_label = max(loop_body.keys())
            body_block = ir.Block(scope, loc)
            loop_body[body_label] = body_block
            mask = ir.Var(scope, mk_unique_var("$mask_val"), loc)
            self.typemap[mask.name] = mask_typ
            mask_val = ir.Expr.getitem(mask_var, index_var, loc)
            body_block.body.extend([
               ir.Assign(mask_val, mask, loc),
               ir.Branch(mask, true_label, false_label, loc)
            ])

        parfor = Parfor(loopnests, init_block, loop_body, loc, index_var,
                        equiv_set, ('{} function'.format(call_name),), self.flags)
        return parfor


    def fuse_parfors(self, array_analysis, blocks):
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
                        # we have to update equiv_set since they have changed due to
                        # variables being renamed before fusion.
                        equiv_set = array_analysis.get_equiv_set(label)
                        stmt.equiv_set = equiv_set
                        next_stmt.equiv_set = equiv_set
                        fused_node = try_fuse(equiv_set, stmt, next_stmt)
                        if fused_node is not None:
                            fusion_happened = True
                            new_body.append(fused_node)
                            self.fuse_recursive_parfor(fused_node, equiv_set)
                            i += 2
                            continue
                    new_body.append(stmt)
                    if isinstance(stmt, Parfor):
                        self.fuse_recursive_parfor(stmt, equiv_set)
                    i += 1
                new_body.append(block.body[-1])
                block.body = new_body
        return

    def fuse_recursive_parfor(self, parfor, equiv_set):
        blocks = wrap_parfor_blocks(parfor)
        # print("in fuse_recursive parfor for ", parfor.id)
        maximize_fusion(self.func_ir, blocks)
        arr_analysis = array_analysis.ArrayAnalysis(self.typingctx, self.func_ir,
                                                self.typemap, self.calltypes)
        arr_analysis.run(blocks, equiv_set)
        self.fuse_parfors(arr_analysis, blocks)
        unwrap_parfor_blocks(parfor)

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

    if call_name == 'randint':
        # has 4 args, 3rd one is size
        if len(expr.args) == 3:
            expr.args.pop()
        if len(expr.args) == 4:
            dt_arg = expr.args.pop()
            expr.args.pop()  # remove size
            expr.args.append(dt_arg)

    if call_name == 'triangular':
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


def _arrayexpr_tree_to_ir(
        func_ir,
        typingctx,
        typemap,
        calltypes,
        equiv_set,
        init_block,
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
                                            typingctx,
                                            typemap,
                                            calltypes,
                                            equiv_set,
                                            init_block,
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
                func_var_name = _find_func_var(typemap, op, avail_vars)
                func_var = ir.Var(scope, mk_unique_var(func_var_name), loc)
                typemap[func_var.name] = typemap[func_var_name]
                func_var_def = func_ir.get_definition(func_var_name)
                if isinstance(func_var_def, ir.Expr) and func_var_def.op == 'getattr' and func_var_def.attr == 'sqrt':
                     g_math_var = ir.Var(scope, mk_unique_var("$math_g_var"), loc)
                     typemap[g_math_var.name] = types.misc.Module(math)
                     g_math = ir.Global('math', math, loc)
                     g_math_assign = ir.Assign(g_math, g_math_var, loc)
                     func_var_def = ir.Expr.getattr(g_math_var, 'sqrt', loc)
                     out_ir.append(g_math_assign)
#                     out_ir.append(func_var_def)
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
                typingctx,
                typemap,
                init_block,
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
        typingctx,
        typemap,
        init_block,
        out_ir):
    """if there is implicit dimension broadcast, generate proper access variable
    for getitem. For example, if indices are (i1,i2,i3) but shape is (c1,0,c3),
    generate a tuple with (i1,0,i3) for access.  Another example: for (i1,i2,i3)
    and (c1,c2) generate (i2,i3).
    """
    loc = var.loc
    index_var = parfor_index_tuple_var
    var_typ =  typemap[var.name]
    ndims = typemap[var.name].ndim
    num_indices = len(all_parfor_indices)
    size_vars = equiv_set.get_shape(var) or []
    size_consts = [equiv_set.get_equiv_const(x) for x in size_vars]
    if ndims == 0:
        # call np.ravel
        ravel_var = ir.Var(var.scope, mk_unique_var("$ravel"), loc)
        ravel_typ = types.npytypes.Array(dtype=var_typ.dtype, ndim=1, layout='C')
        typemap[ravel_var.name] = ravel_typ
        stmts = ir_utils.gen_np_call('ravel', numpy.ravel, ravel_var, [var], typingctx, typemap, calltypes)
        init_block.body.extend(stmts)
        var = ravel_var
        # Const(0)
        const_node = ir.Const(0, var.loc)
        const_var = ir.Var(var.scope, mk_unique_var("$const_ind_0"), loc)
        typemap[const_var.name] = types.uintp
        const_assign = ir.Assign(const_node, const_var, loc)
        out_ir.append(const_assign)
        index_var = const_var
    elif ndims == 1:
        # Use last index for 1D arrays
        index_var = all_parfor_indices[-1]
    elif any([x != None for x in size_consts]):
        # Need a tuple as index
        ind_offset = num_indices - ndims
        tuple_var = ir.Var(var.scope, mk_unique_var(
            "$parfor_index_tuple_var_bcast"), loc)
        typemap[tuple_var.name] = types.containers.UniTuple(types.uintp, ndims)
        # Just in case, const var for size 1 dim access index: $const0 =
        # Const(0)
        const_node = ir.Const(0, var.loc)
        const_var = ir.Var(var.scope, mk_unique_var("$const_ind_0"), loc)
        typemap[const_var.name] = types.uintp
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
    ir_utils._max_label = max(ir_utils._max_label,
                              ir_utils.find_max_label(func_ir.blocks))
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
    # add dels since simplify removes dels
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run()
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
        if isinstance(inst, Parfor) and not inst.no_sequential_lowering:
            return i
    return -1


def get_parfor_params(blocks, options_fusion):
    """find variables used in body of parfors from outside and save them.
    computed as live variables at entry of first block.
    """

    # since parfor wrap creates a back-edge to first non-init basic block,
    # live_map[first_non_init_block] contains variables defined in parfor body
    # that could be undefined before. So we only consider variables that are
    # actually defined before the parfor body in the program.
    parfor_ids = set()
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
            parfor.params = get_parfor_params_inner(parfor, pre_defs, options_fusion)
            parfor_ids.add(parfor.id)

        pre_defs |= all_defs[label]

    return parfor_ids


def get_parfor_params_inner(parfor, pre_defs, options_fusion):

    blocks = wrap_parfor_blocks(parfor)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    parfor_ids = get_parfor_params(blocks, options_fusion)
    if config.DEBUG_ARRAY_OPT_STATS:
        n_parfors = len(parfor_ids)
        if n_parfors > 0:
             after_fusion = ("After fusion" if options_fusion
                             else "With fusion disabled")
             print(('After fusion, parallel for-loop {} has '
                   '{} nested Parfor(s) #{}.').format(
                  after_fusion, parfor.id, n_parfors, parfor_ids))
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

def get_parfor_reductions(parfor, parfor_params, calltypes, reductions=None,
        reduce_varnames=None, param_uses=None, param_nodes=None,
        var_to_param=None):
    """find variables that are updated using their previous values and an array
    item accessed with parfor index, e.g. s = s+A[i]
    """
    if reductions is None:
        reductions = {}
    if reduce_varnames is None:
        reduce_varnames = []

    # for each param variable, find what other variables are used to update it
    # also, keep the related nodes
    if param_uses is None:
        param_uses = defaultdict(list)
    if param_nodes is None:
        param_nodes = defaultdict(list)
    if var_to_param is None:
        var_to_param = {}

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    for label in reversed(topo_order):
        for stmt in reversed(parfor.loop_body[label].body):
            if (isinstance(stmt, ir.Assign)
                    and (stmt.target.name in parfor_params
                        or stmt.target.name in var_to_param)):
                lhs = stmt.target.name
                rhs = stmt.value
                cur_param = lhs if lhs in parfor_params else var_to_param[lhs]
                used_vars = []
                if isinstance(rhs, ir.Var):
                    used_vars = [rhs.name]
                elif isinstance(rhs, ir.Expr):
                    used_vars = [v.name for v in stmt.value.list_vars()]
                param_uses[cur_param].extend(used_vars)
                for v in used_vars:
                    var_to_param[v] = cur_param
                # save copy of dependent stmt
                stmt_cp = copy.deepcopy(stmt)
                if stmt.value in calltypes:
                    calltypes[stmt_cp.value] = calltypes[stmt.value]
                param_nodes[cur_param].append(stmt_cp)
            if isinstance(stmt, Parfor):
                # recursive parfors can have reductions like test_prange8
                get_parfor_reductions(stmt, parfor_params, calltypes,
                    reductions, reduce_varnames, param_uses, param_nodes, var_to_param)
    for param, used_vars in param_uses.items():
        # a parameter is a reduction variable if its value is used to update it
        # check reduce_varnames since recursive parfors might have processed
        # param already
        if param in used_vars and param not in reduce_varnames:
            reduce_varnames.append(param)
            param_nodes[param].reverse()
            reduce_nodes = get_reduce_nodes(param, param_nodes[param])
            init_val = guard(get_reduction_init, reduce_nodes)
            reductions[param] = (init_val, reduce_nodes)
    return reduce_varnames, reductions

def get_reduction_init(nodes):
    """
    Get initial value for known reductions.
    Currently, only += and *= are supported. We assume the inplace_binop node
    is followed by an assignment.
    """
    require(len(nodes) >=2)
    require(isinstance(nodes[-1].value, ir.Var))
    require(nodes[-2].target.name == nodes[-1].value.name)
    acc_expr = nodes[-2].value
    require(isinstance(acc_expr, ir.Expr) and acc_expr.op=='inplace_binop')
    if acc_expr.fn == '+=':
        return 0
    if acc_expr.fn == '*=':
        return 1
    return None

def get_reduce_nodes(name, nodes):
    """
    Get nodes that combine the reduction variable with a sentinel variable.
    Recognizes the first node that combines the reduction variable with another
    variable.
    """
    reduce_nodes = None
    defs = {}

    def lookup(var, varonly=True):
        val = defs.get(var.name, None)
        if isinstance(val, ir.Var):
            return lookup(val)
        else:
            return var if (varonly or val == None) else val

    for i, stmt in enumerate(nodes):
        lhs = stmt.target
        rhs = stmt.value
        defs[lhs.name] = rhs
        if isinstance(rhs, ir.Var) and rhs.name in defs:
            rhs = lookup(rhs)
        if isinstance(rhs, ir.Expr):
            in_vars = set(lookup(v, True).name for v in rhs.list_vars())
            if name in in_vars:
                args = [ (x.name, lookup(x, True)) for x in get_expr_args(rhs) ]
                non_red_args = [ x for (x, y) in args if y.name != name ]
                assert len(non_red_args) == 1
                args = [ (x, y) for (x, y) in args if x != y.name ]
                replace_dict = dict(args)
                replace_dict[non_red_args[0]] = ir.Var(lhs.scope, name+"#init", lhs.loc)
                replace_vars_inner(rhs, replace_dict)
                reduce_nodes = nodes[i:]
                break;
    assert reduce_nodes, "Invalid reduction format"
    return reduce_nodes

def get_expr_args(expr):
    """
    Get arguments of an expression node
    """
    if expr.op in ['binop', 'inplace_binop']:
        return [expr.lhs, expr.rhs]
    if expr.op == 'call':
        return [v for v in expr.args]
    raise NotImplementedError("get arguments for expression {}".format(expr))

def visit_parfor_pattern_vars(parfor, callback, cbdata):
    # currently, only stencil pattern has variables
    for pattern in parfor.patterns:
        if pattern[0] == 'stencil':
            left_lengths = pattern[1][0]
            for i in range(len(left_lengths)):
                if isinstance(left_lengths[i], ir.Var):
                    left_lengths[i] = visit_vars_inner(left_lengths[i],
                                                            callback, cbdata)
            right_lengths = pattern[1][1]
            for i in range(len(right_lengths)):
                if isinstance(right_lengths[i], ir.Var):
                    right_lengths[i] = visit_vars_inner(right_lengths[i],
                                                            callback, cbdata)

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
    visit_parfor_pattern_vars(parfor, callback, cbdata)
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
    use_set |= get_parfor_pattern_vars(parfor)

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


def maximize_fusion(func_ir, blocks, up_direction=True):
    """
    Reorder statements to maximize parfor fusion. Push all parfors up or down
    so they are adjacent.
    """
    call_table, _ = get_call_table(blocks)
    for block in blocks.values():
        order_changed = True
        while order_changed:
            order_changed = maximize_fusion_inner(func_ir, block,
                                                    call_table, up_direction)

def maximize_fusion_inner(func_ir, block, call_table, up_direction=True):
    order_changed = False
    i = 0
    # i goes to body[-3] (i+1 to body[-2]) since body[-1] is terminator and
    # shouldn't be reordered
    while i < len(block.body) - 2:
        stmt = block.body[i]
        next_stmt = block.body[i+1]
        can_reorder = (_can_reorder_stmts(stmt, next_stmt, func_ir, call_table)
                        if up_direction else _can_reorder_stmts(next_stmt, stmt,
                        func_ir, call_table))
        if can_reorder:
            block.body[i] = next_stmt
            block.body[i+1] = stmt
            order_changed = True
        i += 1
    return order_changed

def _can_reorder_stmts(stmt, next_stmt, func_ir, call_table):
    """
    Check dependencies to determine if a parfor can be reordered in the IR block
    with a non-parfor statement.
    """
    # swap only parfors with non-parfors
    # don't reorder calls with side effects (e.g. file close)
    # only read-read dependencies are OK
    # make sure there is no write-write, write-read dependencies
    if (isinstance(
            stmt, Parfor) and not isinstance(
            next_stmt, Parfor) and not isinstance(
            next_stmt, ir.Print)
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
            return True
    return False

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

    # find parfor1's defs, only body is considered since init_block will run
    # first after fusion as well
    p1_body_usedefs = compute_use_defs(parfor1.loop_body)
    p1_body_defs = set()
    for defs in p1_body_usedefs.defmap.values():
        p1_body_defs |= defs

    p2_usedefs = compute_use_defs(parfor2.loop_body)
    p2_uses = compute_use_defs({0: parfor2.init_block}).usemap[0]
    for uses in p2_usedefs.usemap.values():
        p2_uses |= uses

    if not p1_body_defs.isdisjoint(p2_uses):
        dprint("try_fuse parfor2 depends on parfor1 body")
        return None

    return fuse_parfors_inner(parfor1, parfor2)


def fuse_parfors_inner(parfor1, parfor2):
    # fuse parfor2 into parfor1
    # append parfor2's init block on parfor1's
    parfor1.init_block.body.extend(parfor2.init_block.body)

    # append parfor2's first block to parfor1's last block
    parfor2_first_label = min(parfor2.loop_body.keys())
    parfor2_first_block = parfor2.loop_body[parfor2_first_label].body
    parfor1_first_label = min(parfor1.loop_body.keys())
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

    # re-order labels from min to max
    blocks = wrap_parfor_blocks(parfor1, entry_label=parfor1_first_label)
    blocks = rename_labels(blocks)
    unwrap_parfor_blocks(parfor1, blocks)

    nameset = set(x.name for x in index_dict.values())
    remove_duplicate_definitions(parfor1.loop_body, nameset)
    parfor1.patterns.extend(parfor2.patterns)
    if config.DEBUG_ARRAY_OPT_STATS:
        print('Parallel for-loop #{} is fused into for-loop #{}.'.format(
              parfor2.id, parfor1.id))

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


def get_parfor_pattern_vars(parfor):
    """ get the variables used in parfor pattern information
    """
    out = set()
    # currently, only stencil pattern has variables
    for pattern in parfor.patterns:
        if pattern[0] == 'stencil':
            left_lengths = pattern[1][0]
            right_lengths = pattern[1][1]
            for v in left_lengths+right_lengths:
                if isinstance(v, ir.Var):
                    out.add(v.name)
    return out

def remove_dead_parfor(parfor, lives, arg_aliases, alias_map, func_ir, typemap):
    """ remove dead code inside parfor including get/sets
    """

    with dummy_return_in_loop_body(parfor.loop_body):
        labels = find_topo_order(parfor.loop_body)

    # get/setitem replacement should ideally use dataflow to propagate setitem
    # saved values, but for simplicity we handle the common case of propagating
    # setitems in the first block (which is dominant) if the array is not
    # potentially changed in any way
    first_label = labels[0]
    first_block_saved_values = {}
    _update_parfor_get_setitems(
        parfor.loop_body[first_label].body,
        parfor.index_var, alias_map,
        first_block_saved_values,
        lives
        )

    # remove saved first block setitems if array potentially changed later
    saved_arrs = set(first_block_saved_values.keys())
    for l in labels:
        if l == first_label:
            continue
        for stmt in parfor.loop_body[l].body:
            if (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == 'getitem'
                    and stmt.value.index.name == parfor.index_var.name):
                continue
            varnames = set(v.name for v in stmt.list_vars())
            rm_arrs = varnames & saved_arrs
            for a in rm_arrs:
                first_block_saved_values.pop(a, None)


    # replace getitems with available value
    # e.g. A[i] = v; ... s = A[i]  ->  s = v
    for l in labels:
        if l == first_label:
            continue
        block = parfor.loop_body[l]
        saved_values = first_block_saved_values.copy()
        _update_parfor_get_setitems(block.body, parfor.index_var, alias_map,
                                        saved_values, lives)


    # after getitem replacement, remove extra setitems
    blocks = parfor.loop_body.copy()  # shallow copy is enough
    last_label = max(blocks.keys())
    return_label, tuple_var = _add_liveness_return_block(blocks, lives, typemap)
    # jump to return label
    jump = ir.Jump(return_label, ir.Loc("parfors_dummy", -1))
    blocks[last_label].body.append(jump)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    alias_set = set(alias_map.keys())

    for label, block in blocks.items():
        new_body = []
        in_lives = {v.name for v in block.terminator.list_vars()}
        # find live variables at the end of block
        for out_blk, _data in cfg.successors(label):
            in_lives |= live_map[out_blk]
        for stmt in reversed(block.body):
            # aliases of lives are also live for setitems
            alias_lives = in_lives & alias_set
            for v in alias_lives:
                in_lives |= alias_map[v]
            if (isinstance(stmt, ir.SetItem) and stmt.index.name ==
                    parfor.index_var.name and stmt.target.name not in in_lives and
                    stmt.target.name not in arg_aliases):
                continue
            in_lives |= {v.name for v in stmt.list_vars()}
            new_body.append(stmt)
        new_body.reverse()
        block.body = new_body

    typemap.pop(tuple_var.name)  # remove dummy tuple type
    blocks[last_label].body.pop()  # remove jump


    # process parfor body recursively
    remove_dead_parfor_recursive(
        parfor, lives, arg_aliases, alias_map, func_ir, typemap)

    # remove parfor if empty
    is_empty = len(parfor.init_block.body) == 0
    for block in parfor.loop_body.values():
        is_empty &= len(block.body) == 0
    if is_empty:
        return None
    return parfor

def _update_parfor_get_setitems(block_body, index_var, alias_map,
                                  saved_values, lives):
    """
    replace getitems of a previously set array in a block of parfor loop body
    """
    for stmt in block_body:
        if (isinstance(stmt, ir.SetItem) and stmt.index.name ==
                index_var.name and stmt.target.name not in lives):
            # saved values of aliases of SetItem target array are invalid
            for w in alias_map.get(stmt.target.name, []):
                saved_values.pop(w, None)
            # set saved value after invalidation since alias_map may
            # contain the array itself (e.g. pi example)
            saved_values[stmt.target.name] = stmt.value
            continue
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
            rhs = stmt.value
            if rhs.op == 'getitem' and isinstance(rhs.index, ir.Var):
                if rhs.index.name == index_var.name:
                    # replace getitem if value saved
                    stmt.value = saved_values.get(rhs.value.name, rhs)
                    continue
        # conservative assumption: array is modified if referenced
        # remove all referenced arrays
        for v in stmt.list_vars():
            saved_values.pop(v.name, None)
            # aliases are potentially modified as well
            for w in alias_map.get(v.name, []):
                saved_values.pop(w, None)

    return

ir_utils.remove_dead_extensions[Parfor] = remove_dead_parfor


def remove_dead_parfor_recursive(parfor, lives, arg_aliases, alias_map,
                                                             func_ir, typemap):
    """create a dummy function from parfor and call remove dead recursively
    """
    blocks = parfor.loop_body.copy()  # shallow copy is enough
    first_body_block = min(blocks.keys())
    assert first_body_block > 0  # we are using 0 for init block here
    last_label = max(blocks.keys())

    return_label, tuple_var = _add_liveness_return_block(blocks, lives, typemap)

    # branch back to first body label to simulate loop
    branch = ir.Branch(0, first_body_block, return_label, ir.Loc("parfors_dummy", -1))
    blocks[last_label].body.append(branch)

    # add dummy jump in init_block for CFG to work
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(first_body_block, ir.Loc("parfors_dummy", -1)))

    # args var including aliases is ok
    remove_dead(blocks, arg_aliases, func_ir, typemap, alias_map, arg_aliases)
    typemap.pop(tuple_var.name)  # remove dummy tuple type
    blocks[0].body.pop()  # remove dummy jump
    blocks[last_label].body.pop()  # remove branch
    return

def _add_liveness_return_block(blocks, lives, typemap):
    last_label = max(blocks.keys())
    return_label = last_label + 1

    loc = blocks[last_label].loc
    scope = blocks[last_label].scope
    blocks[return_label] = ir.Block(scope, loc)

    # add lives in a dummpy return to last block to avoid their removal
    tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
    # dummy type for tuple_var
    typemap[tuple_var.name] = types.containers.UniTuple(
        types.uintp, 2)
    live_vars = [ir.Var(scope, v, loc) for v in lives]
    tuple_call = ir.Expr.build_tuple(live_vars, loc)
    blocks[return_label].body.append(ir.Assign(tuple_call, tuple_var, loc))
    blocks[return_label].body.append(ir.Return(tuple_var, loc))
    return return_label, tuple_var


def find_potential_aliases_parfor(parfor, args, typemap, func_ir, alias_map, arg_aliases):
    blocks = wrap_parfor_blocks(parfor)
    ir_utils.find_potential_aliases(
        blocks, args, typemap, func_ir, alias_map, arg_aliases)
    unwrap_parfor_blocks(parfor)
    return

ir_utils.alias_analysis_extensions[Parfor] = find_potential_aliases_parfor

def simplify_parfor_body_CFG(blocks):
    """simplify CFG of body loops in parfors"""
    for block in blocks.values():
        for stmt in block.body:
            if isinstance(stmt, Parfor):
                parfor = stmt
                # add dummy return to enable CFG creation
                # can't use dummy_return_in_loop_body since body changes
                last_block = parfor.loop_body[max(parfor.loop_body.keys())]
                last_block.body.append(ir.Return(0, ir.Loc("parfors_dummy", -1)))
                parfor.loop_body = simplify_CFG(parfor.loop_body)
                last_block = parfor.loop_body[max(parfor.loop_body.keys())]
                last_block.body.pop()
                # call on body recursively
                simplify_parfor_body_CFG(parfor.loop_body)


def wrap_parfor_blocks(parfor, entry_label = None):
    """wrap parfor blocks for analysis/optimization like CFG"""
    blocks = parfor.loop_body.copy()  # shallow copy is enough
    if entry_label == None:
        entry_label = min(blocks.keys())
    assert entry_label > 0  # we are using 0 for init block here

    # add dummy jump in init_block for CFG to work
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(entry_label, blocks[0].loc))
    for block in blocks.values():
        if len(block.body) == 0 or (not block.body[-1].is_terminator):
            block.body.append(ir.Jump(entry_label, block.loc))
    return blocks


def unwrap_parfor_blocks(parfor, blocks=None):
    """
    unwrap parfor blocks after analysis/optimization.
    Allows changes to the parfor loop.
    """
    if blocks is not None:
        # make sure init block isn't removed
        init_block_label = min(blocks.keys())
        # update loop body blocks
        blocks.pop(init_block_label)
        parfor.loop_body = blocks

    # make sure dummy jump to loop body isn't altered
    first_body_label = min(parfor.loop_body.keys())
    assert isinstance(parfor.init_block.body[-1], ir.Jump)

    # remove dummy jump to loop body
    parfor.init_block.body.pop()

    # make sure dummy jump back to loop body isn't altered
    for block in parfor.loop_body.values():
        if (isinstance(block.body[-1], ir.Jump) and
            block.body[-1].target == first_body_label):
            # remove dummy jump back to loop
            block.body.pop()
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


def apply_copies_parfor(parfor, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate recursively in parfor"""
    # replace variables in pattern metadata like stencil neighborhood
    for i, pattern in enumerate(parfor.patterns):
        if pattern[0] == 'stencil':
            parfor.patterns[i] = ('stencil',
                replace_vars_inner(pattern[1], var_dict))

    # replace loop boundary variables
    for l in parfor.loop_nests:
        l.start = replace_vars_inner(l.start, var_dict)
        l.stop = replace_vars_inner(l.stop, var_dict)
        l.step = replace_vars_inner(l.step, var_dict)

    blocks = wrap_parfor_blocks(parfor)
    # add dummy assigns for each copy
    assign_list = []
    for lhs_name, rhs in var_dict.items():
        assign_list.append(ir.Assign(rhs, name_var_table[lhs_name],
                                     ir.Loc("dummy", -1)))
    blocks[0].body = assign_list + blocks[0].body
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks, typemap)
    apply_copy_propagate(blocks, in_copies_parfor, name_var_table, typemap,
                         calltypes, save_copies)
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
            def process_assign(stmt):
                if isinstance(stmt, ir.Assign):
                    rhs = stmt.value
                    lhs = stmt.target
                    if (isinstance(rhs, ir.Global)):
                        saved_globals[lhs.name] = stmt
                        block_defs.add(lhs.name)
                    elif isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
                        if (rhs.value.name in saved_globals
                                or rhs.value.name in saved_getattrs):
                            saved_getattrs[lhs.name] = stmt
                            block_defs.add(lhs.name)

            if isinstance(stmt, Parfor):
                for s in stmt.init_block.body:
                    process_assign(s)
                pblocks = stmt.loop_body.copy()
                push_call_vars(pblocks, saved_globals, saved_getattrs)
                new_body.append(stmt)
                continue
            else:
                process_assign(stmt)
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

def repr_arrayexpr(arrayexpr):
    """Extract operators from arrayexpr to represent it abstractly as a string.
    """
    if isinstance(arrayexpr, tuple):
        opr = arrayexpr[0]
        # sometimes opr is not string like '+', but is a ufunc object
        if not isinstance(opr, str):
            if hasattr(opr, '__name__'):
                opr = opr.__name__
            else:
                opr = '_'  # can return dummy since repr is not critical
        args = arrayexpr[1]
        if len(args) == 1:
            return '({}{})'.format(opr, repr_arrayexpr(args[0]))
        else:
            return '({})'.format(opr.join([ repr_arrayexpr(x) for x in args ]))
    else:
        return '_'

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
        accesses = set()
    blocks = wrap_parfor_blocks(parfor)
    accesses = ir_utils.get_array_accesses(blocks, accesses)
    unwrap_parfor_blocks(parfor)
    return accesses


# parfor handler is same as
ir_utils.array_accesses_extensions[Parfor] = get_parfor_array_accesses


def parfor_add_offset_to_labels(parfor, offset):
    blocks = wrap_parfor_blocks(parfor)
    blocks = add_offset_to_labels(blocks, offset)
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
    # no need to handle parfor.index_var (tuple of variables), since it will be
    # assigned to a tuple from individual indices
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

def build_parfor_definitions(parfor, definitions=None):
    """get variable definition table for parfors"""
    if definitions is None:
        definitions = defaultdict(list)

    # avoid wrap_parfor_blocks() since build_definitions is called inside
    # find_potential_aliases_parfor where the parfor is already wrapped
    build_definitions(parfor.loop_body, definitions)
    build_definitions({0: parfor.init_block}, definitions)
    return definitions

ir_utils.build_defs_extensions[Parfor] = build_parfor_definitions

@contextmanager
def dummy_return_in_loop_body(loop_body):
    """adds dummy return to last block of parfor loop body for CFG computation
    """
    # max is last block since we add it manually for prange
    last_label = max(loop_body.keys())
    loop_body[last_label].body.append(
        ir.Return(0, ir.Loc("parfors_dummy", -1)))
    yield
    # remove dummy return
    loop_body[last_label].body.pop()

@infer_global(reduce)
class ReduceInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        assert isinstance(args[1], types.Array)
        return signature(args[1].dtype, *args)
