from __future__ import print_function, division, absolute_import

import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys

import numpy as np

from .. import compiler, ir, types, rewrites, six, cgutils
from ..typing import npydecl, signature
from ..targets import npyimpl, imputils
from .dufunc import DUFunc
from .array_exprs import _is_ufunc, _unaryops, _binops, _cmpops
import llvmlite.llvmpy.core as lc

unique_var_count = 0
def _make_unique_var(prefix):
    global unique_var_count
    var = prefix + "." + str(unique_var_count)
    unique_var_count = unique_var_count + 1
    return var

def _mk_tuple(elts):
    if len(elts) == 1:
        return elts[0]
    else:
        return ast.Tuple(elts = elts, ctx = ast.Load())

class LoopNest(object):
    '''The LoopNest class holds information of a single loop including
    the index variable (of a non-negative integer value), and the
    range variable, e.g. range(r) is 0 to r-1 with step size 1.
    '''
    def __init__(self, index_variable, range_variable, correlation=-1):
        self.index_variable = index_variable
        self.range_variable = range_variable
        self.correlation = correlation

class Parfor(ir.Expr):
    '''The Parfor class holds necessary information for a parallelizable
    looping computation over a given set of LoopNests.
    '''
    def __init__(self, expr, loop_body = [], input_info = [], output_info = [],
                 loop_nests = [], pre_parfor = [], post_parfor = [],
                 reductions = [], namespace = []):
        super(Parfor, self).__init__(
            op   = "parfor",
            loc  = expr.loc,
            expr = expr,
            ty   = expr.ty
        )

        self.input_info  = input_info
        self.output_info = output_info
        self.loop_body   = loop_body
        self.pre_parfor  = pre_parfor
        self.post_parfor = post_parfor
        self.loop_nests  = loop_nests
        self.reductions  = reductions
        self.name        = "Parfor"
        self.namespace   = namespace

    '''Convert Parfor to nested for loops in Python ast. The result
    can be treated as the body of a python function and compile
    separately.
    '''
    def to_ast(self):
        def mk_loop(loop_nests, loop_body):
            if len(loop_nests) == 0:
                #print("return loop_body: ", len(loop_body))
                return loop_body
            else:
                nest, *nests = loop_nests
                return [ ast.For(
                    target = ast.Name(nest.index_variable, ast.Store()),
                    iter = ast.Call(
                        func = ast.Name('range', ast.Load()),
                        args = [ast.Name(nest.range_variable, ast.Load())],
                        keywords = []),
                    body = mk_loop(nests, loop_body),
                    orelse = []) ]
        #print("number of loop nests = ", len(self.loop_nests))
        #debug = [ ast.Expr(ast.Call(ast.Name('print', ast.Load()), [ast.Attribute(ast.Name(self.output_info[0][0], ast.Load()), 'shape', ast.Load())], [])) ]
        return self.pre_parfor + mk_loop(self.loop_nests, self.loop_body) + self.post_parfor

@rewrites.register_rewrite('after-inference')
class RewriteParforExtra(rewrites.Rewrite):
    """The RewriteParforExtra class is responsible for converting Numpy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """
    def __init__(self, pipeline, *args, **kws):
        super(RewriteParforExtra, self).__init__(pipeline, *args, **kws)
        self.array_analysis = pipeline.array_analysis
        # Install a lowering hook if we are using this rewrite.
        special_ops = self.pipeline.targetctx.special_ops
        if 'parfor' not in special_ops:
            special_ops['parfor'] = _lower_parfor

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
            assert self._get_ndims(in1)<=2 and self._get_ndims(in2)==1
            # loop range correlation is same as first dimention of 1st input
            corr = self.array_analysis.array_shape_classes[in1][0]
            size_var = self.array_analysis.array_size_vars[in1][0]
            index_var = ir.Var(lhs.scope, _make_unique_var("parfor_index"), lhs.loc)
            self.type_annotation.typemap[index_var.name] = int_typ
            loopnests = [ LoopNest(index_var, size_var, corr) ]
            if self._get_ndims(in1)==2:
                # for 2D input, there is an inner loop
                # correlation of inner dimension
                inner_size_var = self.array_analysis.array_size_vars[in1][1]
                body = []

        # return error if we couldn't handle it (avoid rewrite infinite loop)
        raise NotImplementedError("parfor translation failed for ", expr)


@rewrites.register_rewrite('after-inference')
class RewriteParfor(rewrites.Rewrite):
    '''The RewriteParfor class is responsible for converting ArrayExpr
    expressions in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    '''
    def __init__(self, pipeline, *args, **kws):
        super(RewriteParfor, self).__init__(pipeline, *args, **kws)
        # Install a lowering hook if we are using this rewrite.
        special_ops = self.pipeline.targetctx.special_ops
        if 'parfor' not in special_ops:
            special_ops['parfor'] = _lower_parfor

    def match(self, interp, block, typemap, calltypes):
        """
        We'll match 'arrayexpr' operator.
        Return True when one or more matches were found, False otherwise.
        """
        # We can trivially reject everything if there are no
        # calls in the type results.
        if len(calltypes) == 0:
            return False

        self.crnt_block = block
        self.typemap = typemap
        # { variable name: IR assignment (of 'arrayexpr') }
        self.array_exprs = OrderedDict()

        assignments = block.find_insts(ir.Assign)
        for instr in assignments:
            target_name = instr.target.name
            expr = instr.value
            # Does it assign an expression to an array variable?
            if (isinstance(expr, ir.Expr) and expr.op == 'arrayexpr'):
                self.array_exprs[target_name] = instr

        #print("RewriteParfor match arrayexpr:", len(self.array_exprs) > 0)
        return len(self.array_exprs) > 0

    def apply(self):
        '''When we've found ArrayExpr in a basic block, rewrite that
        block, returning a Parfor block.
        '''
        array_exprs = self.array_exprs
        result = ir.Block(self.crnt_block.scope, self.crnt_block.loc)
        block = self.crnt_block
        scope = block.scope
        for instr in block.body:
            if isinstance(instr, ir.Assign):
                if instr.target.name in array_exprs:
                    expr = instr.value
                    #print("Parfor apply: ", expr.expr)
                    #expr.op = 'parfor'
                    #ast_body, namespace = _arr_expr_to_ast(expr.expr)
                    #print("namespace = ", namespace)
                    #expr.expr = Parfor(namespace, ast_body, {})
                    #expr.expr = Parfor("parfor", expr.loc, {}, expr.expr, {})
                    instr.value = _arr_expr_to_parfor(instr.target.name, expr, self.typemap)

            result.append(instr)

        return result

def _arr_expr_to_ast(expr, typemap, subscripts):
    '''Build a Python expression AST from an array expression built by
    RewriteParfor.
    '''
    if isinstance(expr, tuple):
        op, arr_expr_args = expr
        ast_args = []
        env = {}
        for arg in arr_expr_args:
            ast_arg, child_env = _arr_expr_to_ast(arg, typemap, subscripts)
            ast_args.append(ast_arg)
            env.update(child_env)
        if op in npydecl.supported_array_operators:
            if len(ast_args) == 2:
                if op in _binops:
                    return ast.BinOp(
                        ast_args[0], _binops[op](), ast_args[1]), env
                if op in _cmpops:
                    return ast.Compare(
                        ast_args[0], [_cmpops[op]()], [ast_args[1]]), env
            else:
                assert op in _unaryops
                return ast.UnaryOp(_unaryops[op](), ast_args[0]), env
        elif _is_ufunc(op):
            fn_name = "__ufunc_or_dufunc_{0}".format(
                hex(hash(op)).replace("-", "_"))
            fn_ast_name = ast.Name(fn_name, ast.Load())
            env[fn_name] = op # Stash the ufunc or DUFunc in the environment
            if sys.version_info >= (3, 5):
                ast_call = ast.Call(fn_ast_name, ast_args, [])
            else:
                ast_call = ast.Call(fn_ast_name, ast_args, [], None, None)
            return ast_call, env
    elif isinstance(expr, ir.Var):
        name = expr.name
        var = ast.Name(name, ast.Load(),
                        lineno=expr.loc.line,
                        col_offset=expr.loc.col if expr.loc.col else 0)
        typ = typemap[name]
        if isinstance(typ, types.Array):
            var = ast.Subscript(
                value = var,
                slice = ast.Index(value = _mk_tuple([ast.Name(v, ast.Load()) for v in subscripts])),
                ctx = ast.Load())
        return var, {}

    elif isinstance(expr, ir.Const):
        return ast.Num(expr.value), {}
    raise NotImplementedError(
        "Don't know how to translate array expression '%r'" % (expr,))

def _arr_expr_to_parfor(out_var, expr, typemap):
    expr_var_list = expr.list_vars()
    #print("out_var", out_var)
    #print("expr_var_list", expr_var_list)
    expr_var_unique = sorted(set(expr_var_list), key=lambda var: var.name)
    #print("expr_var_unique", expr_var_unique)
    expr_inps = [ var.name for var in expr_var_unique ]
    inp_types = [ typemap[name] for name in expr_inps ]
    input_info = list(zip(expr_inps, inp_types))
    #print("expr input_info = ", input_info)
    expr_outs = [ out_var ]
    out_types = [ typemap[out_var] ]
    output_info = list(zip(expr_outs, out_types))
    #print("expr output_info = ", output_info)
    ndim = 0
    # Find out number of dimensions, all arrays must match
    for idx, typ in enumerate(out_types + inp_types):
        if isinstance(typ, types.Array):
            if ndim == 0:
                ndim = typ.ndim
            else:
                if ndim != typ.ndim:
                    raise NotImplementedError(
                        "Don't know how to make loop nests of unmatching dimension, expect {0} but got {1}.".format(ndim, typ.ndim))
    if ndim == 0:
        raise NotImplementedError("Don't know to make loop nests when no arrays are found")
    #print("ndim = ", ndim)
    # Make variables that calculate the size of each dimension
    size_vars = [ _make_unique_var("s" + str(i)) for i in range(ndim) ]
    # Make index variables for the loop nest
    idx_vars = [ _make_unique_var("i" + str(i)) for i in range(ndim) ]
    # make prestatement: (s0,...) = out.shape()
    pre = [ ast.Assign(
              targets = [ast.Tuple(elts = [ast.Name(v, ast.Store()) for v in size_vars], ctx = ast.Store())],
              value = ast.Attribute(
                  value = ast.Name(out_var, ast.Load()),
                  attr = 'shape',
                  ctx = ast.Load())) ]
    #print("pre = ", ast.dump(pre[0]))
    # body is assigning expr to out_var, but replacing all array with explicit subscripts
    body_ast, namespace = _arr_expr_to_ast(expr.expr, typemap, idx_vars)
    body = [ ast.Assign(
              targets = [ ast.Subscript(
                value = ast.Name(out_var, ast.Load()),
                slice = ast.Index(value = _mk_tuple([ast.Name(v, ast.Load()) for v in idx_vars])),
                ctx = ast.Store()) ],
              value = body_ast) ]
    #print("body = ", ast.dump(body[0]))
    loop_nests = [ LoopNest(i, r) for (i, r) in zip(idx_vars, size_vars) ]
    parfor = Parfor(expr, loop_body = body, input_info = input_info, output_info = output_info,
                  loop_nests = loop_nests, pre_parfor = pre, namespace = namespace)
    #print("parfor = ", ast.dump(ast.Module(body = parfor.to_ast())))
    return parfor

class LegalizeNames(ast.NodeTransformer):
    def __init__(self):
        self.namedict = {}

    def visit_Name(self, node):
        #print("visit_Name: ", ast.dump(node))
        old_name = node.id
        new_name = None
        if old_name in self.namedict:
            new_name = self.namedict[old_name]
        else:
            new_name = old_name.replace("$", "_").replace(".", "_")
            self.namedict[old_name] = new_name
            if new_name == old_name:
                return node
        new_node = ast.Name(new_name, node.ctx)
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return ast.Name(new_name, node.ctx)


def _lower_parfor(lowerer, expr):
    '''Lower an array expression built by RewriteParfor.
    '''
    expr_name = "__numba_parfor_%s" % (hex(hash(expr)).replace("-", "_"))
    expr_filename = expr.loc.filename
    # generate the ast
    parfor_ast = expr.to_ast()
    #print("parfor_ast = ", ast.dump(ast.Module(body = parfor_ast)))
    legalize = LegalizeNames()
    parfor_ast = legalize.visit(ast.Module(body = parfor_ast)).body
    # get the legalized name dictionary
    namedict = legalize.namedict
    #print("namedict = ", namedict)
    # argument contain inputs and outputs, since we are lowering parfor to gufunc
    expr_var_list = list(expr.input_info) + list(expr.output_info)
    # Arguments are the names external to the new closure
    expr_args = [ var[0] for var in expr_var_list ]
    # Parameters are what we need to declare the function formal params
    #print("expr_args = ", expr_args)
    expr_params = [ namedict[name] for name in expr_args ]
    #print("expr_params = ", expr_params)
    # 1. Create an AST tree from the array expression.
    if hasattr(ast, "arg"):
        # Should be Python 3.x
        ast_args = [ast.arg(param_name, None) for param_name in expr_params]
    else:
        # Should be Python 2.x
        ast_args = [ast.Name(param_name, ast.Param()) for param_name in expr_params]
    # Parse a stub function to ensure the AST is populated with
    # reasonable defaults for the Python version.
    ast_module = ast.parse('def {0}(): return'.format(expr_name), expr_filename, 'exec')
    assert hasattr(ast_module, 'body') and len(ast_module.body) == 1
    ast_fn = ast_module.body[0]
    ast_fn.args.args = ast_args
    ast_fn.body = parfor_ast + [ast.Return(None)]
    namespace = expr.namespace
    #ast_fn.body[0].value, namespace = _arr_expr_to_ast(expr.expr.expr)
    ast.fix_missing_locations(ast_module)

    # 2. Compile the AST module and extract the Python function.
    #print("lower_parfor: ast_module = ", ast.dump(ast_module)," namespace=", namespace)
    code_obj = compile(ast_module, expr_filename, 'exec')
    six.exec_(code_obj, namespace)
    impl = namespace[expr_name]
    #print("impl = ", impl)

    # 3. Prepare signatures as well as a gu_signature in the form of ('m','n',...)
    outer_typs = []
    gu_sin = []
    gu_sout = []
    #print("input_info = ", list(expr.input_info))
    num_inputs = len(list(expr.input_info))
    #print("num_inputs = ", num_inputs)
    count = 0
    for var, typ in expr_var_list:
        #print("var = ", var, " typ = ", typ)
        count = count + 1
        outer_typs.append(typ)
        if isinstance(typ, types.Array):
            dim_syms = tuple([ chr(109 + i) for i in range(typ.ndim) ]) # chr(109) = 'm'
        else:
            dim_syms = ()
        if (count > num_inputs):
            gu_sout.append(dim_syms)
        else:
            gu_sin.append(dim_syms)
    gu_signature = (gu_sin, gu_sout)
    #print("gu_signature = ", gu_signature)

    # 4. Now compile a gufunc using the Python function as kernel.
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library
    #outer_sig = expr.ty(*outer_typs)
    outer_sig = signature(types.none, *outer_typs)

    if context.auto_parallel:
        return make_parallel_loop(lowerer, impl, gu_signature, outer_sig, expr_args)
    else:
        return make_sequential_loop(lowerer, impl, gu_signature, outer_sig, expr_args)


def _prepare_arguments(lowerer, gu_signature, outer_sig, expr_args):
    context = lowerer.context
    builder = lowerer.builder
    sin, sout = gu_signature
    num_inputs = len(sin)
    num_args = len(outer_sig.args)
    arguments = []
    inputs = []
    output = None
    out_ty = None
    input_sig_args = outer_sig.args[:num_inputs]
    for i in range(num_args):
        arg_ty = outer_sig.args[i]
        #print("arg_ty = ", arg_ty)
        if i < num_inputs:
            #print("as input")
            var = lowerer.loadvar(expr_args[i])
            arg = npyimpl._prepare_argument(context, builder, var, arg_ty)
            arguments.append(arg)
            inputs.append(arg)
        else:
            if isinstance(arg_ty, types.ArrayCompatible):
                #print("as output array")
                output = npyimpl._build_array(context, builder, arg_ty, input_sig_args, inputs)
                out_ty = arg_ty
                arguments.append(output)
            else:
                #print("as output scalar")
                output = npyimpl._prepare_argument(context, builder,
                         lc.Constant.null(context.get_value_type(arg_ty)), arg_ty)
                out_ty = arg_ty
                arguments.append(output)
    return inputs, output, out_ty


def make_sequential_loop(lowerer, impl, gu_signature, outer_sig, expr_args):
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library

    # Follow the Numpy error model.  Note this also allows e.g. vectorizing
    # division (issue #1223).
    flags = compiler.Flags()
    flags.set('error_model', 'numpy')
    cres = context.compile_subroutine_no_cache(builder, impl, outer_sig, flags=flags)

    inputs, output, out_ty = _prepare_arguments(lowerer, gu_signature, outer_sig, expr_args)
    args = [ x.return_val for x in inputs + [output] ]
    # cgutils.printf(builder, "args[0].data = %p\n", inputs[0].data)
    result = context.call_internal(builder, cres.fndesc, outer_sig, args)
    return imputils.impl_ret_new_ref(context, builder, out_ty, output.return_val)


def make_parallel_loop(lowerer, impl, gu_signature, outer_sig, expr_args):
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library

    #from .parallel import ParallelGUFuncBuilder, build_gufunc_wrapper, _launch_threads, _init
    from .ufuncbuilder import GUFuncBuilder, build_gufunc_wrapper #, _launch_threads, _init

    #print("args = ", expr_args)
    #print("outer_sig = ", outer_sig.args, outer_sig.return_type, outer_sig.recvr, outer_sig.pysig)
    #print("inner_sig = ", inner_sig.args, inner_sig.return_type, inner_sig.recvr, inner_sig.pysig)
    # The ufunc takes 4 arguments: args, dims, steps, data
    # ufunc = ParallelGUFuncBuilder(impl, gu_signature)
    sin, sout = gu_signature
    ufunc = GUFuncBuilder(impl, gu_signature)
    ufunc.add(outer_sig)
    #wrapper_func = ufunc.build_ufunc()
    #print("_sigs = ", ufunc._sigs)
    sig = ufunc._sigs[0]
    cres = ufunc._cres[sig]
    #dtypenums, wrapper, env = ufunc.build(cres, sig)
    #_launch_threads()
    #_init()
    llvm_func = cres.library.get_function(cres.fndesc.llvm_func_name)
    wrapper_ptr, env, wrapper_name = build_gufunc_wrapper(llvm_func, cres, sin, sout, {})
    cres.library._ensure_finalized()

    #print("parallel function = ", wrapper, cres, sig)

    byte_t = lc.Type.int(8)
    byte_ptr_t = lc.Type.pointer(byte_t)
    byte_ptr_ptr_t = lc.Type.pointer(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    intp_ptr_t = lc.Type.pointer(intp_t)
    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)

    # prepare arguments: args, dims, steps, data
    inputs, output, out_ty = _prepare_arguments(lowerer, gu_signature, outer_sig, expr_args)

    arguments = [ x.data for x in inputs + [output] ]
    num_args = len(arguments)
    # prepare input/output array args
    args = cgutils.alloca_once(builder, byte_ptr_t, size = context.get_constant(types.intp, num_args), name = "pargs")
    for i in range(num_args):
        dst = builder.gep(args, [context.get_constant(types.intp, i)])
        #cgutils.printf(builder, "arg[" + str(i) + "] = %p\n", arguments[i])
        builder.store(builder.bitcast(arguments[i], byte_ptr_t), dst)

    # prepare dims, which is only a single number, since N-D arrays is treated as 1D array by ufunc
    ndims = len(output.shape)
    dims = cgutils.alloca_once(builder, intp_t, size = 2, name = "pshape")
    # dims = builder.alloca(intp_t)
    size = one
    # print("ndims = ", ndims)
    for i in range(ndims):
       #cgutils.printf(builder, "dims[" + str(i) + "] = %d\n", output.shape[i])
       size = builder.mul(size, output.shape[i])
    #cgutils.printf(builder, wrapper.name + " " + cres.fndesc.llvm_func_name + " total size = %d\n", size)
    # We can't directly use size here, must separate core dimension and loop dimension
    builder.store(one,  builder.gep(dims, [ zero ]))
    builder.store(size, builder.gep(dims, [ one ]))

    # prepare steps for each argument
    steps = cgutils.alloca_once(builder, intp_t, size = context.get_constant(types.intp, num_args + 1), name = "psteps")
    for i in range(num_args):
        # all steps are 0
        # sizeof = context.get_abi_sizeof(context.get_value_type(arguments[i].base_type))
        # stepsize = context.get_constant(types.intp, sizeof)
        stepsize = zero
        #cgutils.printf(builder, "stepsize = %d\n", stepsize)
        dst = builder.gep(steps, [context.get_constant(types.intp, i)])
        builder.store(stepsize, dst)
    # steps for output array goes last
    # sizeof = context.get_abi_sizeof(context.get_value_type(output.base_type))
    # stepsize = context.get_constant(types.intp, sizeof)
    # cgutils.printf(builder, "stepsize = %d\n", stepsize)
    # dst = builder.gep(steps, [lc.Constant.int(lc.Type.int(), num_args)])
    # builder.store(stepsize, dst)

    # prepare data
    data = builder.inttoptr(zero, byte_ptr_t)

    #result = context.call_function_pointer(builder, wrapper, [args, dims, steps, data])
    fnty = lc.Type.function(lc.Type.void(), [byte_ptr_ptr_t, intp_ptr_t,
                                             intp_ptr_t, byte_ptr_t])
    fn = builder.module.get_or_insert_function(fnty, name=wrapper_name)
    #cgutils.printf(builder, "before calling kernel %p\n", fn)
    result = builder.call(fn, [args, dims, steps, data])
    #cgutils.printf(builder, "after calling kernel %p\n", fn)
    #print("result = ", result)

    # return builder.bitcast(output.return_val, ret_ty)
    return imputils.impl_ret_new_ref(context, builder, out_ty, output.return_val)

    # cres = context.compile_subroutine_no_cache(builder, wrapper_func, outer_sig, flags=flags)
    # args = [lowerer.loadvar(name) for name in expr_args]
    # result = context.call_internal(builder, cres.fndesc, outer_sig, args)
    # status, res = context.call_conv.call_function(builder, cres.fndesc, outer_sig.return_type,
    #                                              outer_sig.args, expr_args)
    #with cgutils.if_unlikely(builder, status.is_error):
    #        context.call_conv.return_status_propagate(builder, status)
    # return res
