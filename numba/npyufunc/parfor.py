from __future__ import print_function, division, absolute_import

import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys

import numpy as np

from .. import compiler, ir, types, rewrites, six, cgutils
from numba.ir_utils import *
from ..typing import npydecl, signature
from ..targets import npyimpl, imputils
from .dufunc import DUFunc
from .array_exprs import _is_ufunc, _unaryops, _binops, _cmpops
from numba import config
import llvmlite.llvmpy.core as lc
from numba.parfor2 import LoopNest
import numba
from numba import parfor2

def _mk_tuple(elts):
    if len(elts) == 1:
        return elts[0]
    else:
        return ast.Tuple(elts = elts, ctx = ast.Load())

class ParforReduction(object):
    '''The ParforReduction class holds information about reductions
    in a parfor.  The var field is the reduction variable.  The
    init_value field is the initial value for the reduction variable.
    The func field is the function used to reduce two variables.'''
    def __init__(self, var, init_value, func):
        self.var = var
        self.init_value = init_value
        self.func = func

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

    def __str__(self):
        if self.reductions == []:
            red_str = ""
        else:
            red_str = "\n\t\t\tReductions: " + str(self.reductions)
        pre_body = "Parfor:\n\t\t\tInputInfo: " + str(self.input_info) + "\n\t\t\tOutputInfo: " + str(self.output_info) + "\n\t\t\tPrestatements: " + str(self.pre_parfor) + "\n\t\t\tLoopNests: " + str(self.loop_nests) + "\n\t\t\tBody:\n"
        body = ""
        for stmt in self.loop_body:
            body += "\t\t\t\t" + ast.dump(stmt)
        post_body = red_str + "\n\t\t\tPoststatements: " + str(self.post_parfor) + "\n\t\t\tNamespace: " + str(self.namespace)
        return pre_body + body + post_body

    def __repr__(self):
        return self.__str__()

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
                    if config.DEBUG_ARRAY_OPT:
                        print("Parfor apply: ", expr.expr)
                    #expr.op = 'parfor'
                    #ast_body, namespace = _arr_expr_to_ast(expr.expr)
                    #if config.DEBUG_ARRAY_OPT:
                    #    print("namespace = ", namespace)
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
    if config.DEBUG_ARRAY_OPT:
        print("_arr_expr_to_parfor")
        print("out_var", out_var)
        print("expr_var_list", expr_var_list)
    expr_var_unique = sorted(set(expr_var_list), key=lambda var: var.name)
    if config.DEBUG_ARRAY_OPT:
        print("expr_var_unique", expr_var_unique)
    expr_inps = [ var.name for var in expr_var_unique ]
    inp_types = [ typemap[name] for name in expr_inps ]
    input_info = list(zip(expr_inps, inp_types))
    if config.DEBUG_ARRAY_OPT:
        print("expr input_info = ", input_info)
    expr_outs = [ out_var ]
    out_types = [ typemap[out_var] ]
    output_info = list(zip(expr_outs, out_types))
    if config.DEBUG_ARRAY_OPT:
        print("expr output_info = ", output_info)
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
    if config.DEBUG_ARRAY_OPT:
        print("ndim = ", ndim)
    # Make variables that calculate the size of each dimension
    size_vars = [ mk_unique_var("s" + str(i)) for i in range(ndim) ]
    # Make index variables for the loop nest
    idx_vars = [ mk_unique_var("i" + str(i)) for i in range(ndim) ]
    # make prestatement: (s0,...) = out.shape()
    pre = [ ast.Assign(
              targets = [ast.Tuple(elts = [ast.Name(v, ast.Store()) for v in size_vars], ctx = ast.Store())],
              value = ast.Attribute(
                  value = ast.Name(out_var, ast.Load()),
                  attr = 'shape',
                  ctx = ast.Load())) ]
    if config.DEBUG_ARRAY_OPT:
        print("pre = ", ast.dump(pre[0]))
    # body is assigning expr to out_var, but replacing all array with explicit subscripts
    body_ast, namespace = _arr_expr_to_ast(expr.expr, typemap, idx_vars)
    body = [ ast.Assign(
              targets = [ ast.Subscript(
                value = ast.Name(out_var, ast.Load()),
                slice = ast.Index(value = _mk_tuple([ast.Name(v, ast.Load()) for v in idx_vars])),
                ctx = ast.Store()) ],
              value = body_ast) ]
    if config.DEBUG_ARRAY_OPT:
        print("body = ", ast.dump(body[0]))
    loop_nests = [ LoopNest(i, r) for (i, r) in zip(idx_vars, size_vars) ]
    parfor = Parfor(expr, loop_body = body, input_info = input_info, output_info = output_info,
                  loop_nests = loop_nests, pre_parfor = pre, namespace = namespace)
    if config.DEBUG_ARRAY_OPT:
        print("parfor = ", ast.dump(ast.Module(body = parfor.to_ast())))

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

@contextlib.contextmanager
def _legalize_parameter_names(var_list):
    """
    Legalize names in the variable list for use as a Python function's
    parameter names.
    """
    var_map = OrderedDict()
    for var in var_list:
        old_name = var.name
        new_name = old_name.replace("$", "_").replace(".", "_")
        # Caller should ensure the names are unique
        assert new_name not in var_map
        var_map[new_name] = var, old_name
        var.name = new_name
    param_names = list(var_map)
    try:
        yield param_names
    finally:
        # Make sure the old names are restored, to avoid confusing
        # other parts of Numba (see issue #1466)
        for var, old_name in var_map.values():
            var.name = old_name

def _lower_parfor(lowerer, expr):
    '''Lower an array expression built by RewriteParfor.
    '''
    expr_name = "__numba_parfor_%s" % (hex(hash(expr)).replace("-", "_"))
    expr_filename = expr.loc.filename
    # generate the ast
    parfor_ast = expr.to_ast()
    if config.DEBUG_ARRAY_OPT:
        print("_lower_parfor: expr = ", expr)
        print("parfor_ast = ", ast.dump(ast.Module(body = parfor_ast)))
    legalize = LegalizeNames()
    parfor_ast = legalize.visit(ast.Module(body = parfor_ast)).body
    if config.DEBUG_ARRAY_OPT:
        print("parfor_ast after legalizing = ", ast.dump(ast.Module(body = parfor_ast)))
    # get the legalized name dictionary
    namedict = legalize.namedict
    if config.DEBUG_ARRAY_OPT:
        print("namedict = ", namedict)
    # argument contain inputs and outputs, since we are lowering parfor to gufunc
    expr_var_list = list(expr.input_info) + list(expr.output_info)
    # Arguments are the names external to the new closure
    expr_args = [ var[0] for var in expr_var_list ]
    # Parameters are what we need to declare the function formal params
    if config.DEBUG_ARRAY_OPT:
        print("expr_args = ", expr_args, " ", type(expr_args))
    expr_params = [ namedict[name] for name in expr_args ]
    if config.DEBUG_ARRAY_OPT:
        print("expr_params = ", expr_params, " ", type(expr_params))
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
    if config.DEBUG_ARRAY_OPT:
        print("lower_parfor: ast_module = ", ast.dump(ast_module)," namespace=", namespace)
    code_obj = compile(ast_module, expr_filename, 'exec')
    six.exec_(code_obj, namespace)
    impl = namespace[expr_name]
    if config.DEBUG_ARRAY_OPT:
        print("impl = ", impl, " ", type(impl))

    # 3. Prepare signatures as well as a gu_signature in the form of ('m','n',...)
    outer_typs = []
    gu_sin = []
    gu_sout = []
    if config.DEBUG_ARRAY_OPT:
        print("input_info = ", list(expr.input_info))
    num_inputs = len(list(expr.input_info))
    if config.DEBUG_ARRAY_OPT:
        print("num_inputs = ", num_inputs)
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
    if config.DEBUG_ARRAY_OPT:
        print("gu_signature = ", gu_signature, " ", type(gu_signature))

    # 4. Now compile a gufunc using the Python function as kernel.
    context = lowerer.context
    builder = lowerer.builder
    library = lowerer.library
    #outer_sig = expr.ty(*outer_typs)
    outer_sig = signature(types.none, *outer_typs)

    if config.DEBUG_ARRAY_OPT:
        print("outer_sig = ", outer_sig, " ", type(outer_sig))

    #_create_sched_wrapper(expr, expr_var_list, expr_args, expr_params, namedict, impl)

    if context.auto_parallel:
        return make_parallel_loop(lowerer, impl, gu_signature, outer_sig, expr_args)
    else:
        return make_sequential_loop(lowerer, impl, gu_signature, outer_sig, expr_args)

'''Here we create a function in text form and eval it into existence.
This function creates the schedule for the gufunc call and creates and
initializes reduction arrays equal to the thread count and initialized
to the initial value of the reduction var.  The gufunc is called and
then the reduction function is applied across the reduction arrays
before returning the final answer.
'''
def _create_sched_wrapper(parfor, expr_var_list, expr_args, expr_params, namedict, impl):
    first_input = parfor.input_info[0]
    first_input_typ = first_input[1]
    parfor_dim = first_input_typ.ndim
    if config.DEBUG_ARRAY_OPT:
        print("_create_sched_wrapper ", type(parfor), " ", parfor, " args = ", type(expr_args), " ", expr_args)
        print("First input = ", first_input, " type = ", first_input_typ, " ", type(first_input_typ))
    # Determine the unique names of the scheduling and gufunc functions.
    sched_func_name = "__numba_parfor_sched_%s" % (hex(hash(parfor)).replace("-", "_"))
    gufunc_name = "__numba_parfor_gufunc_%s" % (hex(hash(parfor)).replace("-", "_"))
    if config.DEBUG_ARRAY_OPT:
        print("sched_func_name ", type(sched_func_name), " ", sched_func_name)
        print("gufunc_name ", type(gufunc_name), " ", gufunc_name)

    # Create the scheduling function as text.
    sched_func = "def " + sched_func_name + "("
    sched_func += (", ".join(expr_params))
    sched_func += "):\n"
    assert isinstance(expr_var_list[0][1], types.Array)
    sched_func += "    full_iteration_space = numba.runtime.gufunc_scheduler.create_full_iteration(" + expr_params[0] + ")\n"
    sched_func += "    sched = numba.runtime.gufunc_scheduler.create_schedule(full_iteration_space, numba.npyufunc.parallel.get_thread_count())\n"
    red_arrays = ""
    red_reduces = ""
    for one_red_index in range(len(parfor.reductions)):
        sched_func += "    red" + str(one_red_index) + " = np.full((numba.npyufunc.parallel.get_thread_count(),), parfor.reductions[one_red_index].init_value)\n"
        red_arrays += ", red" + str(one_red_index)
        red_reduces += "functools.reduce(lambda a,b: " + str(parfor.reductions[one_red_index].func) + "(a,b), red" + str(one_red_index) + ", " + parfor.reductions[one_red_index].init_value + "),"
    sched_func += "    " + gufunc_name + "(sched, " + (", ".join(expr_params)) + red_arrays + ")\n"
    out_args = [ namedict[var[0]] for var in list(parfor.output_info)]
    if config.DEBUG_ARRAY_OPT:
        print("out_args = ", out_args)
    sched_func += "    return (" + " ".join([str(i) + ", " for i in out_args]) + red_reduces + ")\n"
    if config.DEBUG_ARRAY_OPT:
        print("sched_func ", type(sched_func), "\n", sched_func)

    # Create the gufunc function.
    gufunc_txt = "def " + gufunc_name + "(sched, " + (", ".join(expr_params)) + "):\n"
    for eachdim in range(parfor_dim):
        for indent in range(eachdim+1):
            gufunc_txt += "    "
        gufunc_txt += "for i" + str(eachdim) + " in range(sched[" + str(eachdim) + "], sched[" + str(eachdim + parfor_dim) + "] + 1):\n"
    for indent in range(parfor_dim+1):
        gufunc_txt += "    "
    gufunc_txt += "None"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
    exec(gufunc_txt)
    gufunc_func = eval(gufunc_name)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_func = ", type(gufunc_func), "\n", gufunc_func)
    gufunc_ir = compiler.run_frontend(gufunc_func)
    gufunc_ir.dump()

    impl_ir = compiler.run_frontend(impl)
    if config.DEBUG_ARRAY_OPT:
        print("impl_ir dump")
    impl_ir.dump()

    # Create the scheduling function from its text.
    exec(sched_func)

def lower_parfor2_parallel(func_ir, typemap, calltypes):
    """lower parfor to sequential or parallel Numba IR.
    """
    print("-"*10, " new parfor2 lower ", "-"*10)
    # TODO: lower to parallel
    new_blocks = {}
    for (block_label, block) in func_ir.blocks.items():
        scope = block.scope
        i = parfor2._find_first_parfor(block.body)
        while i!=-1:
            inst = block.body[i]
            _create_sched_wrapper2(inst)
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    func_ir.blocks = _rename_labels(func_ir.blocks)
    if config.DEBUG_ARRAY_OPT==1:
        print("function after parfor lowering:")
        func_ir.dump()
    return

# numba.parfor2.lower_parfor2_parallel = lower_parfor2_parallel

'''Here we create a function in text form and eval it into existence.
This function creates the schedule for the gufunc call and creates and
initializes reduction arrays equal to the thread count and initialized
to the initial value of the reduction var.  The gufunc is called and
then the reduction function is applied across the reduction arrays
before returning the final answer.
'''
def _create_sched_wrapper2(parfor):

    parfor_dim = len(parfor.loop_nests)
    assert parfor_dim==1
    loop_ranges = [l.range_variable.name for l in parfor.loop_nests]

    parfor_params = parfor2.get_parfor_params(parfor).union(loop_ranges)
    param_dict = legalize_names(parfor_params)
    _replace_names(parfor.loop_body, param_dict)
    parfor_params = list(param_dict.values())

    loop_ranges = [ param_dict[v] for v in loop_ranges ]

    if config.DEBUG_ARRAY_OPT==1:
        print("_create_sched_wrapper ", type(parfor), " ", parfor,
        " args = ", type(parfor_params), " ", parfor_params, " loop_ranges ",
        type(loop_ranges), loop_ranges)

    # Determine the unique names of the scheduling and gufunc functions.
    sched_func_name = "__numba_parfor_sched_%s" % (hex(hash(parfor)).replace("-", "_"))
    gufunc_name = "__numba_parfor_gufunc_%s" % (hex(hash(parfor)).replace("-", "_"))
    if config.DEBUG_ARRAY_OPT:
        print("sched_func_name ", type(sched_func_name), " ", sched_func_name)
        print("gufunc_name ", type(gufunc_name), " ", gufunc_name)

    # Create the scheduling function as text.
    sched_func = "def " + sched_func_name + "("
    sched_func += (", ".join(parfor_params))
    sched_func += "):\n"

    sched_func += "    full_iteration_space = numba.runtime.gufunc_scheduler"
    + ".create_full_iteration(" + loop_ranges + ")\n"
    sched_func += "    sched = numba.runtime.gufunc_scheduler.create_schedule"
    + "(full_iteration_space, numba.npyufunc.parallel.get_thread_count())\n"

    red_arrays = ""
    red_reduces = ""
    # for one_red_index in range(len(parfor.reductions)):
    #     sched_func += "    red" + str(one_red_index)
    #     + " = np.full((numba.npyufunc.parallel.get_thread_count(),),"
    #     + " parfor.reductions[one_red_index].init_value)\n"
    #     red_arrays += ", red" + str(one_red_index)
    #     red_reduces += "functools.reduce(lambda a,b: "
    #     + str(parfor.reductions[one_red_index].func) + "(a,b), red"
    #     + str(one_red_index) + ", "
    #     + parfor.reductions[one_red_index].init_value + "),"
    sched_func += "    " + gufunc_name + "(sched, " + (", ".join(parfor_params)) + red_arrays + ")\n"

    sched_func += "    return (" + red_reduces + ")\n"
    if config.DEBUG_ARRAY_OPT:
        print("sched_func ", type(sched_func), "\n", sched_func)

    # Create the gufunc function.
    gufunc_txt = "def " + gufunc_name + "(sched, " + (", ".join(parfor_params)) + "):\n"
    for eachdim in range(parfor_dim):
        for indent in range(eachdim+1):
            gufunc_txt += "    "
        gufunc_txt += "for i" + str(eachdim) + " in range(sched[" + str(eachdim)
        + "], sched[" + str(eachdim + parfor_dim) + "] + 1):\n"
    for indent in range(parfor_dim+1):
        gufunc_txt += "    "
    gufunc_txt += "__sentinel__ = 0\n"
    gufunc_txt += "     None\n"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
    exec(gufunc_txt)
    gufunc_func = eval(gufunc_name)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_func = ", type(gufunc_func), "\n", gufunc_func)
    gufunc_ir = compiler.run_frontend(gufunc_func)
    gufunc_ir.dump()
    for label, block in gufunc_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name=="__sentinel__":
                loc = inst.loc
                scope = block.scope
                # split block across __sentinel__
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]
                block.body = block.body[i+1:]
                new_label = next_label()
                body_first_label = min(parfor.loop_body.keys())
                prev_block.append(ir.Jump(body_first_label, loc))
                for (l, b) in inst.loop_body.items():
                    gufunc_ir.blocks[l] = b
                body_last_label = max(parfor.loop_body.keys())
                gufunc_ir.blocks[new_label] = block
                gufunc_ir.blocks[label] = prev_block

    # TODO: implement get params etc. functions
    # merege body ir
    # run type inference
    # run backend

    # Create the scheduling function from its text.
    exec(sched_func)

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

    if config.DEBUG_ARRAY_OPT:
        print("make_parallel_loop")
        print("args = ", expr_args)
        print("outer_sig = ", outer_sig.args, outer_sig.return_type, outer_sig.recvr, outer_sig.pysig)
        print("inner_sig = ", inner_sig.args, inner_sig.return_type, inner_sig.recvr, inner_sig.pysig)
    # The ufunc takes 4 arguments: args, dims, steps, data
    # ufunc = ParallelGUFuncBuilder(impl, gu_signature)
    sin, sout = gu_signature
    ufunc = GUFuncBuilder(impl, gu_signature)
    ufunc.add(outer_sig)
    #wrapper_func = ufunc.build_ufunc()
    if config.DEBUG_ARRAY_OPT:
        print("_sigs = ", ufunc._sigs)
    sig = ufunc._sigs[0]
    cres = ufunc._cres[sig]
    #dtypenums, wrapper, env = ufunc.build(cres, sig)
    #_launch_threads()
    #_init()
    llvm_func = cres.library.get_function(cres.fndesc.llvm_func_name)
    wrapper_ptr, env, wrapper_name = build_gufunc_wrapper(llvm_func, cres, sin, sout, {})
    cres.library._ensure_finalized()

    if config.DEBUG_ARRAY_OPT:
        print("parallel function = ", wrapper, cres, sig)

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
    if config.DEBUG_ARRAY_OPT:
         print("ndims = ", ndims)
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
    if config.DEBUG_ARRAY_OPT:
        print("result = ", result)

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
