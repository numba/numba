from __future__ import print_function, division, absolute_import

import ast
from collections import defaultdict
import sys

from numpy import ufunc

from .. import compiler, ir, types, rewrites, six
from ..typing import npydecl
from .dufunc import DUFunc


@rewrites.register_rewrite('after-inference')
class RewriteArrayExprs(rewrites.Rewrite):
    '''The RewriteArrayExprs class is responsible for finding array
    expressions in Numba intermediate representation code, and
    rewriting those expressions to a single operation that will expand
    into something similar to a ufunc call.
    '''
    def __init__(self, pipeline, *args, **kws):
        super(RewriteArrayExprs, self).__init__(pipeline, *args, **kws)
        # Install a lowering hook if we are using this rewrite.
        special_ops = self.pipeline.targetctx.special_ops
        if 'arrayexpr' not in special_ops:
            special_ops['arrayexpr'] = _lower_array_expr

    def match(self, interp, block, typemap, calltypes):
        '''Using typing and a basic block, search the basic block for array
        expressions.  Returns True when one or more matches were
        found, False otherwise.
        '''
        matches = []
        # We can trivially reject everything if there are fewer than 2
        # calls in the type results since we'll only rewrite when
        # there are two or more calls.
        if len(calltypes) > 1:
            self.crnt_block = block
            self.typemap = typemap
            self.matches = matches
            array_assigns = {}
            self.array_assigns = array_assigns
            const_assigns = {}
            self.const_assigns = const_assigns
            assignments = block.find_insts(ir.Assign)
            for instr in assignments:
                target_name = instr.target.name
                expr = instr.value
                if isinstance(expr, ir.Expr) and isinstance(
                        typemap.get(target_name, None), types.Array):
                    # We've matched a subexpression assignment to an
                    # array variable.  Now see if the expression is an
                    # array expression.
                    expr_op = expr.op
                    if ((expr_op in ('unary', 'binop')) and (
                            expr.fn in npydecl.supported_array_operators)):
                        # Matches an array operation that maps to a ufunc.
                        array_assigns[target_name] = instr
                    elif ((expr_op == 'call') and (expr.func.name in typemap)):
                        # Could be a match for a ufunc or DUFunc call.
                        func_type = typemap[expr.func.name]
                        # Note: func_type can be a types.Dispatcher, which
                        #       doesn't have the `.template` attribute.
                        if isinstance(func_type, types.Function):
                            func_key = func_type.typing_key
                            if isinstance(func_key, (ufunc, DUFunc)):
                                # If so, match it as a potential subexpression.
                                array_assigns[target_name] = instr
                    # Now check to see if we matched anything of
                    # interest; if so, check to see if one of the
                    # expression's dependencies isn't also a matching
                    # expression.
                    if target_name in array_assigns:
                        operands = set(var.name
                                       for var in expr.list_vars())
                        if operands.intersection(array_assigns.keys()):
                            # We've identified a nested array
                            # expression.  Rewrite it.
                            matches.append(target_name)
                elif isinstance(expr, ir.Const):
                    # Track constants since we might need them for an
                    # array expression.
                    const_assigns[target_name] = expr
        return len(matches) > 0

    def _get_array_operator(self, ir_expr):
        ir_op = ir_expr.op
        if ir_op in ('unary', 'binop'):
            return ir_expr.fn
        elif ir_op == 'call':
            return self.typemap[ir_expr.func.name].typing_key
        raise NotImplementedError(
            "Don't know how to find the operator for '{0}' expressions.".format(
                ir_op))

    def _get_operands(self, ir_expr):
        '''Given a Numba IR expression, return the operands to the expression
        in order they appear in the expression.
        '''
        ir_op = ir_expr.op
        if ir_op == 'binop':
            return ir_expr.lhs, ir_expr.rhs
        elif ir_op == 'unary':
            return ir_expr.list_vars()
        elif ir_op == 'call':
            return ir_expr.args
        raise NotImplementedError(
            "Don't know how to find the operands for '{0}' expressions.".format(
                ir_op))

    def _translate_expr(self, ir_expr):
        '''Translate the given expression from Numba IR to an array expression
        tree.
        '''
        ir_op = ir_expr.op
        if ir_op == 'arrayexpr':
            return ir_expr.expr
        operands_or_args = [self.const_assigns.get(op_var.name, op_var)
                            for op_var in self._get_operands(ir_expr)]
        return self._get_array_operator(ir_expr), operands_or_args

    def _handle_matches(self):
        '''Iterate over the matches, trying to find which instructions should
        be rewritten, deleted, or moved.
        '''
        replace_map = {}
        dead_vars = set()
        used_vars = defaultdict(int)
        for match in self.matches:
            instr = self.array_assigns[match]
            expr = instr.value
            arr_inps = []
            arr_expr = self._get_array_operator(expr), arr_inps
            new_expr = ir.Expr(op='arrayexpr',
                               loc=expr.loc,
                               expr=arr_expr,
                               ty=self.typemap[instr.target.name])
            new_instr = ir.Assign(new_expr, instr.target, instr.loc)
            replace_map[instr] = new_instr
            self.array_assigns[instr.target.name] = new_instr
            for operand in self._get_operands(expr):
                operand_name = operand.name
                if operand_name in self.array_assigns:
                    child_assign = self.array_assigns[operand_name]
                    child_expr = child_assign.value
                    child_operands = child_expr.list_vars()
                    for operand in child_operands:
                        used_vars[operand.name] += 1
                    arr_inps.append(self._translate_expr(child_expr))
                    if child_assign.target.is_temp:
                        dead_vars.add(child_assign.target.name)
                        replace_map[child_assign] = None
                elif operand_name in self.const_assigns:
                    arr_inps.append(self.const_assigns[operand_name])
                else:
                    used_vars[operand.name] += 1
                    arr_inps.append(operand)
        return replace_map, dead_vars, used_vars

    def _get_final_replacement(self, replacement_map, instr):
        '''Find the final replacement instruction for a given initial
        instruction by chasing instructions in a map from instructions
        to replacement instructions.
        '''
        replacement = replacement_map[instr]
        while replacement in replacement_map:
            replacement = replacement_map[replacement]
        return replacement

    def apply(self):
        '''When we've found array expressions in a basic block, rewrite that
        block, returning a new, transformed block.
        '''
        # Part 1: Figure out what instructions should be rewritten
        # based on the matches found.
        replace_map, dead_vars, used_vars = self._handle_matches()
        # Part 2: Using the information above, rewrite the target
        # basic block.
        result = self.crnt_block.copy()
        result.clear()
        delete_map = {}
        for instr in self.crnt_block.body:
            if isinstance(instr, ir.Assign):
                if instr in replace_map:
                    replacement = self._get_final_replacement(
                        replace_map, instr)
                    if replacement:
                        result.append(replacement)
                        for var in replacement.value.list_vars():
                            var_name = var.name
                            if var_name in delete_map:
                                result.append(delete_map.pop(var_name))
                            if used_vars[var_name] > 0:
                                used_vars[var_name] -= 1

                else:
                    result.append(instr)
            elif isinstance(instr, ir.Del):
                instr_value = instr.value
                if used_vars[instr_value] > 0:
                    used_vars[instr_value] -= 1
                    delete_map[instr_value] = instr
                elif instr_value not in dead_vars:
                    result.append(instr)
            else:
                result.append(instr)
        if delete_map:
            for instr in delete_map.values():
                result.insert_before_terminator(instr)
        return result


_unaryops = {
    '+' : ast.UAdd,
    '-' : ast.USub,
    '~' : ast.Invert,
}

_binops = {
    '+' : ast.Add,
    '-' : ast.Sub,
    '*' : ast.Mult,
    '/' : ast.Div,
    '/?' : ast.Div,
    '%' : ast.Mod,
    '|' : ast.BitOr,
    '>>' : ast.RShift,
    '^' : ast.BitXor,
    '<<' : ast.LShift,
    '&' : ast.BitAnd,
    '**' : ast.Pow,
    '//' : ast.FloorDiv,
}

_cmpops = {
    '==' : ast.Eq,
    '!=' : ast.NotEq,
    '<' : ast.Lt,
    '<=' : ast.LtE,
    '>' : ast.Gt,
    '>=' : ast.GtE,
}


def _arr_expr_to_ast(expr):
    '''Build a Python expression AST from an array expression built by
    RewriteArrayExprs.
    '''
    if isinstance(expr, tuple):
        op, arr_expr_args = expr
        ast_args = []
        env = {}
        for arg in arr_expr_args:
            ast_arg, child_env = _arr_expr_to_ast(arg)
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
        elif isinstance(op, (ufunc, DUFunc)):
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
        return ast.Name(expr.name, ast.Load(),
                        lineno=expr.loc.line,
                        col_offset=expr.loc.col if expr.loc.col else 0), {}
    elif isinstance(expr, ir.Const):
        return ast.Num(expr.value), {}
    raise NotImplementedError(
        "Don't know how to translate array expression '%r'" % (expr,))


def _lower_array_expr(lowerer, expr):
    '''Lower an array expression built by RewriteArrayExprs.
    '''
    expr_name = "__numba_array_expr_%s" % (hex(hash(expr)).replace("-", "_"))
    expr_var_list = expr.list_vars()
    expr_var_map = {}
    for expr_var in expr_var_list:
        expr_var_name = expr_var.name
        expr_var_new_name = expr_var_name.replace("$", "_").replace(".", "_")
        # Avoid inserting existing var into the expr_var_map
        if expr_var_new_name not in expr_var_map:
            expr_var_map[expr_var_new_name] = expr_var_name, expr_var
        expr_var.name = expr_var_new_name
    expr_filename = expr_var_list[0].loc.filename
    # Parameters are the names internal to the new closure.
    expr_params = sorted(expr_var_map.keys())
    # Arguments are the names external to the new closure (except in
    # Python abstract syntax, apparently...)
    expr_args = [expr_var_map[key][0] for key in expr_params]
    if hasattr(ast, "arg"):
        # Should be Python 3.x
        ast_args = [ast.arg(param_name, None)
                    for param_name in expr_params]
    else:
        # Should be Python 2.x
        ast_args = [ast.Name(param_name, ast.Param())
                    for param_name in expr_params]
    # Parse a stub function to ensure the AST is populated with
    # reasonable defaults for the Python version.
    ast_module = ast.parse('def {0}(): return'.format(expr_name),
                           expr_filename, 'exec')
    assert hasattr(ast_module, 'body') and len(ast_module.body) == 1
    ast_fn = ast_module.body[0]
    ast_fn.args.args = ast_args
    ast_fn.body[0].value, namespace = _arr_expr_to_ast(expr.expr)
    ast.fix_missing_locations(ast_module)
    code_obj = compile(ast_module, expr_filename, 'exec')
    six.exec_(code_obj, namespace)
    impl = namespace[expr_name]

    context = lowerer.context
    builder = lowerer.builder
    outer_sig = expr.ty(*(lowerer.typeof(name) for name in expr_args))
    inner_sig_args = []
    for argty in outer_sig.args:
        if isinstance(argty, types.Array):
            inner_sig_args.append(argty.dtype)
        else:
            inner_sig_args.append(argty)
    inner_sig = outer_sig.return_type.dtype(*inner_sig_args)

    # Follow the Numpy error model.  Note this also allows e.g. vectorizing
    # division (issue #1223).
    flags = compiler.Flags()
    flags.set('error_model', 'numpy')
    cres = context.compile_subroutine_no_cache(builder, impl, inner_sig, flags=flags)

    # Create kernel subclass calling our native function
    from ..targets import npyimpl

    class ExprKernel(npyimpl._Kernel):
        def generate(self, *args):
            arg_zip = zip(args, self.outer_sig.args, inner_sig.args)
            cast_args = [self.cast(val, inty, outty)
                         for val, inty, outty in arg_zip]
            result = self.context.call_internal(
                builder, cres.fndesc, inner_sig, cast_args)
            return self.cast(result, inner_sig.return_type,
                             self.outer_sig.return_type)

    args = [lowerer.loadvar(name) for name in expr_args]
    return npyimpl.numpy_ufunc_kernel(
        context, builder, outer_sig, args, ExprKernel, explicit_output=False)
