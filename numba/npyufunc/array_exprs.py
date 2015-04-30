from __future__ import print_function, division, absolute_import
import ast

from .. import ir, types, rewrites
from ..typing import npydecl
from ..targets import npyimpl


@rewrites.register_rewrite
class RewriteArrayExprs(rewrites.Rewrite):
    '''The RewriteArrayExprs class is responsible for finding array
    expressions in Numba intermediate representation code, and
    rewriting those expressions to a single operation that will expand
    into something similar to a ufunc call.
    '''
    _operators = set(npydecl.NumpyRulesArrayOperator._op_map.keys()).union(
        npydecl.NumpyRulesUnaryArrayOperator._op_map.keys())

    def __init__(self, pipeline, *args, **kws):
        # At time of codeing, there shouldn't be anything in args or
        # kws, but they are there for possible forward compatibility.
        super(RewriteArrayExprs, self).__init__(*args, **kws)
        # Install a lowering hook if we are using this rewrite.
        special_ops = pipeline.targetctx.special_ops
        if 'arrayexpr' not in special_ops:
            special_ops['arrayexpr'] = _lower_array_expr

    def match(self, block, typemap, calltypes):
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
            for instr in block.body:
                is_array_expr = (
                    isinstance(instr, ir.Assign)
                    and isinstance(typemap.get(instr.target.name, None),
                                   types.Array)
                    and isinstance(instr.value, ir.Expr)
                    and instr.value.op in ('unary', 'binop')
                    and instr.value.fn in self._operators
                )
                if is_array_expr:
                    target_name = instr.target.name
                    array_assigns[target_name] = instr
                    operands = set(var.name for var in instr.value.list_vars())
                    if operands.intersection(array_assigns.keys()):
                        matches.append(target_name)
        return len(matches) > 0

    def apply(self):
        '''When we've found array expressions in a basic block, rewrite that
        block, returning a new, transformed block.
        '''
        # Part 1: Iterate over the matches, trying to find which
        # instructions should be rewritten.
        replace_map = {}
        dead_vars = set()
        used_vars = set()
        for match in self.matches:
            instr = self.array_assigns[match]
            arr_inps = []
            arr_expr = instr.value.fn, arr_inps
            new_expr = ir.Expr(op='arrayexpr',
                               loc=instr.value.loc,
                               expr=arr_expr,
                               ty=self.typemap[instr.target.name])
            new_instr = ir.Assign(new_expr, instr.target, instr.loc)
            replace_map[instr] = new_instr
            self.array_assigns[instr.target.name] = new_instr
            for operand in instr.value.list_vars():
                operand_name = operand.name
                if operand_name in self.array_assigns:
                    child_assign = self.array_assigns[operand_name]
                    child_expr = child_assign.value
                    child_operands = child_expr.list_vars()
                    used_vars.update(operand.name
                                     for operand in child_operands)
                    if child_expr.op != 'arrayexpr':
                        arr_inps.append((child_expr.fn, child_operands))
                    else:
                        arr_inps.append(child_expr.expr)
                    if child_assign.target.is_temp:
                        dead_vars.add(child_assign.target.name)
                        replace_map[child_assign] = None
                else:
                    used_vars.add(operand.name)
                    arr_inps.append(operand)
        # Part 2: Using the information above, rewrite the target
        # basic block.
        result = ir.Block(self.crnt_block.scope, self.crnt_block.loc)
        delete_map = {}
        for instr in self.crnt_block.body:
            if isinstance(instr, ir.Assign):
                target_name = instr.target.name
                if instr in replace_map:
                    replacement = replace_map[instr]
                    if replacement:
                        result.append(replacement)
                        for var in replacement.value.list_vars():
                            var_name = var.name
                            if var_name in delete_map:
                                result.append(delete_map.pop(var_name))
                            if var_name in used_vars:
                                used_vars.remove(var_name)
                else:
                    result.append(instr)
            elif isinstance(instr, ir.Del):
                instr_value = instr.value
                if instr_value in used_vars:
                    used_vars.remove(instr_value)
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


def _arr_expr_to_ast(expr):
    '''Build a Python expression AST from an array expression built by
    RewriteArrayExprs.
    '''
    if isinstance(expr, tuple):
        op, args = expr
        if op in RewriteArrayExprs._operators:
            args = [_arr_expr_to_ast(arg) for arg in args]
            if len(args) == 2:
                if op in _binops:
                    return ast.BinOp(args[0], _binops[op](), args[1])
            else:
                assert op in _unaryops
                return ast.UnaryOp(_unaryops[op](), args[0])
    elif isinstance(expr, ir.Var):
        return ast.Name(expr.name, ast.Load(),
                        lineno=expr.loc.line,
                        col_offset=expr.loc.col if expr.loc.col else 0)
    raise NotImplementedError(
        "Don't know how to translate array expression '%r'" % (expr,))


def _lower_array_expr(lowerer, expr):
    '''Lower an array expression built by RewriteArrayExprs.
    '''
    expr_name = "__numba_array_expr_%s" % (hex(hash(expr)).replace("-", "_"))
    expr_args = expr.list_vars()
    expr_arg_names = [arg.name for arg in expr_args]
    ast_args = ast.arguments([ast.arg(arg.name, None)
                              for arg in expr_args],
                             None, [], [], None, [])
    ast_expr = _arr_expr_to_ast(expr.expr)
    ast_module = ast.fix_missing_locations(
        ast.Module([
            ast.FunctionDef(expr_name, ast_args, [
                ast.Return(ast_expr)
            ], [], None)
    ]))
    namespace = {}
    exec(compile(ast_module, expr_args[0].loc.filename, 'exec'), namespace)
    impl = namespace[expr_name]

    context = lowerer.context
    builder = lowerer.builder
    outer_sig = expr.ty(*(lowerer.typeof(name) for name in expr_arg_names))
    inner_sig_args = []
    for argty in outer_sig.args:
        if isinstance(argty, types.Array):
            inner_sig_args.append(argty.dtype)
        else:
            inner_sig_args.append(argty)
    inner_sig = outer_sig.return_type.dtype(*inner_sig_args)

    cres = context.compile_only_no_cache(builder, impl, inner_sig)

    class ExprKernel(npyimpl._Kernel):
        def generate(self, *args):
            arg_zip = zip(args, self.outer_sig.args, inner_sig.args)
            cast_args = [self.cast(val, inty, outty)
                         for val, inty, outty in arg_zip]
            result = self.context.call_internal(
                builder, cres.fndesc, inner_sig, cast_args)
            return self.cast(result, inner_sig.return_type,
                             self.outer_sig.return_type)

    args = [lowerer.loadvar(name) for name in expr_arg_names]
    return npyimpl.numpy_ufunc_kernel(
        context, builder, outer_sig, args, ExprKernel, explicit_output=False)
