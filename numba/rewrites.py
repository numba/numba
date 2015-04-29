from __future__ import print_function, division, absolute_import

from . import ir, types

# Imports specific to a concrete rewrite (maybe move?):
from .typing import npydecl


class Rewrite(object):
    def match(self, block, typemap, calltypes):
        return False

    def apply(self):
        raise NotImplementedError("Abstract Rewrite.apply() called!")


class RewriteRegistry(object):
    def __init__(self):
        self.rewrites = []

    def register(self, rewrite_cls):
        assert issubclass(rewrite_cls, Rewrite)
        self.rewrites.append(rewrite_cls)
        return rewrite_cls

    def apply(self, blocks, typemap, calltypes):
        old_blocks = blocks.copy()
        for rewrite_cls in self.rewrites:
            # Exhaustively apply a rewrite until it stops matching.
            rewrite = rewrite_cls()
            work_list = list(blocks.items())
            while work_list:
                key, block = work_list.pop()
                matches = rewrite.match(block, typemap, calltypes)
                if matches:
                    new_block = rewrite.apply()
                    blocks[key] = new_block
                    work_list.append((key, new_block))
        # If any blocks were changed, perform a sanity check.
        for key, block in blocks.items():
            if block != old_blocks[key]:
                block.verify()


rewrite_registry = RewriteRegistry()
register_rewrite = rewrite_registry.register


@register_rewrite
class RewriteArrayExprs(Rewrite):
    _operators = set(npydecl.NumpyRulesArrayOperator._op_map.keys()).union(
        npydecl.NumpyRulesUnaryArrayOperator._op_map.keys())

    def __init__(self, *args, **kws):
        super(RewriteArrayExprs, self).__init__(*args, **kws)
        self.match_map = {}

    def _get_operands(self, expr):
        result = set()
        if expr.op in ('unary', 'binop') and expr.fn in self._operators:
            result.update(var.name for var in expr.list_vars())
        return result

    def match(self, block, typemap, calltypes):
        matches = []
        # We can trivially reject everything if there are fewer than 2
        # calls in the type results since we'll only rewrite when
        # there are two or more calls.
        if len(calltypes) > 1:
            self.crnt_block = block
            self.matches = matches
            array_assigns = {}
            self.array_assigns = array_assigns
            for instr in block.body:
                is_array_expr = (
                    isinstance(instr, ir.Assign)
                    and isinstance(instr.value, ir.Expr)
                    and isinstance(typemap.get(instr.target.name, None),
                                   types.Array)
                )
                if is_array_expr:
                    operands = self._get_operands(instr.value)
                    if operands:
                        target_name = instr.target.name
                        array_assigns[target_name] = instr, operands
                        if operands.intersection(array_assigns.keys()):
                            matches.append(target_name)
        return len(matches) > 0

    def apply(self):
        replace_map = {}
        dead_vars = set()
        for match in self.matches:
            instr, operands = self.array_assigns[match]
            arr_inps = []
            arr_expr = instr.value.fn, arr_inps
            new_expr = ir.Expr(op='arrayexpr',
                               loc=instr.value.loc,
                               expr=arr_expr)
            new_instr = ir.Assign(new_expr, instr.target, instr.loc)
            replace_map[instr] = new_instr
            for operand in instr.value.list_vars():
                if operand.name in self.array_assigns:
                    child_assign, child_operands = self.array_assigns[
                        operand.name]
                    child_expr = child_assign.value
                    arr_inps.append((child_expr.fn, child_expr.list_vars()))
                    if child_assign.target.is_temp:
                        dead_vars.add(child_assign.target)
                        replace_map[child_assign] = None
                else:
                    arr_inps.append(operand)
        raise NotImplementedError("Development frontier.")
