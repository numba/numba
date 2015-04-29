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
        replace_map = {}
        dead_vars = set()
        used_vars = set()
        for match in self.matches:
            instr = self.array_assigns[match]
            arr_inps = []
            arr_expr = instr.value.fn, arr_inps
            new_expr = ir.Expr(op='arrayexpr',
                               loc=instr.value.loc,
                               expr=arr_expr)
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
